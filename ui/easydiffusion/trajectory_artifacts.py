"""Bounded background persistence for trajectory checkpoint artefacts.

The writer owns disk I/O only. GPU-to-CPU snapshotting happens synchronously at
an explicitly selected checkpoint before a job is submitted, ensuring the
writer never retains a live CUDA tensor or computation graph.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import tempfile
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional


log = logging.getLogger(__name__)
_SENTINEL = object()
_PREVIEW_FORMATS = {
    "jpeg": ("JPEG", "jpg"),
    "jpg": ("JPEG", "jpg"),
    "png": ("PNG", "png"),
    "webp": ("WEBP", "webp"),
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_latent_saver(latent: Any, path: str) -> None:
    """Persist one CPU tensor without pickle using safetensors."""

    from safetensors.torch import save_file

    save_file({"latent": latent}, path)


def _default_preview_saver(image: Any, path: str, image_format: str, quality: int) -> None:
    pil_format, _ = _PREVIEW_FORMATS[image_format]
    kwargs = {}
    if pil_format in ("JPEG", "WEBP"):
        kwargs["quality"] = quality
    image.save(path, format=pil_format, **kwargs)


def snapshot_tensor_to_cpu(x_samples: Any) -> Any:
    """Own a detached contiguous CPU copy of a live tensor-like value.

    The explicit clone makes ownership unambiguous even if a future CPU backend
    passes an already-CPU tensor. The returned object has no reference to the
    live autograd graph/device storage.
    """

    if x_samples is None:
        raise ValueError("cannot persist a null latent")
    try:
        return x_samples.detach().cpu().clone().contiguous()
    except AttributeError as exc:
        raise TypeError("latent persistence requires a torch-compatible tensor") from exc


class ArtifactWriter:
    """Serialize checkpoint artefacts through a bounded single writer thread.

    Backpressure policy is intentionally conservative: ``submit`` blocks when
    the queue is full. Requested checkpoints are never silently dropped. Disk
    failures are reported per checkpoint through ``on_result`` and the worker
    continues draining the queue so ``close`` cannot deadlock.
    """

    def __init__(
        self,
        run_dir: str,
        preview_format: str = "jpeg",
        preview_quality: int = 75,
        queue_size: int = 2,
        storage_budget_mb: Optional[float] = None,
        on_result: Optional[Callable[[str, Dict[str, Any], Optional[str]], None]] = None,
        latent_saver: Optional[Callable[[Any, str], None]] = None,
        preview_saver: Optional[Callable[[Any, str, str, int], None]] = None,
    ) -> None:
        preview_format = str(preview_format).lower()
        if preview_format not in _PREVIEW_FORMATS:
            raise ValueError(f"unsupported trajectory preview format: {preview_format!r}")
        if not isinstance(queue_size, int) or isinstance(queue_size, bool) or queue_size <= 0:
            raise ValueError("trajectory writer queue_size must be a positive integer")
        if not isinstance(preview_quality, int) or isinstance(preview_quality, bool) or not 1 <= preview_quality <= 100:
            raise ValueError("trajectory preview_quality must be an integer in 1..100")
        if storage_budget_mb is not None:
            storage_budget_mb = float(storage_budget_mb)
            if storage_budget_mb <= 0:
                raise ValueError("trajectory storage_budget_mb must be positive when set")

        self.run_dir = os.path.abspath(run_dir)
        self.preview_format = preview_format
        self.preview_quality = preview_quality
        self.storage_budget_bytes = (
            None if storage_budget_mb is None else max(1, int(storage_budget_mb * 1024 * 1024))
        )
        self.on_result = on_result
        self.latent_saver = latent_saver or _default_latent_saver
        self.preview_saver = preview_saver or _default_preview_saver
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._bytes_committed = 0
        self._closed = False
        self._callback_errors: List[str] = []
        self._thread = threading.Thread(
            target=self._worker,
            name="trajectory-artifact-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def bytes_committed(self) -> int:
        return self._bytes_committed

    def submit(
        self,
        checkpoint_id: str,
        step: int,
        latent: Any = None,
        previews: Optional[Iterable[Any]] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("trajectory artifact writer is already closed")
        preview_list = list(previews or [])
        if latent is None and not preview_list:
            raise ValueError("artifact job contains neither latent nor preview")
        self._queue.put(
            {
                "checkpoint_id": str(checkpoint_id),
                "step": int(step),
                "latent": latent,
                "previews": preview_list,
            },
            block=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._queue.join()
        self._queue.put(_SENTINEL)
        self._thread.join()
        self._closed = True
        if self._callback_errors:
            raise RuntimeError("; ".join(self._callback_errors))

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is _SENTINEL:
                self._queue.task_done()
                return
            try:
                result = self._write_job(job)
                self._notify(job["checkpoint_id"], result, None)
            except Exception as exc:
                self._notify(job["checkpoint_id"], {}, f"{type(exc).__name__}: {exc}")
            finally:
                # Drop references to potentially large CPU tensors/images before
                # acknowledging the queue item.
                job = None
                self._queue.task_done()

    def _notify(self, checkpoint_id: str, result: Dict[str, Any], error: Optional[str]) -> None:
        if self.on_result is None:
            return
        try:
            self.on_result(checkpoint_id, result, error)
        except Exception as exc:
            # A manifest callback must never kill the writer thread, but it must
            # also not fail invisibly. ``close`` will surface the accumulated
            # error to the recorder after all queued writes have drained.
            message = f"artifact result callback failed for {checkpoint_id}: {type(exc).__name__}: {exc}"
            self._callback_errors.append(message)
            log.exception(message)

    def _write_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint_id = job["checkpoint_id"]
        short_id = checkpoint_id.split("-")[0]
        step = job["step"]
        staged: List[Dict[str, Any]] = []

        try:
            latent = job.get("latent")
            if latent is not None:
                final_dir = os.path.join(self.run_dir, "latents")
                final_name = f"step-{step:05d}-{short_id}.safetensors"
                staged.append(
                    self._stage_file(
                        final_dir,
                        final_name,
                        lambda path: self.latent_saver(latent, path),
                        kind="latent",
                    )
                )

            for index, image in enumerate(job.get("previews") or []):
                final_dir = os.path.join(self.run_dir, "previews")
                _, extension = _PREVIEW_FORMATS[self.preview_format]
                final_name = f"step-{step:05d}-{index:02d}-{short_id}.{extension}"
                staged.append(
                    self._stage_file(
                        final_dir,
                        final_name,
                        lambda path, image=image: self.preview_saver(
                            image, path, self.preview_format, self.preview_quality
                        ),
                        kind="preview",
                        index=index,
                    )
                )

            job_bytes = sum(item["bytes"] for item in staged)
            if self.storage_budget_bytes is not None and self._bytes_committed + job_bytes > self.storage_budget_bytes:
                raise RuntimeError(
                    "trajectory storage budget exceeded: "
                    f"committed={self._bytes_committed} bytes, job={job_bytes} bytes, "
                    f"budget={self.storage_budget_bytes} bytes"
                )

            for item in staged:
                os.replace(item["temp_path"], item["final_path"])
                item["committed"] = True

            self._bytes_committed += job_bytes
            latent_result = None
            preview_results = []
            for item in staged:
                metadata = {
                    "path": os.path.relpath(item["final_path"], self.run_dir).replace(os.sep, "/"),
                    "sha256": item["sha256"],
                    "bytes": item["bytes"],
                }
                if item["kind"] == "latent":
                    metadata["format"] = "safetensors"
                    metadata["tensor_key"] = "latent"
                    latent_result = metadata
                else:
                    metadata["format"] = self.preview_format
                    metadata["index"] = item["index"]
                    preview_results.append(metadata)

            return {
                "latent": latent_result,
                "previews": preview_results,
                "bytes": job_bytes,
            }
        except Exception:
            # If committing one member of a multi-file checkpoint fails, remove
            # any files already committed by this job as well as remaining temp
            # files. The manifest will then correctly report the whole
            # checkpoint as failed instead of pointing at a partial artefact set.
            for item in staged:
                path = item["final_path"] if item.get("committed") else item["temp_path"]
                try:
                    os.unlink(path)
                except OSError:
                    pass
            raise

    def _stage_file(
        self,
        final_dir: str,
        final_name: str,
        saver: Callable[[str], None],
        kind: str,
        index: Optional[int] = None,
    ) -> Dict[str, Any]:
        os.makedirs(final_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".trajectory-artifact-", suffix=".tmp", dir=final_dir)
        os.close(fd)
        try:
            saver(temp_path)
            file_size = os.path.getsize(temp_path)
            checksum = _sha256_file(temp_path)
            return {
                "kind": kind,
                "index": index,
                "temp_path": temp_path,
                "final_path": os.path.join(final_dir, final_name),
                "bytes": file_size,
                "sha256": checksum,
                "committed": False,
            }
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
