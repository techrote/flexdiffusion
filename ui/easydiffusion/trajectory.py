"""Trajectory capture primitives for FlexDiffusion.

Schedule parsing and manifest management remain dependency-light. Heavy tensor
serialization is delegated lazily to ``trajectory_artifacts`` only when an
opt-in capture actually requests persisted artefacts.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple


TRAJECTORY_FORMAT_VERSION = 2
_VALID_PERSISTENCE_MODES = {"preview", "latent", "hybrid"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)(?:\s*:\s*(\d+))?$")
_INTEGER_RE = re.compile(r"^\d+$")
_PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")


class CaptureScheduleError(ValueError):
    """Raised when a trajectory capture schedule is invalid."""


def compile_capture_schedule(spec: str, total_steps: int) -> Tuple[int, ...]:
    """Compile a user schedule into sorted, unique, one-based step numbers.

    Supported forms may be mixed with commas:

    * exact steps: ``5,10,20``
    * inclusive ranges: ``1-10`` or ``1-10:2``
    * percentages: ``10%,25%,50%,100%``

    User-facing steps are intentionally one-based. Easy Diffusion/sdkit
    callbacks are currently zero-based, so :class:`TrajectoryRecorder` maps
    callback index ``i`` to displayed step ``i + 1`` before matching.

    Percentage checkpoints use ``ceil(total_steps * percentage / 100)`` so a
    positive percentage never resolves to a non-existent step zero. ``100%``
    always resolves to ``total_steps``.
    """

    if not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0:
        raise CaptureScheduleError("total_steps must be a positive integer")

    if spec is None:
        return ()
    if not isinstance(spec, str):
        raise CaptureScheduleError("capture schedule must be a string")

    spec = spec.strip()
    if not spec:
        return ()

    steps = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise CaptureScheduleError("capture schedule contains an empty item")

        percent_match = _PERCENT_RE.match(token)
        if percent_match:
            percent = float(percent_match.group(1))
            if percent <= 0 or percent > 100:
                raise CaptureScheduleError(f"percentage must be > 0 and <= 100: {token!r}")
            resolved = int(math.ceil(total_steps * percent / 100.0))
            steps.add(max(1, min(total_steps, resolved)))
            continue

        range_match = _RANGE_RE.match(token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            stride = int(range_match.group(3) or 1)
            if stride <= 0:
                raise CaptureScheduleError(f"range stride must be positive: {token!r}")
            if start < 1 or end < 1 or start > end:
                raise CaptureScheduleError(f"invalid capture range: {token!r}")
            if end > total_steps:
                raise CaptureScheduleError(
                    f"capture range ends after total_steps ({total_steps}): {token!r}"
                )
            steps.update(range(start, end + 1, stride))
            continue

        if _INTEGER_RE.match(token):
            step = int(token)
            if step < 1 or step > total_steps:
                raise CaptureScheduleError(f"capture step {step} is outside 1..{total_steps}")
            steps.add(step)
            continue

        raise CaptureScheduleError(f"unrecognised capture schedule item: {token!r}")

    return tuple(sorted(steps))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: Optional[str], fallback: str) -> str:
    value = (value or "").strip()
    if not value:
        value = fallback
    value = _SAFE_NAME_RE.sub("-", value).strip("-._")
    return value or fallback


def _normalise_config(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "dict"):
        return dict(config.dict())
    raise TypeError("trajectory config must be a mapping or expose dict()")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _tensor_metadata(x_samples: Any) -> Dict[str, Any]:
    """Extract tensor-like metadata without importing torch."""

    if x_samples is None:
        return {"shape": None, "dtype": None, "device": None}

    shape = getattr(x_samples, "shape", None)
    if shape is not None:
        try:
            shape = [int(x) for x in shape]
        except (TypeError, ValueError):
            shape = [str(x) for x in shape]

    dtype = getattr(x_samples, "dtype", None)
    device = getattr(x_samples, "device", None)
    return {
        "shape": shape,
        "dtype": str(dtype) if dtype is not None else None,
        "device": str(device) if device is not None else None,
    }


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".trajectory-", suffix=".json.tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


class TrajectoryRecorder:
    """Incremental recorder for one diffusion trajectory.

    Selected live CUDA tensors are detached and copied to CPU synchronously at
    the sampler callback boundary. Disk I/O then runs through a bounded worker
    queue. This prevents an unbounded chain of GPU tensors while also ensuring
    requested checkpoints are never silently dropped.
    """

    def __init__(
        self,
        config: Any,
        total_steps: int,
        run_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = _normalise_config(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.total_steps = total_steps
        self.run_id = str(uuid.uuid4())
        self._recorded_steps = set()
        self._checkpoint_by_id: Dict[str, Dict[str, Any]] = {}
        self._closed = False
        self._error_messages = []
        self._lock = threading.RLock()
        self._artifact_writer = None
        self.capture_steps: Tuple[int, ...] = ()
        self.run_dir: Optional[str] = None
        self.manifest_path: Optional[str] = None
        self.manifest: Dict[str, Any] = {}
        self.persistence_mode = "preview"

        if not self.enabled:
            return

        mode = str(self.config.get("persistence_mode", "preview")).lower()
        if mode not in _VALID_PERSISTENCE_MODES:
            raise ValueError(
                f"unsupported trajectory persistence mode {mode!r}; "
                f"expected one of {sorted(_VALID_PERSISTENCE_MODES)}"
            )
        self.persistence_mode = mode
        self.config["persistence_mode"] = mode

        self.capture_steps = compile_capture_schedule(
            str(self.config.get("capture_schedule", "")), total_steps
        )

        max_checkpoints = self.config.get("max_checkpoints")
        if max_checkpoints is not None:
            max_checkpoints = int(max_checkpoints)
            if max_checkpoints <= 0:
                raise ValueError("max_checkpoints must be positive when set")
            if len(self.capture_steps) > max_checkpoints:
                raise ValueError(
                    f"capture schedule requests {len(self.capture_steps)} checkpoints, "
                    f"exceeding max_checkpoints={max_checkpoints}"
                )

        output_root = self.config.get("output_root") or os.path.join(os.getcwd(), "outputs", "trajectories")
        project_name = _safe_name(self.config.get("project_name"), "trajectory")
        run_slug = f"{project_name}-{self.run_id[:8]}"
        self.run_dir = os.path.abspath(os.path.join(str(output_root), run_slug))
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")

        created_at = _utc_now()
        self.manifest = {
            "format_version": TRAJECTORY_FORMAT_VERSION,
            "run_id": self.run_id,
            "status": "recording",
            "created_at": created_at,
            "updated_at": created_at,
            "completed_at": None,
            "total_steps": total_steps,
            "capture_steps": list(self.capture_steps),
            "config": _json_safe(self.config),
            "run_metadata": _json_safe(dict(run_metadata or {})),
            "checkpoints": [],
            "artifact_bytes": 0,
            "errors": [],
        }
        self._flush_manifest()

        if self.capture_steps:
            from easydiffusion.trajectory_artifacts import ArtifactWriter

            try:
                self._artifact_writer = ArtifactWriter(
                    run_dir=self.run_dir,
                    preview_format=self.config.get("preview_format", "jpeg"),
                    preview_quality=int(self.config.get("preview_quality", 75)),
                    queue_size=int(self.config.get("writer_queue_size", 2)),
                    storage_budget_mb=self.config.get("storage_budget_mb"),
                    on_result=self._on_artifact_result,
                )
            except Exception as exc:
                # The initial manifest already exists at this point. Finalise it
                # explicitly so a configuration/initialisation failure never
                # leaves a misleading forever-"recording" run behind.
                with self._lock:
                    message = f"artifact writer initialisation failed: {type(exc).__name__}: {exc}"
                    self._error_messages.append(message)
                    self.manifest["status"] = "initialisation_failed"
                    self.manifest["errors"] = list(self._error_messages)
                    self.manifest["updated_at"] = _utc_now()
                    self.manifest["completed_at"] = _utc_now()
                    self._closed = True
                    self._flush_manifest_locked()
                raise

    @property
    def wants_latent(self) -> bool:
        return self.persistence_mode in ("latent", "hybrid")

    @property
    def wants_preview(self) -> bool:
        return self.persistence_mode in ("preview", "hybrid")

    def wants_callback_index(self, callback_index: int) -> bool:
        if not self.enabled or self._closed:
            return False
        display_step = callback_index + 1
        with self._lock:
            return display_step in self.capture_steps and display_step not in self._recorded_steps

    def needs_preview_at(self, callback_index: int) -> bool:
        return self.wants_preview and self.wants_callback_index(callback_index)

    def record_step(
        self,
        callback_index: int,
        x_samples: Any,
        step_metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Record checkpoint metadata only.

        This compatibility/testing helper deliberately performs no artefact I/O.
        Production callback capture should use :meth:`capture_step`.
        """

        if not self.wants_callback_index(callback_index):
            return False
        checkpoint = self._append_checkpoint(
            callback_index,
            x_samples,
            step_metadata=step_metadata,
            artifact_status="metadata_only",
        )
        return checkpoint is not None

    def capture_step(
        self,
        callback_index: int,
        x_samples: Any,
        preview_images: Optional[Any] = None,
        step_metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Snapshot and queue one requested checkpoint for persistence."""

        if not self.wants_callback_index(callback_index):
            return False

        latent_snapshot = None
        preview_snapshots = []

        if self.wants_latent:
            from easydiffusion.trajectory_artifacts import snapshot_tensor_to_cpu

            latent_snapshot = snapshot_tensor_to_cpu(x_samples)

        if self.wants_preview:
            if preview_images is None:
                raise ValueError("trajectory preview persistence requested but no decoded preview was supplied")
            preview_snapshots = [image.copy() for image in preview_images]

        checkpoint = self._append_checkpoint(
            callback_index,
            x_samples,
            step_metadata=step_metadata,
            artifact_status="queued",
        )
        if checkpoint is None:
            return False

        if self._artifact_writer is None:
            raise RuntimeError("trajectory artifact writer is unavailable")

        try:
            self._artifact_writer.submit(
                checkpoint["checkpoint_id"],
                checkpoint["step"],
                latent=latent_snapshot,
                previews=preview_snapshots,
            )
        except Exception as exc:
            self._on_artifact_result(
                checkpoint["checkpoint_id"],
                {},
                f"{type(exc).__name__}: {exc}",
            )
            raise
        return True

    def note_error(self, message: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._error_messages.append(str(message))
            self.manifest["errors"] = list(self._error_messages)
            self.manifest["updated_at"] = _utc_now()
            self._flush_manifest_locked()

    def finish(self, status: str = "complete", error: Optional[str] = None) -> None:
        if not self.enabled or self._closed:
            return

        if self._artifact_writer is not None:
            try:
                self._artifact_writer.close()
            except Exception as exc:
                self.note_error(f"artifact writer close failed: {type(exc).__name__}: {exc}")

        with self._lock:
            if error:
                self._error_messages.append(str(error))
            if self._error_messages and status == "complete":
                status = "complete_with_capture_errors"
            self.manifest["status"] = status
            self.manifest["errors"] = list(self._error_messages)
            self.manifest["updated_at"] = _utc_now()
            self.manifest["completed_at"] = _utc_now()
            self._flush_manifest_locked()
            self._closed = True

    def _append_checkpoint(
        self,
        callback_index: int,
        x_samples: Any,
        step_metadata: Optional[Mapping[str, Any]],
        artifact_status: str,
    ) -> Optional[Dict[str, Any]]:
        display_step = callback_index + 1
        with self._lock:
            if display_step in self._recorded_steps:
                return None
            checkpoint = {
                "checkpoint_id": str(uuid.uuid4()),
                "callback_index": int(callback_index),
                "step": display_step,
                "total_steps": self.total_steps,
                "captured_at": _utc_now(),
                "tensor": _tensor_metadata(x_samples),
                "artifact_status": artifact_status,
                "latent": None,
                "previews": [],
                "artifact_bytes": 0,
                "resume_fidelity": "UNCLASSIFIED",
                "metadata": _json_safe(dict(step_metadata or {})),
            }
            self.manifest["checkpoints"].append(checkpoint)
            self._checkpoint_by_id[checkpoint["checkpoint_id"]] = checkpoint
            self._recorded_steps.add(display_step)
            self.manifest["updated_at"] = _utc_now()
            self._flush_manifest_locked()
            return checkpoint

    def _on_artifact_result(
        self,
        checkpoint_id: str,
        result: Dict[str, Any],
        error: Optional[str],
    ) -> None:
        with self._lock:
            checkpoint = self._checkpoint_by_id.get(checkpoint_id)
            if checkpoint is None:
                raise KeyError(f"unknown trajectory checkpoint id: {checkpoint_id}")

            if error:
                checkpoint["artifact_status"] = "error"
                checkpoint["artifact_error"] = error
                message = f"checkpoint step {checkpoint['step']} artifact persistence failed: {error}"
                self._error_messages.append(message)
                self.manifest["errors"] = list(self._error_messages)
            else:
                checkpoint["artifact_status"] = "persisted"
                checkpoint["latent"] = result.get("latent")
                checkpoint["previews"] = result.get("previews", [])
                checkpoint["artifact_bytes"] = int(result.get("bytes", 0))
                self.manifest["artifact_bytes"] = int(self.manifest.get("artifact_bytes", 0)) + checkpoint[
                    "artifact_bytes"
                ]

            self.manifest["updated_at"] = _utc_now()
            self._flush_manifest_locked()

    def _flush_manifest(self) -> None:
        with self._lock:
            self._flush_manifest_locked()

    def _flush_manifest_locked(self) -> None:
        if self.manifest_path is None:
            return
        _atomic_write_json(self.manifest_path, self.manifest)
