"""Trajectory capture primitives for FlexDiffusion.

This module is deliberately dependency-light.  Schedule parsing and manifest
management must be testable without importing torch or starting an Easy
Diffusion backend.  Actual latent/preview persistence is added in the next
stage; the recorder introduced here establishes stable run/checkpoint metadata
and the callback interception contract.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


TRAJECTORY_FORMAT_VERSION = 1
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

    User-facing steps are intentionally one-based.  Easy Diffusion/sdkit
    callbacks are currently zero-based, so :class:`TrajectoryRecorder` maps
    callback index ``i`` to displayed step ``i + 1`` before matching.

    Percentage checkpoints use ``ceil(total_steps * percentage / 100)`` so a
    positive percentage never resolves to a non-existent step zero.  ``100%``
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
                raise CaptureScheduleError(
                    f"capture step {step} is outside 1..{total_steps}"
                )
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
    """Incremental manifest recorder for one diffusion trajectory.

    The current implementation intentionally records metadata only.  It is the
    stable scaffold onto which bounded asynchronous latent/preview persistence
    will be attached.  No live tensor references are retained.
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
        self._closed = False
        self._error_messages = []
        self.capture_steps: Tuple[int, ...] = ()
        self.run_dir: Optional[str] = None
        self.manifest_path: Optional[str] = None
        self.manifest: Dict[str, Any] = {}

        if not self.enabled:
            return

        mode = str(self.config.get("persistence_mode", "preview")).lower()
        if mode not in _VALID_PERSISTENCE_MODES:
            raise ValueError(
                f"unsupported trajectory persistence mode {mode!r}; "
                f"expected one of {sorted(_VALID_PERSISTENCE_MODES)}"
            )
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

        output_root = self.config.get("output_root") or os.path.join(
            os.getcwd(), "outputs", "trajectories"
        )
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
            "errors": [],
        }
        self._flush_manifest()

    def wants_callback_index(self, callback_index: int) -> bool:
        if not self.enabled or self._closed:
            return False
        return callback_index + 1 in self.capture_steps

    def record_step(
        self,
        callback_index: int,
        x_samples: Any,
        step_metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Record one requested checkpoint.

        Returns ``True`` if a new checkpoint was recorded.  The recorder stores
        only tensor metadata in this scaffold and never retains ``x_samples``.
        """

        if not self.wants_callback_index(callback_index):
            return False

        display_step = callback_index + 1
        if display_step in self._recorded_steps:
            return False

        checkpoint = {
            "checkpoint_id": str(uuid.uuid4()),
            "callback_index": int(callback_index),
            "step": display_step,
            "total_steps": self.total_steps,
            "captured_at": _utc_now(),
            "tensor": _tensor_metadata(x_samples),
            "latent": None,
            "preview": None,
            "resume_fidelity": "UNCLASSIFIED",
            "metadata": _json_safe(dict(step_metadata or {})),
        }
        self.manifest["checkpoints"].append(checkpoint)
        self._recorded_steps.add(display_step)
        self.manifest["updated_at"] = _utc_now()
        self._flush_manifest()
        return True

    def note_error(self, message: str) -> None:
        if not self.enabled:
            return
        message = str(message)
        self._error_messages.append(message)
        self.manifest["errors"] = list(self._error_messages)
        self.manifest["updated_at"] = _utc_now()
        self._flush_manifest()

    def finish(self, status: str = "complete", error: Optional[str] = None) -> None:
        if not self.enabled or self._closed:
            return
        if error:
            self._error_messages.append(str(error))
        if self._error_messages and status == "complete":
            status = "complete_with_capture_errors"
        self.manifest["status"] = status
        self.manifest["errors"] = list(self._error_messages)
        self.manifest["updated_at"] = _utc_now()
        self.manifest["completed_at"] = _utc_now()
        self._flush_manifest()
        self._closed = True

    def _flush_manifest(self) -> None:
        if self.manifest_path is None:
            return
        _atomic_write_json(self.manifest_path, self.manifest)
