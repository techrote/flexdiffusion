"""Transient intermediate-output capture helpers for classic SD1.x renders.

This module deliberately contains no torch import so its step-selection and
representation semantics can be unit-tested in the dependency-light Trajectory
Lab CI job.

For the fixed-step k-diffusion samplers used by Easy Diffusion classic, callback
index ``i`` observes the current solver latent *before* integration update ``i``.
Consequently callback index 0 is the initial latent (zero completed steps),
callback index N is the state after N completed integration updates, and the
sampler's normal return value is the only authoritative state after the final
step.

FlexDiffusion's current UI uses sparse, printer-style capture expressions such as
``10, 16, 20-24``. Denoised estimates are retained only at the requested
intermediate callback boundaries. Optionally, raw solver states are also kept at
the first and last requested callback boundaries. The ordinary sampler return
remains the authoritative final image.

The legacy ``Output After Step`` protocol is retained internally for backwards
compatibility. The UI transports sparse selections through a compact
``intermediate_representation`` marker so the pinned Easy Diffusion request
model does not need a breaking schema change.
"""

import re


FIXED_STEP_K_DIFFUSION_SAMPLERS = frozenset(
    {
        "euler_a",
        "euler",
        "lms",
        "heun",
        "dpm2",
        "dpm2_a",
        "dpmpp_2s_a",
        "dpmpp_2m",
        "dpmpp_sde",
    }
)

INTERMEDIATE_REPRESENTATIONS = frozenset({"denoised", "solver_state", "both"})
SPARSE_REPRESENTATION_PREFIX = "sparse:"
_REPRESENTATION_ORDER = {"denoised": 0, "solver_state": 1}
_SINGLE_STEP_RE = re.compile(r"^\d+$")
_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_SPARSE_REPRESENTATION_RE = re.compile(r"^sparse:(.+)\|endpoints=([01])$")


def _validate_total_steps(total_steps):
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    return total_steps


def parse_capture_step_expression(value, total_steps):
    """Expand printer-style step syntax into a sorted unique integer list.

    ``10, 16, 20-24`` becomes ``[10, 16, 20, 21, 22, 23, 24]``.
    Ranges are inclusive and must be ascending. Values outside ``1..total_steps``
    are rejected rather than silently clamped.
    """

    total_steps = _validate_total_steps(total_steps)
    if value is None:
        return [total_steps]
    if isinstance(value, bool):
        raise ValueError("capture steps must be integers or printer-style ranges")

    if isinstance(value, (list, tuple, set, frozenset)):
        values = []
        for item in value:
            if isinstance(item, bool):
                raise ValueError("capture steps must contain integers")
            try:
                step = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError("capture steps must contain integers") from exc
            if step < 1 or step > total_steps:
                raise ValueError(f"capture step {step} is outside 1..{total_steps}")
            values.append(step)
        if not values:
            raise ValueError("at least one capture step is required")
        return sorted(set(values))

    text = str(value).strip()
    if not text:
        raise ValueError("at least one capture step is required")

    values = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("empty capture-step item")

        if _SINGLE_STEP_RE.match(token):
            start = end = int(token)
        else:
            match = _RANGE_RE.match(token)
            if not match:
                raise ValueError(f"invalid capture-step item: {token!r}")
            start, end = int(match.group(1)), int(match.group(2))
            if start > end:
                raise ValueError(f"capture-step range must be ascending: {token!r}")

        if start < 1 or end > total_steps:
            raise ValueError(f"capture-step item {token!r} is outside 1..{total_steps}")
        values.update(range(start, end + 1))

    return sorted(values)


def format_capture_steps(steps):
    """Compress sorted capture steps back to printer-style syntax."""

    values = sorted(set(int(step) for step in steps))
    if not values:
        return ""

    groups = []
    start = previous = values[0]
    for step in values[1:]:
        if step == previous + 1:
            previous = step
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = step
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def encode_sparse_representation(capture_steps, total_steps, include_endpoint_solver_states=False):
    steps = parse_capture_step_expression(capture_steps, total_steps)
    expression = format_capture_steps(steps)
    return f"{SPARSE_REPRESENTATION_PREFIX}{expression}|endpoints={1 if include_endpoint_solver_states else 0}"


def decode_sparse_representation(value, total_steps):
    if value is None:
        return None
    text = str(value).strip().lower()
    match = _SPARSE_REPRESENTATION_RE.match(text)
    if not match:
        return None
    steps = parse_capture_step_expression(match.group(1), total_steps)
    return steps, match.group(2) == "1"


def normalize_output_after_step(value, total_steps):
    _validate_total_steps(total_steps)

    if value is None:
        return total_steps
    if isinstance(value, bool):
        raise ValueError("output_after_step must be an integer")

    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("output_after_step must be an integer") from exc

    return max(1, min(value, total_steps))


def normalize_intermediate_representation(value):
    if value is None:
        return "both"
    value = str(value).strip().lower()
    if value in INTERMEDIATE_REPRESENTATIONS:
        return value
    if value.startswith(SPARSE_REPRESENTATION_PREFIX):
        # Full bounds validation needs total_steps and is therefore performed by
        # OutputAfterStepCollector. Validate only the transport envelope here.
        if not _SPARSE_REPRESENTATION_RE.match(value):
            raise ValueError("malformed sparse intermediate representation")
        return value
    valid = ", ".join(sorted(INTERMEDIATE_REPRESENTATIONS))
    raise ValueError(
        f"intermediate_representation must be one of: {valid}, or a FlexDiffusion sparse marker"
    )


def flatten_output_step_results(results, seed, total_steps):
    """Return (image, seed, metadata) tuples in chronological display order."""
    flattened = []
    ordered = sorted(
        results or [],
        key=lambda item: (
            int(item["step"]),
            _REPRESENTATION_ORDER.get(item.get("representation"), 99),
        ),
    )
    for group in ordered:
        step = int(group["step"])
        representation = group.get("representation", "solver_state")
        for index, image in enumerate(group.get("images") or []):
            flattened.append(
                (
                    image,
                    int(seed) + index,
                    {
                        "output_step": step,
                        "total_steps": int(total_steps),
                        "is_intermediate": True,
                        "intermediate_representation": representation,
                    },
                )
            )
    return flattened


def _owned_cpu_snapshot(tensor):
    return tensor.detach().cpu().clone().contiguous()


class OutputAfterStepCollector:
    """Own selected k-diffusion callback representations on CPU until decode.

    Legacy representation values retain their original behaviour. A sparse
    marker captures denoised estimates only at explicitly requested steps and,
    when requested, raw solver states only at the first/last requested callback
    boundaries.
    """

    def __init__(self, start_step, total_steps, representation="both"):
        self.total_steps = int(total_steps)
        self.start_step = normalize_output_after_step(start_step, self.total_steps)
        self.representation = normalize_intermediate_representation(representation)
        self._snapshots = []
        self._captured_steps = set()

        sparse = decode_sparse_representation(self.representation, self.total_steps)
        if sparse is None:
            legacy_steps = set(range(self.start_step, self.total_steps))
            self.capture_steps = sorted(legacy_steps)
            self._denoised_steps = legacy_steps if self.representation in ("denoised", "both") else set()
            self._solver_steps = legacy_steps if self.representation in ("solver_state", "both") else set()
        else:
            capture_steps, include_endpoints = sparse
            self.capture_steps = capture_steps
            callback_steps = {step for step in capture_steps if step < self.total_steps}
            self._denoised_steps = set(callback_steps)
            self._solver_steps = set()
            if include_endpoints and callback_steps:
                first_requested = capture_steps[0]
                last_requested = capture_steps[-1]
                if first_requested < self.total_steps:
                    self._solver_steps.add(first_requested)
                if last_requested < self.total_steps:
                    self._solver_steps.add(last_requested)

    @property
    def enabled(self):
        return bool(self._denoised_steps or self._solver_steps)

    def wants_callback_index(self, callback_index):
        if not self.enabled:
            return False
        if isinstance(callback_index, bool):
            return False

        try:
            callback_index = int(callback_index)
        except (TypeError, ValueError):
            return False

        return callback_index in self._denoised_steps or callback_index in self._solver_steps

    def capture_step(self, callback_index, solver_state, denoised=None):
        callback_index = int(callback_index)
        if not self.wants_callback_index(callback_index):
            return False
        if callback_index in self._captured_steps:
            return False

        if callback_index in self._denoised_steps:
            if denoised is None:
                raise ValueError("k-diffusion callback did not provide a denoised estimate")
            self._snapshots.append(
                (callback_index, "denoised", _owned_cpu_snapshot(denoised))
            )

        if callback_index in self._solver_steps:
            self._snapshots.append(
                (callback_index, "solver_state", _owned_cpu_snapshot(solver_state))
            )

        self._captured_steps.add(callback_index)
        return True

    def pop_snapshots(self):
        snapshots = self._snapshots
        self._snapshots = []
        self._captured_steps = set()
        return snapshots
