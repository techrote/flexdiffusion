"""Transient intermediate-output capture helpers for classic SD1.x renders.

This module deliberately contains no torch import so its step-selection and
representation semantics can be unit-tested in the dependency-light Trajectory
Lab CI job.

For the fixed-step k-diffusion samplers used by Easy Diffusion classic, callback
index ``i`` observes the current solver latent *before* integration update ``i``.
Consequently callback index 0 is the initial latent (zero completed steps),
callback index N is the state after N completed integration updates, and the
sampler's normal return value is the only authoritative state after the final
step. ``Output After Step`` therefore captures callback indexes
``start_step .. total_steps - 1`` and uses the ordinary final render for
``total_steps``.

k-diffusion's callback also exposes ``denoised``: the model's current clean-image
estimate at that same solver boundary. FlexDiffusion can retain either the raw
solver state, the denoised estimate, or both, then VAE-decode the selected CPU
snapshots after sampling has finished.
"""

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
_REPRESENTATION_ORDER = {"denoised": 0, "solver_state": 1}


def normalize_output_after_step(value, total_steps):
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")

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
    if value not in INTERMEDIATE_REPRESENTATIONS:
        valid = ", ".join(sorted(INTERMEDIATE_REPRESENTATIONS))
        raise ValueError(f"intermediate_representation must be one of: {valid}")
    return value


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
    """Own selected k-diffusion callback representations on CPU until decode."""

    def __init__(self, start_step, total_steps, representation="both"):
        self.total_steps = int(total_steps)
        self.start_step = normalize_output_after_step(start_step, self.total_steps)
        self.representation = normalize_intermediate_representation(representation)
        self._snapshots = []
        self._captured_steps = set()

    @property
    def enabled(self):
        return self.start_step < self.total_steps

    def wants_callback_index(self, callback_index):
        if not self.enabled:
            return False
        if isinstance(callback_index, bool):
            return False

        try:
            callback_index = int(callback_index)
        except (TypeError, ValueError):
            return False

        return self.start_step <= callback_index < self.total_steps

    def capture_step(self, callback_index, solver_state, denoised=None):
        callback_index = int(callback_index)
        if not self.wants_callback_index(callback_index):
            return False
        if callback_index in self._captured_steps:
            return False

        if self.representation in ("denoised", "both"):
            if denoised is None:
                raise ValueError("k-diffusion callback did not provide a denoised estimate")
            self._snapshots.append(
                (callback_index, "denoised", _owned_cpu_snapshot(denoised))
            )

        if self.representation in ("solver_state", "both"):
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
