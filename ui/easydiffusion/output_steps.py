"""Transient intermediate-output capture helpers for classic SD1.x renders.

This module deliberately contains no torch import so its step-selection semantics
can be unit-tested in the dependency-light Trajectory Lab CI job.

For the fixed-step k-diffusion samplers used by Easy Diffusion classic, callback
index ``i`` observes the current latent *before* integration update ``i``.
Consequently callback index 0 is the initial latent (zero completed steps),
callback index N is the state after N completed integration updates, and the
sampler's normal return value is the only authoritative state after the final
step. ``Output After Step`` therefore captures callback indexes
``start_step .. total_steps - 1`` and uses the ordinary final render for
``total_steps``.
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


def flatten_output_step_results(results, seed, total_steps):
    """Return (image, seed, metadata) tuples in chronological step order."""
    flattened = []
    for group in sorted(results or [], key=lambda item: int(item["step"])):
        step = int(group["step"])
        for index, image in enumerate(group.get("images") or []):
            flattened.append(
                (
                    image,
                    int(seed) + index,
                    {
                        "output_step": step,
                        "total_steps": int(total_steps),
                        "is_intermediate": True,
                    },
                )
            )
    return flattened


class OutputAfterStepCollector:
    """Own selected callback latents on CPU until post-sampling decode."""

    def __init__(self, start_step, total_steps):
        self.total_steps = int(total_steps)
        self.start_step = normalize_output_after_step(start_step, self.total_steps)
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

    def capture_step(self, callback_index, x_samples):
        callback_index = int(callback_index)
        if not self.wants_callback_index(callback_index):
            return False
        if callback_index in self._captured_steps:
            return False

        snapshot = x_samples.detach().cpu().clone().contiguous()
        self._snapshots.append((callback_index, snapshot))
        self._captured_steps.add(callback_index)
        return True

    def pop_snapshots(self):
        snapshots = self._snapshots
        self._snapshots = []
        self._captured_steps = set()
        return snapshots
