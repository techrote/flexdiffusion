"""Conservative per-job VRAM guidance.

This is a risk heuristic, not an allocation forecast. It never changes the
requested resolution or silently changes a GPU into CPU-only rendering.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Optional


LEVEL_ORDER = ("low", "balanced", "high")
HEAVY_MODEL_HINTS = ("sdxl", "sd_xl", "flux", "pony", "sd3", "stable-diffusion-3", "z-image", "lumina", "qwen-image", "anima")


@dataclass(frozen=True)
class VramAdvice:
    requested: str
    recommended: str
    risk: str  # unknown, normal, elevated, high
    reason: str
    disable_live_preview: bool
    prefer_serial_outputs: bool


def advise_vram(
    requested: str,
    width: int,
    height: int,
    num_outputs: int,
    model_name: str,
    total_vram_gb: Optional[float],
    enabled: bool,
) -> VramAdvice:
    """Recommend a more conservative existing profile when the risk is high.

    Total VRAM is more stable than currently free VRAM: models already loaded
    by the renderer are not counted as available GPU memory.
    """
    if requested not in LEVEL_ORDER:
        raise ValueError(f"Unsupported GPU memory profile: {requested!r}")

    untouched = VramAdvice(requested, requested, "unknown", "VRAM preflight disabled or unavailable", False, False)
    if not enabled or total_vram_gb is None:
        return untouched

    try:
        total = float(total_vram_gb)
        pixels = int(width) * int(height)
        outputs = int(num_outputs)
    except (TypeError, ValueError, OverflowError):
        return untouched
    if not isfinite(total) or total <= 0 or pixels <= 0 or outputs <= 0:
        return untouched

    model = str(model_name or "").lower()
    large_model = any(hint in model for hint in HEAVY_MODEL_HINTS)
    risk = "normal"
    cap = "high"
    reason = "Current workload fits the conservative heuristic"

    # These thresholds intentionally err towards slower but safer settings.
    # They are not asserted minimum VRAM requirements for any model family.
    if (
        (total < 4.5 and pixels >= 768 * 768)
        or (total < 6.5 and pixels >= 1024 * 1024)
        or (total < 8.9 and large_model and pixels >= 768 * 768)
        or (total < 6.5 and outputs > 1 and pixels >= 512 * 512)
    ):
        risk, cap = "high", "low"
        reason = "Large image, demanding model and/or parallel outputs relative to installed VRAM"
    elif (
        (total < 4.5 and pixels >= 512 * 512)
        or (total < 8.9 and pixels >= 1024 * 1024)
        or (total < 8.9 and large_model)
        or (total < 8.9 and outputs > 1 and pixels >= 768 * 768)
    ):
        risk, cap = "elevated", "balanced"
        reason = "Workload may put pressure on available VRAM"

    recommended = LEVEL_ORDER[min(LEVEL_ORDER.index(requested), LEVEL_ORDER.index(cap))]
    return VramAdvice(
        requested=requested,
        recommended=recommended,
        risk=risk,
        reason=reason,
        disable_live_preview=(risk == "high"),
        prefer_serial_outputs=(risk == "high" and outputs > 1),
    )
