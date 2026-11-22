"""
(OUTDATED DOC)
A runtime that runs on a specific device (in a thread).

It can run various tasks like image generation, image filtering, model merge etc by using that thread-local context.

This creates an `sdkit.Context` that's bound to the device specified while calling the `init()` function.
"""

context = None


def init(device):
    """
    Initializes the fields that will be bound to this runtime's context, and sets the current torch device
    """

    global context

    from easydiffusion import device_manager
    from easydiffusion.backend_manager import backend
    from easydiffusion.app import getConfig

    context = backend.create_context()

    context.stop_processing = False
    context.temp_images = {}
    context.partial_x_samples = None
    context.model_load_errors = {}
    context.enable_codeformer = True

    device_manager.device_init(context, device)


def set_vram_optimizations(context, requested_level=None):
    """Set the effective profile for THIS worker and task.

    The global config is only a default for startup; queued requests may each
    have a different profile, including an automatically derated one.
    """
    from easydiffusion.app import getConfig

    level = requested_level if requested_level is not None else getConfig().get("vram_usage_level", "balanced")
    if level not in ("high", "balanced", "low"):
        raise ValueError(f"Unknown GPU memory profile: {level!r}")

    if hasattr(context, "vram_usage_level") and level != context.vram_usage_level:
        context.vram_usage_level = level
        return True

    return False
