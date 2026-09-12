from sdkit import Context

from easydiffusion.types import UserInitiatedStop
from easydiffusion.utils import log

from sdkit.utils import (
    diffusers_latent_samples_to_images,
    gc,
    img_to_base64_str,
    latent_samples_to_images,
)

opts = {}


def install_backend():
    pass


def start_backend():
    print("Started sdkit backend")


def stop_backend():
    pass


def uninstall_backend():
    pass


def is_installed():
    return True


def create_sdkit_context(use_diffusers):
    c = Context()
    c.test_diffusers = use_diffusers
    return c


def ping(timeout=1):
    return True


def load_model(context, model_type, **kwargs):
    from sdkit.models import load_model

    load_model(context, model_type, **kwargs)


def unload_model(context, model_type, **kwargs):
    from sdkit.models import unload_model

    unload_model(context, model_type, **kwargs)


def set_options(context, **kwargs):
    if "vae_tiling" in kwargs and context.test_diffusers:
        pipe = context.models["stable-diffusion"]["default"]
        vae_tiling = kwargs["vae_tiling"]

        if vae_tiling:
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
        else:
            if hasattr(pipe, "disable_vae_tiling"):
                pipe.disable_vae_tiling()

    for key in (
        "output_format",
        "output_quality",
        "output_lossless",
        "stream_image_progress",
        "stream_image_progress_interval",
    ):
        if key in kwargs:
            opts[key] = kwargs[key]


def generate_images(
    context: Context,
    callback=None,
    controlnet_filter=None,
    distilled_guidance_scale: float = 3.5,
    scheduler_name: str = "simple",
    output_type="pil",
    ref_images=None,
    trajectory=None,
    **req,
):
    from sdkit.generate import generate_images

    trajectory_recorder = None
    trajectory_status = "failed"

    if trajectory and trajectory.get("enabled", False):
        from easydiffusion.trajectory import TrajectoryRecorder

        if context.test_diffusers:
            raise RuntimeError("Trajectory capture is currently supported only by the classic sdkit path")

        total_steps = req["num_inference_steps"]
        if req.get("init_image") is not None:
            total_steps = int(req["num_inference_steps"] * req.get("prompt_strength", 0.8))

        model_paths = getattr(context, "model_paths", {}) or {}
        run_metadata = {
            "backend": "ed_classic",
            "model_path": model_paths.get("stable-diffusion"),
            "vae_path": model_paths.get("vae"),
            "prompt": req.get("prompt", ""),
            "negative_prompt": req.get("negative_prompt", ""),
            "seed": req.get("seed"),
            "width": req.get("width"),
            "height": req.get("height"),
            "num_outputs": req.get("num_outputs"),
            "num_inference_steps": req.get("num_inference_steps"),
            "guidance_scale": req.get("guidance_scale"),
            "sampler_name": req.get("sampler_name"),
        }
        trajectory_recorder = TrajectoryRecorder(trajectory, total_steps, run_metadata=run_metadata)

    if req["init_image"] is not None and not context.test_diffusers:
        req["sampler_name"] = "ddim"

    gc(context)

    context.stop_processing = False

    if req["control_image"] and controlnet_filter:
        controlnet_filter = convert_ED_controlnet_filter_name(controlnet_filter)
        req["control_image"] = filter_images(context, req["control_image"], controlnet_filter)[0]

    callback = make_step_callback(context, callback, trajectory_recorder=trajectory_recorder)

    try:
        images = generate_images(context, callback=callback, **req)
        trajectory_status = "complete"
    except UserInitiatedStop:
        trajectory_status = "interrupted"
        images = []
        if context.partial_x_samples is not None:
            if context.test_diffusers:
                images = diffusers_latent_samples_to_images(context, context.partial_x_samples)
            else:
                images = latent_samples_to_images(context, context.partial_x_samples)
    finally:
        if hasattr(context, "partial_x_samples") and context.partial_x_samples is not None:
            if not context.test_diffusers:
                del context.partial_x_samples
            context.partial_x_samples = None

        if trajectory_recorder is not None:
            try:
                trajectory_recorder.finish(status=trajectory_status)
            except Exception as e:
                # Trajectory recording is auxiliary. A manifest finalisation error
                # must be visible, but must not destroy a successfully generated image.
                log.error(f"Trajectory manifest finalisation failed: {e}")

    gc(context)

    if output_type == "base64":
        output_format = opts.get("output_format", "jpeg")
        output_quality = opts.get("output_quality", 75)
        output_lossless = opts.get("output_lossless", False)
        images = [img_to_base64_str(img, output_format, output_quality, output_lossless) for img in images]

    return images


def filter_images(context: Context, images, filters, filter_params={}, input_type="pil"):
    gc(context)

    if "nsfw_checker" in filters:
        filters.remove("nsfw_checker")  # handled by ED directly

    if len(filters) == 0:
        return images

    images = _filter_images(context, images, filters, filter_params)

    if input_type == "base64":
        output_format = opts.get("output_format", "jpg")
        output_quality = opts.get("output_quality", 75)
        output_lossless = opts.get("output_lossless", False)
        images = [img_to_base64_str(img, output_format, output_quality, output_lossless) for img in images]

    return images


def _filter_images(context, images, filters, filter_params={}):
    from sdkit.filter import apply_filters

    filters = filters if isinstance(filters, list) else [filters]
    filters = convert_ED_controlnet_filter_name(filters)

    for filter_name in filters:
        params = filter_params.get(filter_name, {})

        previous_state = before_filter(context, filter_name, params)

        try:
            images = apply_filters(context, filter_name, images, **params)
        finally:
            after_filter(context, filter_name, params, previous_state)

    return images


def before_filter(context, filter_name, filter_params):
    if filter_name == "codeformer":
        from easydiffusion.model_manager import DEFAULT_MODELS, resolve_model_to_use

        default_realesrgan = DEFAULT_MODELS["realesrgan"][0]["file_name"]
        prev_realesrgan_path = None

        upscale_faces = filter_params.get("upscale_faces", False)
        if upscale_faces and default_realesrgan not in context.model_paths["realesrgan"]:
            prev_realesrgan_path = context.model_paths.get("realesrgan")
            context.model_paths["realesrgan"] = resolve_model_to_use(default_realesrgan, "realesrgan")
            load_model(context, "realesrgan")

        return prev_realesrgan_path


def after_filter(context, filter_name, filter_params, previous_state):
    if filter_name == "codeformer":
        prev_realesrgan_path = previous_state
        if prev_realesrgan_path:
            context.model_paths["realesrgan"] = prev_realesrgan_path
            load_model(context, "realesrgan")


def get_url():
    pass


def stop_rendering(context):
    context.stop_processing = True


def refresh_models():
    pass


def list_controlnet_filters():
    from sdkit.models.model_loader.controlnet_filters import filters as cn_filters

    return cn_filters


def make_step_callback(context, callback, trajectory_recorder=None):
    capture_failed = False

    def on_step(x_samples, i, *args):
        nonlocal capture_failed
        stream_image_progress = opts.get("stream_image_progress", False)
        stream_image_progress_interval = opts.get("stream_image_progress_interval", 3)

        if context.test_diffusers:
            context.partial_x_samples = (x_samples, args[0])
        else:
            context.partial_x_samples = x_samples

        if trajectory_recorder is not None and not capture_failed:
            try:
                trajectory_recorder.record_step(i, x_samples)
            except Exception as e:
                capture_failed = True
                log.error(f"Trajectory checkpoint recording failed at callback step {i}: {e}")
                try:
                    trajectory_recorder.note_error(f"checkpoint callback {i}: {e}")
                except Exception as manifest_error:
                    log.error(f"Trajectory recorder could not persist its error state: {manifest_error}")

        if stream_image_progress and stream_image_progress_interval > 0 and i % stream_image_progress_interval == 0:
            if context.test_diffusers:
                images = diffusers_latent_samples_to_images(context, context.partial_x_samples)
            else:
                images = latent_samples_to_images(context, context.partial_x_samples)
        else:
            images = None

        if callback:
            callback(images, i, *args)

        if context.stop_processing:
            raise UserInitiatedStop("User requested that we stop processing")

    return on_step


def convert_ED_controlnet_filter_name(filter):
    def cn(n):
        if n.startswith("controlnet_"):
            return n[len("controlnet_") :]
        return n

    if isinstance(filter, list):
        return [cn(f) for f in filter]
    return cn(filter)
