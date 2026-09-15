from sdkit import Context

from easydiffusion.output_steps import (
    FIXED_STEP_K_DIFFUSION_SAMPLERS,
    OutputAfterStepCollector,
    normalize_intermediate_representation,
    normalize_output_after_step,
)
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


def _install_kdiff_callback_state_bridge():
    """Temporarily expose k-diffusion's full callback state to Easy Diffusion.

    sdkit 0.0.12's classic adapter currently reduces the k-diffusion callback
    dictionary to ``callback(info["x"], info["i"])``. For Output After Step we
    also need ``info["denoised"]``. Reproduce the tiny pinned adapter here while
    this render is active, forwarding the original callback dictionary as a
    third argument. The caller restores the original function immediately after
    sampling, so ordinary classic renders retain upstream behaviour.
    """
    from sdkit.generate.sampler import k_samplers as sdkit_k_samplers

    original_sample = sdkit_k_samplers.sample

    def sample_with_callback_state(
        context,
        sampler_name=None,
        noise=None,
        batch_size=1,
        shape=(),
        steps=50,
        cond=None,
        uncond=None,
        guidance_scale=0.8,
        callback=None,
        **kwargs,
    ):
        model = context.models["stable-diffusion"]
        denoiser = (
            sdkit_k_samplers.k_diffusion.external.CompVisVDenoiser
            if model.parameterization == "v"
            else sdkit_k_samplers.k_diffusion.external.CompVisDenoiser
        )
        wrapped_model = sdkit_k_samplers.DenoiserWrap(denoiser(model))
        sigmas = wrapped_model.inner_model.get_sigmas(steps)

        sample_fn = sdkit_k_samplers.samplers.get(sampler_name)
        x_latent = noise
        x_latent *= sigmas[0]

        params = {
            "model": wrapped_model,
            "x": x_latent,
            "callback": (
                lambda info: callback(info["x"], info["i"], info)
                if callback is not None
                else None
            ),
            "extra_args": {
                "uncond": uncond,
                "cond": cond,
                "guidance_scale": guidance_scale,
            },
        }

        if sampler_name in ("dpm_fast", "dpm_adaptive"):
            params["sigma_min"] = sigmas[-2]
            params["sigma_max"] = sigmas[0]
            if sampler_name == "dpm_fast":
                params["n"] = steps - 1
        else:
            params["sigmas"] = sigmas

        return sample_fn(**params)

    sdkit_k_samplers.sample = sample_with_callback_state

    def restore():
        if sdkit_k_samplers.sample is sample_with_callback_state:
            sdkit_k_samplers.sample = original_sample

    return restore


def _decode_output_step_snapshots(context, collector):
    """Decode owned CPU snapshots sequentially after denoising has finished."""
    output_format = opts.get("output_format", "jpeg")
    output_quality = opts.get("output_quality", 75)
    output_lossless = opts.get("output_lossless", False)
    results = []

    gc(context)
    for completed_step, representation, cpu_samples in collector.pop_snapshots():
        device_samples = None
        try:
            device_samples = cpu_samples.to(context.torch_device)
            images = latent_samples_to_images(context, device_samples)
            encoded = [
                img_to_base64_str(img, output_format, output_quality, output_lossless)
                for img in images
            ]
            results.append(
                {
                    "step": completed_step,
                    "representation": representation,
                    "images": encoded,
                }
            )
        finally:
            if device_samples is not None:
                del device_samples
            gc(context)

    return results


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
    output_step_collector = None

    # This context is reused between jobs. Always clear prior transient outputs.
    context.output_step_results = []
    output_after_step = req.pop("output_after_step", None)
    intermediate_representation = req.pop("intermediate_representation", "both")

    total_steps = req["num_inference_steps"]
    if req.get("init_image") is not None:
        total_steps = int(req["num_inference_steps"] * req.get("prompt_strength", 0.8))

    # The classic backend forces img2img through DDIM. Apply that rule before
    # trajectory metadata is frozen so manifests record the effective sampler,
    # not merely the sampler requested by the caller.
    if req["init_image"] is not None and not context.test_diffusers:
        req["sampler_name"] = "ddim"

    output_after_step = normalize_output_after_step(output_after_step, total_steps)
    intermediate_representation = normalize_intermediate_representation(intermediate_representation)
    if output_after_step < total_steps:
        if context.test_diffusers:
            raise RuntimeError("Output After Step currently supports only the classic Easy Diffusion backend")
        if req.get("init_image") is not None:
            raise RuntimeError("Output After Step currently supports txt2img only")
        if req.get("sampler_name") not in FIXED_STEP_K_DIFFUSION_SAMPLERS:
            supported = ", ".join(sorted(FIXED_STEP_K_DIFFUSION_SAMPLERS))
            raise RuntimeError(
                f'Output After Step is not yet supported for sampler {req.get("sampler_name")!r}. '
                f"Supported classic fixed-step k-diffusion samplers: {supported}"
            )
        output_step_collector = OutputAfterStepCollector(
            output_after_step,
            total_steps,
            representation=intermediate_representation,
        )

    if trajectory and trajectory.get("enabled", False):
        from easydiffusion.trajectory import TrajectoryRecorder

        if context.test_diffusers:
            raise RuntimeError("Trajectory capture is currently supported only by the classic sdkit path")

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

    images = []
    restore_kdiff_bridge = None
    try:
        gc(context)
        context.stop_processing = False

        if req["control_image"] and controlnet_filter:
            controlnet_filter = convert_ED_controlnet_filter_name(controlnet_filter)
            req["control_image"] = filter_images(context, req["control_image"], controlnet_filter)[0]

        callback = make_step_callback(
            context,
            callback,
            trajectory_recorder=trajectory_recorder,
            output_step_collector=output_step_collector,
        )

        if output_step_collector is not None:
            restore_kdiff_bridge = _install_kdiff_callback_state_bridge()
        try:
            images = generate_images(context, callback=callback, **req)
        finally:
            if restore_kdiff_bridge is not None:
                restore_kdiff_bridge()
                restore_kdiff_bridge = None

        trajectory_status = "complete"

        # Do not interrupt the denoising loop with VAE work. Intermediate
        # snapshots are tiny SD1.x latents copied to CPU in the callback and
        # decoded here, one at a time, after the sampler has returned normally.
        if output_step_collector is not None:
            try:
                context.output_step_results = _decode_output_step_snapshots(context, output_step_collector)
            except Exception as error:
                context.output_step_results = []
                log.error(f"Output After Step decode failed: {type(error).__name__}: {error}")
    except UserInitiatedStop:
        trajectory_status = "interrupted"
        partial_x_samples = getattr(context, "partial_x_samples", None)
        images = []
        if partial_x_samples is not None:
            if context.test_diffusers:
                images = diffusers_latent_samples_to_images(context, partial_x_samples)
            else:
                images = latent_samples_to_images(context, partial_x_samples)
    finally:
        if restore_kdiff_bridge is not None:
            restore_kdiff_bridge()

        partial_x_samples = getattr(context, "partial_x_samples", None)
        if partial_x_samples is not None:
            if not context.test_diffusers:
                del context.partial_x_samples
            context.partial_x_samples = None

        if trajectory_recorder is not None:
            try:
                trajectory_recorder.finish(status=trajectory_status)
            except Exception as e:
                # Trajectory persistence is auxiliary. Finalisation failures are
                # visible in logs but must not destroy an otherwise valid render.
                log.error(f"Trajectory finalisation failed: {e}")

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


def make_step_callback(context, callback, trajectory_recorder=None, output_step_collector=None):
    capture_failed = False
    output_step_capture_failed = False

    def decode_current_latent():
        if context.test_diffusers:
            return diffusers_latent_samples_to_images(context, context.partial_x_samples)
        return latent_samples_to_images(context, context.partial_x_samples)

    def note_capture_failure(i, error):
        nonlocal capture_failed
        capture_failed = True
        log.error(f"Trajectory checkpoint capture failed at callback step {i}: {error}")
        if trajectory_recorder is not None:
            try:
                trajectory_recorder.note_error(f"checkpoint callback {i}: {type(error).__name__}: {error}")
            except Exception as manifest_error:
                log.error(f"Trajectory recorder could not persist its error state: {manifest_error}")

    def note_output_step_failure(i, error):
        nonlocal output_step_capture_failed
        output_step_capture_failed = True
        log.error(f"Output After Step capture failed at callback step {i}: {type(error).__name__}: {error}")

    def on_step(x_samples, i, *args):
        stream_image_progress = opts.get("stream_image_progress", False)
        stream_image_progress_interval = opts.get("stream_image_progress_interval", 3)

        if context.test_diffusers:
            context.partial_x_samples = (x_samples, args[0])
        else:
            context.partial_x_samples = x_samples

        live_preview_requested = (
            stream_image_progress
            and stream_image_progress_interval > 0
            and i % stream_image_progress_interval == 0
        )
        trajectory_requested = (
            trajectory_recorder is not None
            and not capture_failed
            and trajectory_recorder.wants_callback_index(i)
        )
        trajectory_preview_requested = trajectory_requested and trajectory_recorder.wants_preview

        # k-diffusion invokes this callback before integration update i. The x
        # observed at callback index N is therefore the state after N completed
        # updates. The callback dictionary also contains the model's current
        # clean estimate under `denoised`.
        output_step_requested = (
            output_step_collector is not None
            and not output_step_capture_failed
            and output_step_collector.wants_callback_index(i)
        )
        if output_step_requested:
            try:
                callback_state = args[0] if args and isinstance(args[0], dict) else {}
                output_step_collector.capture_step(
                    i,
                    x_samples,
                    denoised=callback_state.get("denoised"),
                )
            except Exception as error:
                note_output_step_failure(i, error)

        live_images = None
        trajectory_images = None

        if live_preview_requested:
            # Preserve upstream behaviour: failures in the explicitly requested
            # Easy Diffusion live preview are generation errors.
            live_images = decode_current_latent()
            if trajectory_preview_requested:
                trajectory_images = live_images
        elif trajectory_preview_requested:
            # Trajectory preview persistence is auxiliary. A VAE decode failure
            # disables further trajectory capture but does not kill the render.
            try:
                trajectory_images = decode_current_latent()
            except Exception as error:
                note_capture_failure(i, error)
                trajectory_requested = False

        if trajectory_requested:
            try:
                trajectory_recorder.capture_step(
                    i,
                    x_samples,
                    preview_images=trajectory_images,
                )
            except Exception as error:
                note_capture_failure(i, error)

        if callback:
            callback(live_images, i, *args)

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
