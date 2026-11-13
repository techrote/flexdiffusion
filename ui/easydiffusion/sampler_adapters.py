"""Classic sampler observations with per-render state and native solver math.

sdkit has no sampler-factory injection API. Install pass-through dispatchers
once, then route only the active context through a ContextVar. No temporary
process-wide monkeypatch or sampler history is shared between worker threads.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import Lock

from easydiffusion.circuit_bend import ExperimentalSamplingStopped, NoiseMemory, PLMSHistoryFreeze, sample_bent
from easydiffusion.output_steps import FIXED_STEP_K_DIFFUSION_SAMPLERS


LDM_SAMPLERS = frozenset({"ddim", "plms"})
UNIPC_SAMPLERS = frozenset({"unipc_snr", "unipc_tu", "unipc_tq", "unipc_snr_2", "unipc_tu_2"})
CAPTURE_SAMPLERS = FIXED_STEP_K_DIFFUSION_SAMPLERS | LDM_SAMPLERS | UNIPC_SAMPLERS
_active = ContextVar("flexdiff_sampling_session", default=None)
_install_lock = Lock()


def _scalar(value):
    return float(value.detach().flatten()[0].item()) if hasattr(value, "detach") else float(value)


def _install_dispatchers():
    from sdkit.generate.sampler import default_samplers, k_samplers, unipc_samplers

    with _install_lock:
        for module, family in ((default_samplers, "ldm"), (k_samplers, "kdiff"), (unipc_samplers, "unipc")):
            if getattr(module.sample, "_flexdiff_dispatcher", False):
                continue
            original = module.sample

            def make_dispatcher(original, family):
                @wraps(original)
                def dispatch(context, *args, **kwargs):
                    session = _active.get()
                    if session is not None and session.context is context:
                        return session.sample(family, original, context, *args, **kwargs)
                    return original(context, *args, **kwargs)
                dispatch._flexdiff_dispatcher = True
                return dispatch

            module.sample = make_dispatcher(original, family)


@contextmanager
def sampling_session(context, collector=None, bend=None, enabled=True):
    if not enabled:
        yield None
        return
    _install_dispatchers()
    session = SamplerSession(context, collector, bend)
    token = _active.set(session)
    try:
        yield session
    finally:
        _active.reset(token)


class SamplerSession:
    def __init__(self, context, collector=None, bend=None):
        self.context = context
        self.collector = collector
        self.bend = bend

    def sample(self, family, original, context, sampler_name=None, noise=None, batch_size=1,
               shape=(), steps=50, cond=None, uncond=None, guidance_scale=0.8, callback=None, **kwargs):
        if sampler_name not in CAPTURE_SAMPLERS or "init_image_latent" in kwargs:
            return original(context, sampler_name, noise, batch_size, shape, steps,
                            cond, uncond, guidance_scale, callback, **kwargs)
        self.context.sampling_metadata = {
            "sampler_name": sampler_name, "requested_steps": steps,
            "actual_steps": steps, "completed_steps": 0, "status": "sampling",
        }
        if self.bend is not None and self.bend.enabled:
            self.context.sampling_metadata["circuit_bend"] = self.bend.dict()
        model = context.models["stable-diffusion"]

        def emit(x, i, denoised=None, **metadata):
            self.context.sampling_metadata["completed_steps"] = i
            if callback is not None:
                callback(x, i, {"denoised": denoised, "observation": metadata,
                                "actual_steps": self.context.sampling_metadata["actual_steps"]})

        common = dict(S=steps, batch_size=batch_size, shape=shape, conditioning=cond,
                      verbose=False, unconditional_guidance_scale=guidance_scale,
                      unconditional_conditioning=uncond, eta=0.0, x_T=noise)
        if family == "ldm":
            result = self._ldm(model, sampler_name, common, emit)
        elif family == "unipc":
            result = self._unipc(model, sampler_name, noise, steps, cond, uncond, guidance_scale, emit)
        else:
            result = self._kdiff(model, sampler_name, noise, steps, cond, uncond, guidance_scale, emit)
        self.context.sampling_metadata.update(
            completed_steps=self.context.sampling_metadata["actual_steps"], status="complete"
        )
        return result

    def _ldm(self, model, name, params, emit):
        from sdkit.generate.sampler import default_samplers
        from ldm.modules.diffusionmodules.util import make_ddim_timesteps

        # Inspect the actual native timetable before it indexes alpha buffers.
        if not 1 <= params["S"] <= model.num_timesteps:
            raise ValueError(f"{name.upper()} steps must be within 1..{model.num_timesteps}")
        bend = self.bend if self.bend is not None and self.bend.enabled else None
        exact = bend is not None and bend.schedule == "exact"
        timetable_fn = exact_ldm_timesteps if exact else make_ddim_timesteps
        timetable = timetable_fn("uniform", params["S"], model.num_timesteps, verbose=False)
        if min(timetable) < 0 or max(timetable) >= len(model.alphas_cumprod):
            raise ValueError(
                f"{name.upper()} native schedule is invalid for {params['S']} requested steps. "
                "Try 20, 25, 40 or 50 steps, or enable the circuit bend Exact steps schedule."
            )
        actual = len(timetable)
        self.context.sampling_metadata["actual_steps"] = actual
        self.context.sampling_metadata.update(schedule="exact" if exact else "native",
                                              timesteps=[int(t) for t in timetable])
        if self.collector is not None:
            self.collector.set_actual_steps(actual)
        if bend is not None and bend.mode == "sampler_mix":
            from easydiffusion.sampler_mixing import sample_mixed
            self.context.sampling_metadata.update(sampler_pair=["plms", "ddim"], mix_method=bend.mix_method,
                                                  mix_history=bend.mix_history)
            return sample_mixed(model, params, bend, exact, actual, emit)
        sampler = default_samplers.samplers[name](model)
        if exact:
            sampler.make_schedule = instance_function(sampler.make_schedule, make_ddim_timesteps=exact_ldm_timesteps)
        active_bend = bend if bend is not None and bend.active else None
        if name == "ddim" and active_bend is not None:
            import torch
            params["eta"] = bend.eta
            source = NoiseMemory(params["x_T"], bend.seed, bend.noise_memory)
            # The cloned native update owns its noise provider; no module/global RNG patch.
            sampler.p_sample_ddim = instance_function(sampler.p_sample_ddim,
                noise_like=lambda shape, device, repeat=False: source.next() * bend.noise
                    if sampler._bend_noise_active else torch.zeros_like(params["x_T"]))
        observe_ldm_steps(sampler, name, emit, active_bend, actual, params["x_T"])
        samples, _ = sampler.sample(**params)
        return samples

    def _unipc(self, model, name, noise, steps, cond, uncond, guidance_scale, emit):
        import torch
        from sdkit.generate.sampler import unipc_samplers
        from sdkit.generate.sampler.unipc_samplers.unipc_sampler import NoiseScheduleVP, UniPC, model_wrapper

        preset = unipc_samplers.samplers[name]
        if steps < preset["order"]:
            raise ValueError(f"{name} requires at least {preset['order']} inference steps")
        if name == "unipc_snr_2" and noise.shape[0] != 1:
            raise ValueError("Experimental UniPC SNR 2 currently requires one image per batch")
        # Matches UniPCSampler.sample's small constructor; integration remains upstream.
        alphas = model.alphas_cumprod.clone().detach().to(dtype=torch.float32, device=model.device)
        schedule = NoiseScheduleVP("discrete", alphas_cumprod=alphas)
        model_fn = model_wrapper(
            lambda x, t, c: model.apply_model(x, t, c), schedule,
            model_type="v" if model.parameterization == "v" else "noise",
            guidance_type="classifier-free", condition=cond, unconditional_condition=uncond,
            guidance_scale=guidance_scale,
        )
        solver = UniPC(model_fn, schedule, predict_x0=True, thresholding=preset["thresholding"],
                       variant=preset["variant"])
        observe_unipc_steps(solver, emit)
        return solver.sample(noise, steps=steps, skip_type=preset["time_skip"], method="multistep",
                             order=preset["order"], lower_order_final=preset["lower_order_final"]).to(model.device)

    def _kdiff(self, model, name, noise, steps, cond, uncond, guidance_scale, emit):
        from sdkit.generate.sampler import k_samplers

        denoiser = (k_samplers.k_diffusion.external.CompVisVDenoiser if model.parameterization == "v"
                    else k_samplers.k_diffusion.external.CompVisDenoiser)
        wrapped = k_samplers.DenoiserWrap(denoiser(model))
        sigmas = wrapped.inner_model.get_sigmas(steps)
        noise *= sigmas[0]

        def on_step(info):
            emit(info["x"], info["i"], info["denoised"],
                 observation_phase="after_churn" if info.get("bend_applied") else "before_update",
                 estimate_basis="current_solver_state", sigma=_scalar(info["sigma"]),
                 sigma_hat=_scalar(info["sigma_hat"]),
                 **{key: info[key] for key in ("bend_weight", "noise_memory") if key in info})

        params = dict(model=wrapped, x=noise, sigmas=sigmas, callback=on_step,
                      extra_args=dict(uncond=uncond, cond=cond, guidance_scale=guidance_scale))
        if self.bend is not None and self.bend.active:
            return sample_bent(sampler_name=name, options=self.bend, **params)
        return k_samplers.samplers[name](**params)


def exact_ldm_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=False):
    """Exactly S unique, bounded indices; matches native uniform on safe divisors."""
    import numpy as np
    if ddim_discr_method != "uniform" or not 1 <= num_ddim_timesteps <= num_ddpm_timesteps:
        raise ValueError("Exact schedule requires uniform steps within the model timetable")
    # Keep native +1 offset, except full DDPM length must include zero to stay unique.
    offset = 0 if num_ddim_timesteps == num_ddpm_timesteps else 1
    return offset + np.floor(np.arange(num_ddim_timesteps) * num_ddpm_timesteps / num_ddim_timesteps).astype(int)


def instance_function(method, **replacements):
    """Bind upstream code with instance-local dependencies, never mutate its globals."""
    import inspect
    import torch
    from types import FunctionType, MethodType
    original = inspect.unwrap(method.__func__)
    function = FunctionType(original.__code__, dict(original.__globals__, **replacements),
                            original.__name__, original.__defaults__, original.__closure__)
    function.__kwdefaults__ = original.__kwdefaults__
    return MethodType(torch.no_grad()(function), method.__self__)


def seed_noise_like(reference, seed):
    """Match sdkit's seed-per-image initial noise without touching global generators."""
    import torch
    return torch.cat([torch.randn((1, *reference.shape[1:]), device=reference.device,
        dtype=reference.dtype, generator=torch.Generator(device=reference.device).manual_seed(seed + i))
        for i in range(reference.shape[0])])


def mix_tensor(current, donor, weight):
    # Exact endpoints matter for both reference comparisons and hard interlacing.
    return current if weight == 0 else donor if weight == 1 else current + weight * (donor - current)


def observe_ldm_steps(sampler, name, emit, bend=None, steps=None, initial_noise=None):
    """Observe native boundaries; an optional independent donor adds one trajectory."""
    method_name = "p_sample_ddim" if name == "ddim" else "p_sample_plms"
    original = getattr(sampler, method_name)
    completed = 0
    freeze = PLMSHistoryFreeze(bend, steps) if name == "plms" and bend is not None and bend.history_source == "freeze" else None
    use_donor = (name == "plms" and bend is not None and bend.history_source == "seed"
                 and any(bend.blend_weight(i, steps) > 0 for i in range(1, steps + 1)))
    donor_x = seed_noise_like(initial_noise, bend.donor_seed) if use_donor else None
    donor_history = []
    independent = use_donor and bend.blend_target == "independent"
    primary_x = initial_noise
    primary_history = []

    def update(x, c, t, *args, **kwargs):
        nonlocal completed, donor_x, donor_history, primary_x, primary_history
        metadata = {}
        donor_result = None
        strength = 0
        if use_donor:
            strength = bend.blend_weight(completed + 1, steps)
            # Independent branch advances at every timestep, even outside the bend window.
            # It shares model weights and schedule, never the primary branch's history.
            donor_result = original(donor_x, c, t, *args, **dict(kwargs, old_eps=donor_history))
            if bend.blend_target == "history" and strength:
                kwargs["old_eps"] = [mix_tensor(a, b, strength)
                    for a, b in zip(kwargs.get("old_eps", []), donor_history)]
            metadata.update(donor_seed=bend.donor_seed, blend_target=bend.blend_target,
                            blend_curve=bend.blend_curve, blend_weight=strength)
        if freeze is not None:
            history, strength = freeze.apply(kwargs.get("old_eps", []), completed + 1)
            kwargs["old_eps"] = history
            metadata = {"history_freeze_at": freeze.frozen_at, "history_strength": strength}
        if name == "ddim" and bend is not None:
            index = kwargs.get("index", args[0] if args else None)
            weight = bend.weight(completed + 1, steps)
            sigma = sampler.ddim_sigmas[index].clone()
            sampler.ddim_sigmas[index] = sigma * weight
            sampler._bend_noise_active = bool(sigma * weight != 0)
            metadata.update(eta=bend.eta * weight, noise_seed=bend.seed, noise_memory=bend.noise_memory)
            try:
                result = original(x, c, t, *args, **kwargs)
            finally:
                sampler.ddim_sigmas[index] = sigma
        else:
            if independent:
                result = original(primary_x, c, t, *args, **dict(kwargs, old_eps=primary_history))
                primary_x = result[0]
                primary_history = (primary_history + [result[2]])[-3:]
            else:
                result = original(x, c, t, *args, **kwargs)
        if use_donor:
            if bend.blend_target in ("trajectory", "independent"):
                result = tuple(mix_tensor(a, b, strength) for a, b in zip(result, donor_result))
            donor_x = donor_result[0]
            donor_history = (donor_history + [donor_result[2]])[-3:]
        if bend is not None:
            import torch
            finite_values = (*result, *(donor_result or ()), *((primary_x, *primary_history) if independent else ()))
            if any(not torch.isfinite(value).all() for value in finite_values):
                raise ExperimentalSamplingStopped(x, completed)
        trajectory_mix = use_donor and bend.blend_target in ("trajectory", "independent") and strength > 0
        emit(x, completed, result[1], timestep=_scalar(t), observation_phase="before_update",
             estimate_basis="model_prediction" if name == "ddim" else "plms_trajectory_blend" if trajectory_mix else "plms_history",
             denoised_label="Denoised estimate" if name == "ddim" else "Blended PLMS clean estimate" if trajectory_mix else "PLMS solver clean estimate",
             **metadata)
        completed += 1
        return result

    setattr(sampler, method_name, update)


def observe_unipc_steps(solver, emit):
    """The native image callback changes phase after warm-up; observe returns."""
    original = solver.multistep_uni_pc_update
    completed = 0

    def update(x, model_prev_list, t_prev_list, t, order, **kwargs):
        nonlocal completed
        result = original(x, model_prev_list, t_prev_list, t, order, **kwargs)
        completed += 1
        emit(result[0], completed, result[1], timestep=_scalar(t), order_used=order,
             observation_phase="after_update", estimate_basis="predictor_candidate",
             denoised_label="Predictor clean estimate",
             solver_state_label="Corrected solver state" if kwargs.get("use_corrector", True) else "Solver state")
        return result

    solver.multistep_uni_pc_update = update
