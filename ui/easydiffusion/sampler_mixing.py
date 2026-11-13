"""PLMS/DDIM share a timetable and either a live latent or coupled trajectories."""
from easydiffusion.circuit_bend import ExperimentalSamplingStopped, NoiseMemory


class PredictionRecorder:
    """Instance-local model proxy; capture DDIM's prediction without another UNet call."""
    def __init__(self, model):
        self.model = model
        self.prediction = None

    def __getattr__(self, key):
        return getattr(self.model, key)

    def apply_model(self, *args, **kwargs):
        self.prediction = self.model.apply_model(*args, **kwargs)
        return self.prediction


def sample_mixed(model, params, options, exact, steps, emit):
    import torch
    from sdkit.generate.sampler import default_samplers
    from easydiffusion.sampler_adapters import exact_ldm_timesteps, instance_function, mix_tensor

    # Installed PLMS assumes epsilon prediction. Do not silently combine it with v math.
    if model.parameterization != "eps":
        raise ValueError("PLMS/DDIM mixing currently requires an epsilon-prediction classic model (such as SD1.x)")
    plms = default_samplers.samplers['plms'](model)
    recorder = PredictionRecorder(model)
    ddim = default_samplers.samplers['ddim'](recorder)
    for sampler, eta in ((plms, 0), (ddim, options.eta)):
        if exact:
            sampler.make_schedule = instance_function(sampler.make_schedule, make_ddim_timesteps=exact_ldm_timesteps)
        sampler.make_schedule(params['S'], ddim_eta=eta, verbose=False)
    if list(plms.ddim_timesteps) != list(ddim.ddim_timesteps):
        raise ValueError("Sampler mixing requires identical timesteps")
    noise = NoiseMemory(params['x_T'], options.seed, options.noise_memory)
    ddim._bend_noise_active = False
    ddim.p_sample_ddim = instance_function(ddim.p_sample_ddim,
        noise_like=lambda shape, device, repeat=False: noise.next() * options.noise
        if ddim._bend_noise_active else torch.zeros_like(params['x_T']))
    native_sigmas = ddim.ddim_sigmas.clone()
    times = list(reversed(plms.ddim_timesteps))
    x = params['x_T']
    xp, xd = x, x.clone()
    history = []
    previous_sampler = None
    c = params['conditioning']
    cfg = dict(unconditional_guidance_scale=params['unconditional_guidance_scale'],
               unconditional_conditioning=params['unconditional_conditioning'])

    def ddim_step(state, t, index, weight):
        ddim.ddim_sigmas[index] = native_sigmas[index] * weight
        ddim._bend_noise_active = bool(ddim.ddim_sigmas[index] != 0)
        out = ddim.p_sample_ddim(state, c, t, index=index, **cfg)
        eps = recorder.prediction
        if cfg['unconditional_conditioning'] is not None and cfg['unconditional_guidance_scale'] != 1:
            uncond, cond = eps.chunk(2)
            eps = uncond + cfg['unconditional_guidance_scale'] * (cond - uncond)
        return (*out, eps)

    def output(a, b):
        return a if options.mix_output == 'plms' else b if options.mix_output == 'ddim' else mix_tensor(a, b, .5)

    def exchange_pair(a, b, weight):
        if weight == .5:
            joined = (a + b) * .5
            return joined, joined
        return mix_tensor(a, b, weight), mix_tensor(b, a, weight)

    with torch.no_grad():
        for i, timestep in enumerate(times):
            iteration, index = i + 1, steps - i - 1
            t = torch.full((x.shape[0],), int(timestep), device=x.device, dtype=torch.long)
            t_next = torch.full_like(t, int(times[min(i + 1, steps - 1)]))
            previous = x
            window = options.weight(iteration, steps)
            exchange = 0
            if options.mix_method == 'alternate':
                first, _ = options.resolve_window(steps)
                other = 'ddim' if options.mix_first == 'plms' else 'plms'
                chosen = other if window > 0 and ((iteration - first) // options.mix_block) % 2 else options.mix_first
                if options.mix_history == 'reset' and chosen != previous_sampler:
                    history = []
                history_length = len(history)
                result = (plms.p_sample_plms(x, c, t, index=index, old_eps=history, t_next=t_next, **cfg)
                          if chosen == 'plms' else ddim_step(x, t, index, window))
                if options.mix_history == 'shared' or chosen == 'plms':
                    history = (history + [result[2]])[-3:]
                x, clean = result[:2]
                finite_values = result
                previous_sampler = chosen
            else:
                chosen = 'plms+ddim'
                history_length = len(history)
                a = plms.p_sample_plms(xp, c, t, index=index, old_eps=history, t_next=t_next, **cfg)
                b = ddim_step(xd, t, index, window)
                exchange = options.exchange_weight(iteration, steps)
                # Simultaneous exchange: both new states use the same pre-exchange pair.
                xp, xd = exchange_pair(a[0], b[0], exchange)
                cp, cd = exchange_pair(a[1], b[1], exchange)
                eps = mix_tensor(a[2], b[2], exchange) if options.mix_history == 'shared' else a[2]
                history = [] if options.mix_history == 'reset' and exchange > 0 else (history + [eps])[-3:]
                x, clean = output(xp, xd), output(cp, cd)
                finite_values = (*a, *b, xp, xd, x, clean, *history)
            if any(not torch.isfinite(v).all() for v in finite_values):
                raise ExperimentalSamplingStopped(previous, i)
            emit(x, iteration, clean, timestep=int(timestep), observation_phase='after_update',
                 estimate_basis='sampler_mix_update', denoised_label=f'{chosen.upper()} update {iteration} clean estimate',
                 active_sampler=chosen, sampler_iteration=iteration, mix_method=options.mix_method,
                 mix_history=options.mix_history, history_length=history_length, exchange_weight=exchange,
                 mix_output=options.mix_output if options.mix_method == 'coupled' else 'shared_latent',
                 eta=options.eta * window if 'ddim' in chosen else 0, noise_seed=options.seed)
    return x
