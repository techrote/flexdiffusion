# SD1.x classic sampler resume research

**Status:** preliminary source audit only. No sampler is declared restartable by FlexDiffusion yet.

## Why this exists

A saved latent is only one part of a diffusion solver's state. FlexDiffusion must not expose an "exact branch" operation until an interrupted/resumed run has been compared numerically with an uninterrupted reference for that sampler.

Easy Diffusion's classic backend currently uses `sdkit`, whose pinned dependency is `k-diffusion==0.0.12`. `sdkit/generate/sampler/k_samplers.py` builds the full sigma schedule, invokes k-diffusion, and reduces the k-diffusion callback dictionary to only:

```python
callback(info["x"], info["i"])
```

That discards `sigma`, `sigma_hat`, `denoised`, and any solver-private history. The current Trajectory Lab therefore records a useful latent observation, but not yet a complete solver checkpoint.

## Critical callback-boundary detail

For the audited k-diffusion 0.0.12 samplers, the callback is generally invoked **inside the current iteration before that iteration updates `x`**. For example, Euler computes the model derivative, invokes the callback with the current `x`, then performs the Euler update.

Consequences:

- a callback checkpoint is best understood as a **solver boundary / state entering the current integration update**, not blindly as "the image after step N";
- the last callback does not itself contain the sampler's final returned latent;
- the existing Easy Diffusion live-preview convention inherits this same boundary;
- UI naming must remain explicit until the desired human-facing step convention is frozen;
- exact resume should restart at the captured boundary, not rerun initial noise scaling.

The current Stage 1 manifest retains both the raw zero-based `callback_index` and its existing one-based `step` label so this can be migrated without losing the original observation point.

## Preliminary sampler classes

### Euler (`euler`) — best first exact-resume candidate

With the default k-diffusion settings used by sdkit (`s_churn=0`), Euler has no multistep history. At the callback boundary, the essential dynamic state is approximately:

- current latent `x`;
- current position in the sigma schedule / exact sigma value;
- unchanged conditioning/unconditioning and CFG parameters;
- exact model/VAE/sampler configuration.

The normal `sdkit` sampler entrypoint cannot be reused directly for resume because it assumes its input is initial noise and multiplies it by `sigmas[0]`. A resume path must bypass that initialization and continue from the captured `x` with the remaining sigma schedule.

This is the first sampler that should receive a numerical exactness experiment.

### Heun (`heun`) and deterministic DPM2 (`dpm2`)

These are still one-step methods in the sense that no derivative history is retained across completed outer iterations. Each iteration performs additional model evaluation(s) internally after the callback.

A restart at the callback boundary can in principle recompute the current iteration from `x` plus the correct sigma schedule. They are plausible second-wave exact candidates after Euler, but must be measured rather than inferred.

### Euler ancestral (`euler_a`) and DPM2 ancestral (`dpm2_a`)

These inject random noise after the callback. A latent and sigma alone are insufficient for bitwise continuation.

An exact checkpoint must preserve or deterministically reconstruct the stochastic source state used for the *next* noise draw. For the default k-diffusion noise sampler this likely means controlled RNG state/seed at the callback boundary, but this needs a dedicated experiment.

### DPM++ 2S ancestral (`dpmpp_2s_a`)

No long derivative history is expected, but it is ancestral/stochastic. Treat it like the other ancestral samplers: capture the noise process state in addition to latent/schedule state and prove equality.

### LMS (`lms`) — explicitly stateful

`sample_lms` retains a rolling `ds` list of derivative tensors (up to order 4). At callback time the current derivative has already been appended, and the upcoming update combines it with prior derivatives.

Therefore a latent-only checkpoint cannot reproduce an interior LMS continuation exactly. An exact implementation needs either:

- capture/restore the derivative history and order context; or
- replay enough preceding solver history to reconstruct it.

Direct state capture is preferable for interactive branching.

### DPM++ 2M (`dpmpp_2m`) — explicitly stateful

The sampler retains `old_denoised` from the prior iteration and uses it in the next multistep update. The existing sdkit callback wrapper cannot access that private variable.

Exact resume requires sampler instrumentation or a FlexDiffusion-owned implementation that exposes/restores `old_denoised` and the schedule position.

### DPM++ SDE (`dpmpp_sde`) — stochastic process state

k-diffusion constructs a `BrownianTreeNoiseSampler` when none is supplied. That object represents a consistent Brownian path over the sigma interval; ordinary global RNG state alone is not a sufficient abstraction after construction.

A robust FlexDiffusion implementation should own the Brownian-tree seed/configuration explicitly and prove that recreating the tree reproduces future increments across a branch boundary.

### `dpm_fast` / `dpm_adaptive`

These use solver-controlled/adaptive integration rather than a simple fixed one-callback-per-user-step model. They should be treated as late resume targets. Checkpoint semantics, internal solver state, and the relationship between callback count and requested step count all need separate characterization.

### DDIM / PLMS / classic DPM Solver

These go through the non-k-diffusion sampler family in `sdkit/default_samplers.py` and require a separate source audit of the exact LDM versions in the installed classic environment.

Likely concerns:

- DDIM with `eta=0` should be a relatively tractable deterministic case once callback boundary semantics are confirmed;
- PLMS is a multistep method and is expected to require prior epsilon/history state;
- the classic DPM solver may maintain its own multistep state.

Do not generalize k-diffusion findings to these samplers.

## Proposed resume evidence ladder

For every sampler, add support only through the following progression:

1. **Boundary identification** — prove exactly what state the callback represents.
2. **State inventory** — identify latent, schedule position, RNG/noise-process state, and solver history required for continuation.
3. **Unmodified resume** — interrupt at a checkpoint, restore it, and continue with no edit.
4. **Numerical comparison** — compare every downstream checkpoint and final latent/image against uninterrupted reference.
5. **Repeated seeds/steps** — run a matrix rather than one lucky example.
6. **Only then classify fidelity** as `EXACT`, `STATEFUL_EXACT`, or `APPROXIMATE`.
7. **Branch edit support** is enabled only after unmodified resume has passed.

## Recommended first implementation after the capture gate

Implement a deliberately narrow **Euler Resume Probe**, not the full history DAG:

- classic SD1.x text-to-image only;
- deterministic Euler;
- batch size 1;
- same prompt/conditioning/model/seed;
- capture exact sigma schedule position alongside latent;
- bypass initial-noise scaling on resume;
- compare resumed downstream latents/final image against uninterrupted run.

If that fails, investigate boundary/index/precision semantics before expanding scope. If it passes, Euler becomes the reference implementation around which branch lineage and insert-edit experiments can be built.
