# Sampler comparison mode

FlexDiffusion can enqueue the same request through multiple samplers as separate ordinary Easy Diffusion tasks. The primary **Sampler** dropdown remains unchanged; an **Also run** expander appears beneath it and allows additional samplers to be checked.

Each selected sampler receives the same prompt, model, dimensions, CFG, inference-step settings, Output After Step settings, and seed. The tasks remain separate in the normal output history, matching manual serial runs and making timing/output comparisons easy to inspect.

## Fixed seed policy

The upstream **Random** seed checkbox is disabled and hidden in FlexDiffusion. Normal generation therefore uses the explicit Seed field (default `0`) unless another action deliberately supplies its own seed. This is intentional: trajectory and sampler comparisons are much more useful when accidental seed changes cannot invalidate the comparison.

## Heun versus DPM++ 2M

Heun is a valid trajectory-research sampler for the current classic SD1.x work. With the current k-diffusion defaults (`s_churn = 0`) it is deterministic for a fixed seed/configuration and exposes the same callback dictionary fields used by Output After Step, including `x`, `denoised`, `sigma`, and `sigma_hat`.

Its numerical path is different from DPM++ 2M. Heun is an explicit second-order predictor/corrector method and, except for the final zero-sigma step, performs a second denoiser evaluation for the correction. Pinned k-diffusion's DPM++ 2M performs one model evaluation per iteration and carries the previous denoised estimate as multistep history. Consequently equal nominal step counts are not equal model-evaluation counts and the two samplers should not be expected to produce the same composition or detail decisions.

If Heun removes a recurring artifact in the cyberdino reference, that is a legitimate sampler-dependent result rather than a problem for FlexDiffusion. It also makes Heun useful as a second trajectory family for comparison. However, a single wall-time observation should not be taken as intrinsic sampler speed: Easy Diffusion task timing can include first-load/warm-up effects, VAE decode, checkpoint-output decode, cache state and GPU thermal state. Multi-sampler mode makes repeated same-session A/B runs straightforward.

## Output After Step interaction

Multi-sampler mode creates separate tasks; it does not try to splice different samplers into one trajectory. Therefore each sampler gets its own Output After Step observations and its own final output.

The current Output After Step backend intentionally supports only the classified fixed-step k-diffusion subset. If an additional sampler is selected that is not yet supported by Output After Step while intermediate output is active, that task fails explicitly rather than silently changing semantics.

Current supported set:

- `dpm2`
- `dpm2_a`
- `dpmpp_2m`
- `dpmpp_2s_a`
- `dpmpp_sde`
- `euler`
- `euler_a`
- `heun`
- `lms`

### Confirmed compatibility-gate failures from the controlled sampler sweep

These are **not sampler-generation failures**. They are confirmed rejections by FlexDiffusion's current Output After Step compatibility gate because their callback/step semantics have not yet been adapted:

- `ddim`
- `plms`
- `dpm_adaptive`
- `dpm_solver_stability`
- UniPC family:
  - `unipc_snr`
  - `unipc_tu`
  - `unipc_snr_2`
  - `unipc_tu_2`
  - `unipc_tq`

Representative error:

```text
Error: Output After Step is not yet supported for sampler 'dpm_solver_stability'. Supported classic fixed-step k-diffusion samplers: dpm2, dpm2_a, dpmpp_2m, dpmpp_2s_a, dpmpp_sde, euler, euler_a, heun, lms
```

This distinction matters for the implementation plan: Easy Diffusion/sdkit already contains DDIM, PLMS, Stability DPM Solver and the UniPC implementations. Supporting them in Trajectory Lab primarily requires family-specific callback-state adapters, step-number classification, and clean-estimate extraction rather than implementing the samplers themselves.

`dpm_adaptive` is a separate case because its adaptive step count does not naturally fit the current fixed `Output After Step = N` interpretation. It will need an explicit observation-index/solver-boundary policy before being enabled.

Any additional failures from the same sweep should be appended here with the exact error so we can distinguish deliberate compatibility rejection from a genuine sampler/runtime fault.
