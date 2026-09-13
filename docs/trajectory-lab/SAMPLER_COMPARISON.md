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

The current Output After Step backend intentionally supports only the classified fixed-step k-diffusion subset. If an additional sampler is selected that is not yet supported by Output After Step while intermediate output is active, that task will fail explicitly rather than silently changing semantics.
