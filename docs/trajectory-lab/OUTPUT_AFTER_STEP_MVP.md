# Sparse trajectory capture MVP

FlexDiffusion's human-facing trajectory observation control uses printer-style iteration syntax rather than the original contiguous **Output After Step** control.

Example:

```text
Inference Steps:      30
Capture iterations:   10, 16, 20-24
☐ Include first/last solver states
```

The expression expands to:

```text
10, 16, 20, 21, 22, 23, 24
```

Only those requested intermediate boundaries are retained and VAE-decoded. The sampler still runs normally to completion and its ordinary final image is always included.

## Why sparse denoised capture is the default

For visual trajectory comparison, k-diffusion's callback `denoised` value is usually the useful observation: it is the model's current predicted clean image at that solver boundary. The raw callback `x` value is the noisy solver state required for restart/branch research but is not normally needed for every review corpus.

Routine sampler sweeps therefore capture requested **denoised estimates only**. This substantially reduces image count, PNG archive size, VAE decode work and review clutter while retaining deterministic regeneration from the saved model/prompt/seed/settings metadata.

## Expression grammar

The UI follows familiar paper-printer page syntax:

- comma separates items;
- `a-b` is an inclusive ascending range;
- whitespace is ignored;
- duplicate values collapse;
- output ordering is numerical ascending;
- every value must be within `1..Inference Steps`;
- malformed or reversed ranges are rejected visibly rather than silently corrected.

Examples:

```text
10, 16, 20-24
=> 10, 16, 20, 21, 22, 23, 24

24, 10, 16, 20-22, 21
=> 10, 16, 20, 21, 22, 24
```

The UI displays the expected output count per sampler before generation.

## Endpoint solver-state option

**Include first/last solver states** adds the raw noisy solver state at the first and last requested callback boundaries. For example, with 30 inference steps and:

```text
Capture iterations: 10, 16, 20-24
```

the requested observations are:

```text
Step 10 · denoised estimate
Step 10 · solver state
Step 16 · denoised estimate
Step 20 · denoised estimate
Step 21 · denoised estimate
Step 22 · denoised estimate
Step 23 · denoised estimate
Step 24 · denoised estimate
Step 24 · solver state
Step 30 · final sampler output
```

If the highest requested iteration is the true final inference step, FlexDiffusion does **not** fabricate a post-final callback solver state. k-diffusion's callback occurs before each integration update, so there is no callback index after the final update. The sampler's ordinary return remains the authoritative final result.

## Callback semantics

For the supported classic fixed-step k-diffusion samplers, callback index `i` observes the current latent before integration update `i`:

- callback index `0` = initial solver latent;
- callback index `N` = state after `N` completed integration updates, immediately before update `N`;
- the sampler return = state after all requested updates.

The callback dictionary also exposes `denoised`, the model's current clean-image estimate at that same boundary.

FlexDiffusion copies selected latent tensors to owned CPU memory during sampling, then decodes them sequentially after denoising finishes. Intermediate VAE work therefore does not perturb the denoising loop itself.

## Current sampler scope

Sparse capture deliberately retains the same proven sampler scope as the earlier Output After Step MVP:

- `euler`
- `euler_a`
- `lms`
- `heun`
- `dpm2`
- `dpm2_a`
- `dpmpp_2s_a`
- `dpmpp_2m`
- `dpmpp_sde`

The current compatibility gate still rejects DDIM, PLMS, Stability DPM Solver, DPM Adaptive and the UniPC family while sparse/intermediate capture is active. Those sampler families need their callback/state semantics classified separately; this UI redesign does not claim support for them.

## Compatibility transport

The pinned Easy Diffusion request schema already contains `output_after_step` and `intermediate_representation`. To avoid a broad schema migration during this UI change, the browser sends:

- `output_after_step` = the minimum requested iteration, which activates the existing callback bridge;
- `intermediate_representation` = a FlexDiffusion sparse marker containing the canonical capture expression and endpoint-state flag;
- `capture_steps` and `capture_endpoint_solver_states` are also kept on the browser task object for review/provenance metadata.

The backend collector interprets that marker and retains only the explicitly selected callback boundaries. Legacy `denoised`, `solver_state`, and `both` requests keep their original contiguous semantics for backwards compatibility.

## Non-goals

This feature remains observational. It does not yet provide exact sampler resume, branching, cross-sampler state transfer, img2img sparse capture, or support for adaptive-step semantics. Those require sampler-specific state/history/RNG work documented elsewhere in the Trajectory Lab research notes.
