# Output After Step MVP

`Output After Step` is the first small user-facing Trajectory Lab control. It exposes
selected states from one denoising trajectory in Easy Diffusion's normal output
panel without yet claiming that those states are restartable.

## User behaviour

The controls sit directly below **Inference Steps**. For example:

```text
Inference Steps:      20
Output After Step:    10
Intermediate View:    Both
```

runs one 20-step denoising trajectory and returns observations for completed steps
10 through 19, followed by the sampler's ordinary final result as step 20.

`Intermediate View` selects what is retained at each requested k-diffusion callback:

- **Denoised estimate** — the model's current predicted clean latent (`denoised` / x0-like estimate);
- **Solver state** — the actual noisy latent `x` being integrated;
- **Both** — emits the denoised estimate first and solver state second for each step.

When `Output After Step` equals `Inference Steps`, generation behaves like stock
Easy Diffusion and only the final image is returned. `Output After Step` follows
changes to `Inference Steps` until the user explicitly chooses an earlier value.

## Why two representations exist

The first GTX 1650 Super prototype rendered raw solver states for a 30-step
DPM++ 2M run. Those images behaved exactly like noisy unfinished versions of the
final image. That is expected: after 20 completed updates of a 30-step schedule,
`x` still contains the noise that the remaining ten integrations are meant to
remove.

k-diffusion also computes `denoised` before each integration update. This is the
model's current estimate of the clean image and is generally the more useful
representation for human trajectory browsing. The raw solver state remains
important for later resume/branching research, so FlexDiffusion now exposes both
rather than replacing one with the other.

## Callback bridge

Pinned classic sdkit reduces k-diffusion's callback dictionary to:

```python
callback(info["x"], info["i"])
```

and discards `denoised`, `sigma`, and `sigma_hat`. While Output After Step is
active, FlexDiffusion temporarily substitutes the small pinned k-diffusion adapter
with an equivalent bridge that forwards the original callback dictionary as an
additional argument. The upstream function is restored immediately after the
sampler returns.

This bridge does not alter the sampler equations, sigma schedule, model calls, or
returned final latent. It only preserves callback metadata that sdkit previously
dropped.

## Capture architecture

The MVP deliberately avoids VAE work inside the denoising loop. At each requested
callback boundary the classic backend:

1. receives the solver state and full k-diffusion callback dictionary;
2. detaches/copies the requested representation(s) to owned CPU tensors;
3. immediately continues sampling.

After the sampler has returned normally, saved latents are moved back to the render
device one at a time, decoded through the existing SD1.x VAE path, encoded using
the requested output format, and inserted before the ordinary final image in the
normal render response.

An SD1.x 512x512 latent is only `4 x 64 x 64` values. Retaining both representations
roughly doubles the tiny CPU snapshot cost but still avoids repeated UNet runs.
VAE decoding still costs time, but happens after denoising.

## Step semantics

For the fixed-step k-diffusion samplers in the current classic sdkit path, the
callback observes `info["x"]` and `info["denoised"]` before integration update
`info["i"]`.

Therefore:

- callback index 0 is the initial solver state: zero completed updates;
- callback index N is the boundary after N completed updates, immediately before update N;
- `denoised` is the clean estimate evaluated at that same boundary;
- the last callback is not the sampler's final returned latent;
- the ordinary sampler result is the authoritative state after the final step.

Consequently `Output After Step = 10` captures callback indexes 10 onward and
labels callback index 10 as **Step 10**. The final step always comes from the
ordinary final render.

This convention is intentionally more precise than the older trajectory-manifest
provisional `callback_index + 1` display label. Existing manifests retain their raw
callback indexes so that convention can be migrated without losing the original
observation boundary.

## MVP scope

Supported now:

- Easy Diffusion `ed_classic`;
- SD1.x classic txt2img;
- batch/parallel output;
- fixed-step k-diffusion samplers:
  - Euler / Euler Ancestral;
  - Heun;
  - LMS;
  - DPM2 / DPM2 Ancestral;
  - DPM++ 2S Ancestral;
  - DPM++ 2M;
  - DPM++ SDE;
- denoised-estimate, solver-state, or dual representation output;
- standard output panel and per-image step/representation metadata;
- representation-specific download filenames so paired outputs do not collide;
- global NSFW policy on intermediate observations.

Deliberately deferred:

- img2img/inpainting;
- DDIM, PLMS and Stability DPM Solver callback semantics;
- adaptive/flexible-count `dpm_fast` and `dpm_adaptive`;
- non-classic backends;
- face correction, upscaling and other post-filters on every intermediate;
- automatic save-to-disk of every intermediate;
- resume, branching, or exact solver-state restoration.

Intermediate outputs are observations only. No exact-resume claim is made.

## Reference prototype test

The reference case is:

```text
Prompt:                    cyberpunk dinosaur mercenary
Seed:                      2
Model:                     sd-v1-4
Sampler:                   DPM++ 2M (Karras)
Image Size:                512 x 512
CFG:                       7.5
Inference Steps:           30
Output After Step:         20
Intermediate View:         Both
```

Expected output is paired `Denoised estimate` and `Solver state` images for Steps
20/30 through 29/30, followed by the ordinary Step 30/30 final image.

The key comparison is now between:

1. the separately generated 20-step final image;
2. Step 20/30 **Denoised estimate**;
3. Step 20/30 **Solver state**.

This distinguishes schedule-path differences from residual-noise differences. If
the 20-step final is aesthetically distinct from the 30-step run's clean estimate
at Step 20, the shorter Karras schedule is genuinely traversing a different state
rather than merely stopping the same trajectory early.

## Non-invasive final-output requirement

Enabling intermediate observations must not alter the ordinary final sampler
result. CPU snapshot operations are read-only with respect to the live tensors,
and the final image is still produced through the pre-existing generation path.

The raw solver-state prototype has been exercised successfully on the GTX 1650
Super. The dual-representation callback bridge still requires a real-GPU visual
check before its behaviour is considered hardware-validated.
