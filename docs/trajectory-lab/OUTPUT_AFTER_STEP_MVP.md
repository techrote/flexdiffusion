# Output After Step MVP

`Output After Step` is the first small user-facing Trajectory Lab control. It exposes
selected states from one denoising trajectory in Easy Diffusion's normal output
panel without yet claiming that those states are restartable.

## User behaviour

The control sits directly below **Inference Steps**.

For example:

```text
Inference Steps:    20
Output After Step:  10
```

runs one 20-step denoising trajectory and returns observations for completed steps
10 through 19, followed by the sampler's ordinary final result as step 20.

When the two fields are equal, generation behaves like stock Easy Diffusion and
only the final image is returned. `Output After Step` follows changes to
`Inference Steps` until the user explicitly chooses an earlier value.

## Capture architecture

The MVP deliberately avoids VAE work inside the denoising loop.

At each requested callback boundary the classic backend:

1. detaches the current latent;
2. copies an owned snapshot to CPU;
3. immediately continues sampling.

After the sampler has returned normally, the saved latents are moved back to the
render device one at a time, decoded through the existing SD1.x VAE path, encoded
using the requested output format, and inserted before the ordinary final image in
the normal render response.

An SD1.x 512x512 latent is only `4 x 64 x 64` values, so retaining a modest run of
CPU snapshots is cheap compared with repeatedly running the UNet. Decoding still
costs time, but it happens after denoising and does not require separate generation
runs.

## Step semantics

For the fixed-step k-diffusion samplers in the current classic sdkit path, the
callback receives `info["x"]` before integration update `info["i"]`.

Therefore:

- callback index 0 is the initial latent: zero completed updates;
- callback index N is the state after N completed updates, immediately before
  update N;
- the last callback is not the sampler's final returned latent;
- the ordinary sampler result is the authoritative state after the final step.

Consequently `Output After Step = 10` captures callback indexes 10 onward and
labels callback index 10 as **Step 10**. The final step always comes from the
ordinary final render.

This user-facing convention is intentionally more precise than the older
trajectory-manifest provisional `callback_index + 1` display label. Existing
manifests retain their raw callback indexes so that convention can be migrated
without losing the original observation boundary.

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
- standard output panel and per-image step metadata;
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

The first visual test is the SD1.4 case that motivated the control:

```text
Prompt:             cyberpunk dinosaur mercenary
Seed:               2
Model:              sd-v1-4
Sampler:            DPM++ 2M (Karras)
Image Size:         512 x 512
CFG:                7.5
Inference Steps:    30
Output After Step:  15
```

Expected output is Step 15/30 through Step 30/30 from one denoising run.

This directly distinguishes two hypotheses:

1. an appealing earlier image exists inside the 30-step trajectory and is later
   overworked; or
2. the separate 20-step Karras schedule follows a different, preferable
   trajectory and the corresponding image never exists inside the 30-step run.

Either result informs the next Trajectory Lab experiments.

## Non-invasive final-output requirement

Enabling intermediate observations must not alter the ordinary final sampler
result. The CPU snapshot operation is read-only with respect to the live latent.
The final image is still produced through the pre-existing generation path.

A real-GPU check on the GTX 1650 Super remains required before treating the MVP
as hardware-validated.
