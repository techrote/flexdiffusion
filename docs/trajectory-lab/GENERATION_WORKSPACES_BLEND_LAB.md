# Generation workspaces and Blend Lab (first implementation)

## Intent and integration boundary

The existing **Generate** tab is the production baseline. **Sampler Lab** reuses its
actual prompt/image editor, seed controls, model selector, task queue, batching,
and preview DOM nodes. It does not clone HTML controls or generate a second
backend request path. The experiment-specific Circuit Bend and parameter-grid
controls are visible there, and Circuit Bend is excluded from ordinary Generate
requests even if an experimental recipe was restored earlier.

**Blend Lab** operates on two *decoded image frames*. It is a standalone
post-generation/iteration-frame composition tool; it does **not** intercept
solver updates. Its output can be saved as an RGBA PNG or sent to the shared
img2img input for another generation run.

This PR builds on the still-unmerged sampler-circuit work. It is intentionally
stacked on `codex/sampler-art-progress`, not directly on `main`.

## Blend Lab workflow

1. Load **Image A** (base) and **Image B** (upper layer). The output is sized to A.
2. Optionally load a grayscale image or RGBA PNG as the **Coverage map**.
3. Choose a blend mode, upper-layer opacity, and coverage channel.
4. **Luminance × alpha:** 0 = base A, 1 = B. Transparent portions of the map
   contribute zero weight. **Alpha only:** ignores RGB and uses the map's alpha.
   **Invert coverage** exchanges black and white.
5. Download the composited PNG, or select **Use as img2img source** to load the
   result into Generate.

B and the map are smoothly resampled to A's size. This is useful for
comparing iteration checkpoints, but checkpoint frames are not exact
solver-resumption states. Any decoded input (A, B **or** coverage map) above ~8.39 million pixels is
rejected before it is rasterized, even when B or the map would otherwise be
resampled to a small A. Large exported PNGs are not automatically applied as
the generator's render dimensions. Tiny thumbnails below the generator's
64-pixel dimension step are likewise handed off without overriding valid
generation dimensions (the main UI rounds dimensions to 64-pixel multiples).

## Supported GIMP-inspired RGB operations

Initial exact list:

- Normal, Lighten only, Screen, Dodge, Addition
- Darken only, Multiply, Burn, Linear burn
- Overlay, Soft light, Hard light
- Difference, Exclusion, Subtract, Grain extract, Grain merge, Divide

Source A is the lower layer and source B is the upper. For each channel,
`blend(a,b,mode)` produces a blend candidate. The mask multiplies the selected
layer's global opacity; the final RGBA pixel uses a source-over alpha composite,
including the correct case of a transparent bottom layer.

These formulas use normalized, gamma-encoded sRGB channels, with explicit
finite/clamped handling for divide/burn/dodge endpoints. This is **not claimed
to be pixel-identical to current colour-managed GIMP 3/GEGL**, where compositing
space, alpha conventions, and layer-mode interpretation can affect results.
The implementation deliberately does not attempt color-erase, HSV/LCh,
dissolve, or GIMP's other specialised nonseparable operations yet.

## Native per-iteration spatial sampler experiment (next milestone)

The sampler-circuit branch already evaluates PLMS and DDIM on a shared timestep
schedule. The recommended native experiment is a third `mix_method`, tentatively
`spatial`, that updates **one shared latent** by computing both candidate
sampler updates at each iteration and mixing them spatially:

```
P_i = PLMS_step(X_i)
D_i = DDIM_step(X_i)
X_(i+1) = (1 - M) * P_i + M * D_i
```

`M` is a one-channel grayscale raster resampled once to latent dimensions,
then broadcast over the latent channels/batch. Optionally allow a global
blend/opacity envelope `w_i` (`M_i = w_i M`). The same operation can be
applied to per-iteration denoised estimates for inspection. Record the mask
hash, requested and realized timestep mapping, sampler order, and per-step
weights with trajectory artifacts.

This uses **two full sampler/model evaluations per iteration**, not spatially
clipped UNet execution. Spatial context is still coupled through the UNet's
receptive field/attention. It is therefore a *spatial mixture of solver
updates*, not proof that pixels are independently denoised by separate
models. Ensure exact endpoints for constant all-black/all-white masks,
history policy isolation, finite-value recovery, deterministic checkpoints,
and image-size/latent-size alignment.

For nonlinear GIMP operations, perform a separate decoded-frame experiment
with well-defined colour spaces first. Applying multiply/burn/dodge directly
to unbounded signed latents is mathematically **not** the same operation as
RGB compositing and may destabilize the diffusion dynamics. Avoid
decode/re-encode feedback each step without independently quantifying
VAE reconstruction drift, cost, and endpoint equivalence.

## Qualification

```
node tests/test_compositor.js
node tests/test_blend_lab_ui.js
node tests/test_generation_workspaces.js
node tests/test_blend_lab_browser.cjs    # requires Chrome/Chromium
node tests/test_circuit_bend_ui.js
node tests/test_step_viewer.js
```

A single GitHub-hosted CI lane now executes the mode maths, an event-driven
Blend Lab UI fixture, workspace routing, sampler/plugin regressions and a
**real headless Chromium** run loading the actual scripts and styles. The
browser fixture uses genuine decoded PNG files, canvas blending, workspace
reparenting, mask inversion, export and img2img handoff. It does **not** start
the full Easy Diffusion backend or claim a GPU generation. A separate physical
Windows/Firefox app run must validate end-to-end queueing, persisted UI settings,
mask resampling/export and the submitted vanilla versus sampler request payload.
PR #8 and its own trajectory-artifact predecessors must qualify independently.
No native spatially masked diffusion sampler or GEGL pixel-parity test is claimed.
