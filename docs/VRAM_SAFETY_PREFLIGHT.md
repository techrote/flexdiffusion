# FlexDiffusion: VRAM preflight and main-page controls

## Scope

- Move the existing **High / Balanced / Low** GPU Memory Usage selection from System Settings to the Generate page, directly below Image Size.
- Add **Auto-derate risky jobs** (enabled by default for the browser UI, opt-in for external API callers).
- On a classic or diffusers worker, select the effective memory profile **when a queued job is claimed by its actual device**, rather than relying on the most recently saved global setting.
- If a workload is classified high-risk, disable live preview and enable VAE tiling for that job; the browser also changes multi-image parallel batches to one image per batch, **preserving total image count and output dimensions**.
- Preserve manual control: uncheck Auto-derate to disable preflight. No resolution reduction, no CPU-only fallback, no automatic model substitution.

## Technical limitations

This is an intentionally conservative **risk heuristic**, not a guarantee against OOM. It uses device total VRAM, width × height, batch count and model-family hints; a full memory estimate would also require architecture details, allocator state, other GPU processes and peak denoising/decoding allocations.

**SDKit3 / Forge**: memory profiles are configured by process launch arguments. A task-level change cannot truly switch those options on a live process; this patch logs the limitation and applies only live-preview/tiling safeguards. The installed backend must be restarted to change its actual launch-time VRAM preset.

**Ultra Low** is **not added** by this patch. In the Python SDKit classic runtime, Low already uses block-level CPU offloading and fixed attention slicing. The existing 24-way attention-slicing option is unsafe for some power-of-two attention lengths because that implementation falls back to an unsliced pass when the length is not divisible by 24. A real profile below Low requires separate implementation and GPU qualification (potentially more robust memory-budgeted attention chunks and/or model-family-aware CPU VAE/text-encoder offloading). Renaming Low or only disabling previews would be misleading for an already-single-image 1024×1024 run.

If a **single** 1024×1024 job still OOMs on Low with previews disabled, the next candidate is true latent/attention tiling or a two-stage tiled upscale workflow; neither is implemented here. Those approaches may affect image quality or output equivalence.

## Source and compatibility notes

- Existing request profiles, previously stored preferences and external API calls remain accepted.
- `auto_vram_safety` defaults to false for external API requests to avoid surprising existing automation. The browser checkbox defaults to enabled.
- Derated profile is included in task response metadata and the server log. The user's selected preference is retained in persistent app config; auto-derating is not written back as a new permanent preference.
- The implementation does not merge or depend on Sampler Lab, Blend Lab, or inpainting research branches.

## Verification

Run these dependency-light checks from the repository root:

```powershell
python scripts/test_vram_safety.py
node --test scripts/test_vram_safety.cjs
node --check ui/media/js/vram-safety.js
node --check ui/media/js/main.js
node --check ui/media/js/parameters.js
node --check ui/media/js/auto-save.js
python -m py_compile ui/easydiffusion/vram_safety.py ui/easydiffusion/runtime.py ui/easydiffusion/tasks/render_images.py
```

**Required physical owner evaluation**: Windows 11 + Firefox, especially the 4 GB GPU with classic SD1.x at 512, 768 and 1024 square, each with single output, parallel 4, both preview states and Auto-derate on/off. Confirm that total image count, requested dimensions and model/seed semantics are preserved. Repeat a representative 1024 case on SDKit3 and verify that it **does not claim a live mode change**; confirm that an explicit Low setting needs backend restart to take effect. Capture OOM failures and process logs. Test both queue-time setting changes and execution-time GPU selection. No physical GPU result is claimed by this source-only branch.
