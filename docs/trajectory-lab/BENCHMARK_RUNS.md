# Benchmark runs

FlexDiffusion can queue a complete controlled sampler/trajectory test from a small JSON definition instead of requiring the UI to be configured by hand for every run.

## Browser workflow

Use **Load benchmark…** next to the normal Make Image controls. The dialog accepts either:

- pasted JSON in the text area; or
- a local `.json` / `.txt` file selected with **Load file**.

The built-in **Load Cyberdino baseline** button restores the first reference benchmark without needing a file.

The complete benchmark is parsed and preflighted before any render task is queued. The UI is temporarily set to each benchmark case, the ordinary sampler-comparison path queues the selected samplers as separate Easy Diffusion tasks, and the user's previous UI values are restored afterward.

Benchmark runs deliberately use the same normal task path as hand-configured tests. They do not introduce a separate renderer or sampler execution implementation.

## Schema

Current schema identifier:

```text
flexdiffusion-benchmark/v1
```

A benchmark contains:

- `name` and optional `description`;
- `defaults`, containing shared generation settings;
- a root `samplers` list, optionally overridden per case;
- one or more `cases`, each with a stable `id`, label, optional setting overrides, and optional sampler override.

Case settings override the root defaults. Unknown setting names are rejected rather than silently ignored, so spelling mistakes cannot quietly change an experiment.

Example:

```json
{
  "schema": "flexdiffusion-benchmark/v1",
  "name": "Cyberdino sparse sampler baseline",
  "defaults": {
    "prompt": "cyberpunk dinosaur mercenary",
    "negative_prompt": "",
    "seed": 2,
    "model": "sd-v1-4",
    "width": 512,
    "height": 512,
    "inference_steps": 20,
    "guidance_scale": 7.5,
    "output_format": "png",
    "output_quality": 75,
    "capture_iterations": "10, 16, 18-20",
    "endpoint_solver_states": false,
    "num_outputs": 1,
    "num_outputs_parallel": 1,
    "clip_skip": false,
    "vae_model": "",
    "vram_usage_level": "low",
    "stream_image_progress": false,
    "block_nsfw": false,
    "show_only_filtered_image": false
  },
  "samplers": [
    "dpmpp_2m",
    "heun",
    "euler",
    "euler_a",
    "dpm2",
    "dpm2_a",
    "lms",
    "dpmpp_2s_a",
    "dpmpp_sde"
  ],
  "cases": [
    {
      "id": "baseline",
      "label": "SD1.4 seed 2 sparse sampler sweep"
    }
  ]
}
```

The expression `10, 16, 18-20` produces denoised observations at 10, 16, 18 and 19 plus the authoritative final Step-20 image: five output images per successful sampler. With nine currently supported fixed-step k-diffusion samplers this reference benchmark therefore queues nine sampler tasks and expects roughly 45 images.

## Supported v1 settings

`prompt`, `negative_prompt`, `seed`, `model`, `width`, `height`, `inference_steps`, `guidance_scale`, `output_format`, `output_quality`, `capture_iterations`, `endpoint_solver_states`, `num_outputs`, `num_outputs_parallel`, `clip_skip`, `vae_model`, `scheduler`, `vram_usage_level`, `stream_image_progress`, `block_nsfw`, and `show_only_filtered_image`.

The sparse capture expression is validated with the exact same parser used by the normal Capture iterations UI. Model and VAE settings are applied through Easy Diffusion's model-dropdown objects rather than by editing only their visible text fields, so the path used by the actual request is the benchmark value.

## Safety / experimental hygiene

Benchmark schema v1 is intentionally plain txt2img. During preflight it refuses to queue if the current UI would inject an init image, mask, ControlNet, face correction, upscaler, Hypernetwork, LoRA, reference image, or image modifier/tag. This prevents stale UI state from contaminating a nominally controlled benchmark.

It also refuses to start while another render is active and requires **Process newest jobs first** to be off, preserving the benchmark's declared sampler order. The reference Cyberdino benchmark pins low-VRAM mode and disables live previews, NSFW post-filtering, and filtered-only output so timing and outputs are not silently changed by remembered UI settings.

The sampler list is checked against the sampler options available in the current backend. The existing Output After Step compatibility gate remains authoritative, so this feature does not claim new sampler support.

## Provenance

Tasks queued by the benchmark loader carry browser-side provenance fields:

- `benchmark_schema`;
- `benchmark_name`;
- `benchmark_run_id`;
- `benchmark_case_id`;
- `benchmark_case_label`.

These fields are attached to the local task request before enqueueing, so Review Capture's raw task metadata can distinguish benchmark runs and cases without changing sampler mathematics or backend request semantics.

## Versioned definitions and future dropdown

Canonical configs live under `/benchmarks`. The schema and directory are intentionally structured so a later implementation can expose a dropdown/menu backed by an index of repository benchmark files. The MVP avoids adding a server filesystem API: paste/file loading is sufficient for reliable use now and keeps the execution path simple.
