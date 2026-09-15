# Trajectory Lab

FlexDiffusion's initial research track for making the SD1.x denoising trajectory a first-class inspectable and eventually editable object.

Read in this order:

- `ARCHITECTURE.md` — design boundaries and history model;
- `IMPLEMENTATION_PLAN.md` — staged roadmap;
- `BASELINE.md` — upstream starting point and callback semantics;
- `IMPLEMENTATION_STATUS.md` — what is implemented now;
- `TESTING.md` — unit and GPU validation gates;
- `FIRST_GPU_RUN.md` — exact Windows procedure for the first GTX 1650 Super validation;
- `OUTPUT_AFTER_STEP_MVP.md` — first user-facing trajectory probe: render selected completed-step states into the normal output panel;
- `SAMPLER_COMPARISON.md` — repeat one request through multiple samplers as separate tasks, with fixed-seed comparison defaults;
- `BENCHMARK_RUNS.md` — queue a complete versioned sampler/trajectory test from pasted or file-loaded JSON;
- `REVIEW_CAPTURE.md` — export visible experiment results as a timestamp-matched PNG plus editable Markdown/JSON metadata sidecar;
- `SAMPLER_RESUME_RESEARCH.md` — preliminary solver-state audit for later exact branching.

The current target is Easy Diffusion's classic `sdkit` path with SD1.4/SD1.x checkpoints on GTX 1650 Super and Quadro RTX 4000. Other backends and K80 support are deferred.
