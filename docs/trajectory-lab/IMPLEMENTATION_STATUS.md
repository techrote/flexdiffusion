# FlexDiffusion Trajectory Lab — Implementation Status

## Current branch state

The SD1.x/classic trajectory work is now split into three stacked changes:

1. bootstrap capture plumbing (`trajectory-capture-bootstrap`);
2. persisted checkpoint artefacts (`trajectory-artifacts`);
3. first user-facing trajectory probe (`trajectory-output-steps-mvp`).

The bootstrap PR has passing lightweight CI and is ready for review. The artefact branch also has passing dependency-light unit tests, but still requires its first real SD1.x GPU validation before it should be treated as stable. The Output After Step branch is an intentionally narrow prototype stacked on top of that artefact work and requires its own GTX 1650 Super visual/non-invasiveness check.

## Implemented

- default-off `TrajectoryData` render configuration;
- one-based exact/range/percentage capture schedule compiler;
- classic-backend-only trajectory capability routing;
- live classic-`sdkit` latent callback interception;
- atomic versioned trajectory manifests;
- interruption/failure/completion status handling;
- selected latent snapshots detached and copied to owned CPU tensors;
- latent persistence as non-pickle `.safetensors` artefacts;
- optional checkpoint preview persistence (JPEG/PNG/WebP);
- preview decode only at requested checkpoints, independent of Easy Diffusion's browser live-preview interval;
- reuse of an already-requested Easy Diffusion live preview to avoid duplicate VAE decode at the same step;
- bounded single-writer queue with explicit blocking backpressure (requested checkpoints are never silently dropped);
- configurable queue depth and storage budget;
- per-artefact SHA-256, byte counts and relative paths in the manifest;
- all-or-error cleanup for a multi-file checkpoint if a staged/commit operation fails;
- pure unit tests for schedules, config parsing, recorder state, writer commit, hashes and storage-budget failure;
- lightweight GitHub Actions unit-test workflow;
- upstream baseline/callback semantics recorded;
- `Output After Step` MVP for classic SD1.x txt2img fixed-step k-diffusion samplers;
- deferred post-sampling VAE decode of selected intermediate observations into the normal output panel;
- explicit per-image completed-step metadata for those observations.

## Important semantic boundary

Persisting the latent does **not** yet imply that a sampler can resume exactly from that checkpoint.

The manifest therefore records `resume_fidelity: UNCLASSIFIED`. Stage 3 must determine the extra solver state required by each sampler and prove resume fidelity numerically. The current classic k-diffusion adapter forwards only the latent and callback index to Easy Diffusion; it discards the rest of k-diffusion's callback state.

For fixed-step k-diffusion samplers, callback index `N` observes the latent after `N` completed integration updates, immediately before update `N`. `Output After Step` uses that completed-step interpretation and takes the final step only from the ordinary sampler return. The older trajectory-manifest display label remains provisional; raw callback indexes are retained for later migration.

Checkpoint previews are raw decoded trajectory artefacts, not final post-filtered Easy Diffusion outputs. They are intended for trajectory inspection and later branching research. Likewise, Output After Step observations do not receive per-image face correction/upscaling in the MVP.

Normal generation still follows the upstream call path when `trajectory.enabled` is false and `Output After Step` equals the final inference step. No checkpoint writer or intermediate-output CPU snapshot is created in that default case.

## Not yet implemented / not yet validated

- real-GPU validation on GTX 1650 Super and Quadro RTX 4000;
- measured dense-capture VRAM/RAM/disk/timing overhead;
- GTX 1650 Super validation that Output After Step leaves the ordinary final image unchanged;
- sampler sigma/timestep and multistep solver-state capture;
- exact/approximate resume classification;
- branch DAG persistence and lineage operations;
- trajectory viewer/jog/scrub UI;
- insert-edit operations and image/latent interventions;
- GIMP/GEGL bridge;
- Output After Step for DDIM/PLMS/adaptive samplers, img2img, or other Easy Diffusion engines;
- other Easy Diffusion engines or K80 support.

## Next gates

For the lightweight Output After Step prototype, run the SD1.4 `cyberpunk dinosaur mercenary` reference case in `OUTPUT_AFTER_STEP_MVP.md` and verify that:

1. Step 15/30 through Step 30/30 appear from one denoising run;
2. the final Step 30 image matches the ordinary final render for the same request;
3. intermediate images are labelled with their actual completed-step number;
4. CPU latent capture does not cause material GPU-memory growth or denoising stalls.

The broader persisted-artefact gate remains the fixed-seed GTX 1650 Super protocol in `TESTING.md`: baseline, sparse latent/preview/hybrid capture, dense capture, interruption and budget failure. Do not begin exact-resume work until that capture gate passes.
