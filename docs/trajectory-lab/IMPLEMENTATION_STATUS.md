# FlexDiffusion Trajectory Lab — Implementation Status

## Current branch state

The SD1.x/classic trajectory work is now split into two stacked changes:

1. bootstrap capture plumbing (`trajectory-capture-bootstrap`);
2. persisted checkpoint artefacts (`trajectory-artifacts`).

The bootstrap PR has passing lightweight CI and is ready for review. The artefact branch also has passing dependency-light unit tests, but still requires its first real SD1.x GPU validation before it should be treated as stable.

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
- upstream baseline/callback semantics recorded.

## Important semantic boundary

Persisting the latent does **not** yet imply that a sampler can resume exactly from that checkpoint.

The manifest therefore records `resume_fidelity: UNCLASSIFIED`. Stage 3 must determine the extra solver state required by each sampler and prove resume fidelity numerically. The current classic k-diffusion adapter forwards only the latent and callback index to Easy Diffusion; it discards the rest of k-diffusion's callback state.

Checkpoint previews are raw decoded trajectory artefacts, not final post-filtered Easy Diffusion outputs. They are intended for trajectory inspection and later branching research.

Normal generation still follows the upstream call path when `trajectory.enabled` is false. No checkpoint writer or trajectory filesystem I/O exists in that mode.

## Not yet implemented / not yet validated

- real-GPU validation on GTX 1650 Super and Quadro RTX 4000;
- measured dense-capture VRAM/RAM/disk/timing overhead;
- sampler sigma/timestep and multistep solver-state capture;
- exact/approximate resume classification;
- branch DAG persistence and lineage operations;
- trajectory viewer/jog/scrub UI;
- insert-edit operations and image/latent interventions;
- GIMP/GEGL bridge;
- other Easy Diffusion engines or K80 support.

## Next gate

Run the fixed-seed GTX 1650 Super protocol in `TESTING.md` using SD1.4/SD1.x:

1. trajectory disabled baseline;
2. latent-only sparse capture;
3. preview-only sparse capture;
4. hybrid sparse capture;
5. dense every-step capture;
6. interrupted capture;
7. deliberately tiny storage-budget failure.

Verify final-image determinism where the sampler itself is deterministic, manifest/artifact integrity, bounded host/GPU memory, and expected timing cost. Do not begin exact-resume work until this capture gate passes.
