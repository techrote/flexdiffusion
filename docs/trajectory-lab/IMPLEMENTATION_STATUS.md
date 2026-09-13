# FlexDiffusion Trajectory Lab — Implementation Status

## Current branch state

The SD1.x/classic trajectory work now has a usable experimental UI layer on top of the capture plumbing:

1. bootstrap capture plumbing (`trajectory-capture-bootstrap`);
2. persisted checkpoint artefacts (`trajectory-artifacts`);
3. `Output After Step` with denoised/solver-state views;
4. fixed-seed multi-sampler comparison;
5. timestamp-paired review capture export.

The bootstrap and artefact branches have passing lightweight CI. The artefact branch still needs its complete fixed-seed GPU gate before exact-resume work should be treated as justified.

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
- bounded single-writer queue with explicit blocking backpressure;
- configurable queue depth and storage budget;
- per-artefact SHA-256, byte counts and relative paths in the manifest;
- all-or-error cleanup for a multi-file checkpoint if a staged/commit operation fails;
- pure unit tests for schedules, config parsing, recorder state, writer commit, hashes and storage-budget failure;
- lightweight GitHub Actions unit-test workflow;
- upstream baseline/callback semantics recorded;
- `Output After Step` for classic SD1.x txt2img fixed-step k-diffusion samplers;
- deferred post-sampling VAE decode of selected intermediate observations into the normal output panel;
- explicit per-image completed-step metadata;
- selectable intermediate representation: `denoised`, `solver_state`, or `both`;
- temporary callback bridge that preserves k-diffusion's original `denoised` callback value without changing sampler equations;
- representation-specific output labels and download filenames;
- fixed explicit seed as the normal comparison default;
- multi-sampler `Also run` UI that enqueues each selected sampler as a separate ordinary task with the same seed/settings;
- review-capture export that writes timestamp-matched PNG and editable Markdown metadata files for all visible experiment tasks;
- machine-readable raw JSON embedded in each review sidecar, with SHA-256/length descriptors for large embedded values;
- DOM screenshot capture with a deterministic canvas review-sheet fallback.

## Observed hardware behaviour

The first GTX 1650 Super run of the raw solver-state MVP succeeded and produced the expected progression: Step 20/30 looked like a recognisable but noisy/unfinished form of the Step 30 result, with noise progressively removed across later solver states.

That observation clarified an important product distinction. Raw solver state `x` is the state needed for future continuation/branching research, but it is not the best human-facing trajectory preview. k-diffusion's `denoised` value is the model's current clean-image estimate at the same callback boundary, so the UI exposes both representations.

The user is running the 4 GB GTX 1650 Super in low-VRAM mode to avoid overspill into system RAM. Wall-clock sampler timings on this machine therefore include backend memory-management effects as well as sampler arithmetic; they remain useful for this exact hardware configuration but should not be treated as sampler-intrinsic benchmarks.

Heun has already shown a useful aesthetic difference for the fixed cyberdino seed by avoiding an otherwise recurring floating-light artifact. Sampler diversity is therefore being treated as an experimental axis rather than a compatibility problem.

## Important semantic boundary

Persisting the latent does **not** yet imply that a sampler can resume exactly from that checkpoint.

The manifest therefore records `resume_fidelity: UNCLASSIFIED`. Stage 3 must determine the extra solver state required by each sampler and prove resume fidelity numerically.

For fixed-step k-diffusion samplers, callback index `N` observes the solver latent after `N` completed integration updates, immediately before update `N`. `denoised` is the model's clean estimate evaluated at that same boundary. `Output After Step` uses that completed-step interpretation and takes the final step only from the ordinary sampler return. The older trajectory-manifest display label remains provisional; raw callback indexes are retained for later migration.

Checkpoint previews and Output After Step observations are not final post-filtered Easy Diffusion outputs. Per-image face correction/upscaling remains final-only in the MVP.

Normal generation still follows the upstream call path when `trajectory.enabled` is false and `Output After Step` equals the final inference step. No intermediate CPU snapshot is created in that default case.

## Not yet implemented / not yet validated

- complete persisted-artefact GPU validation on GTX 1650 Super and Quadro RTX 4000;
- measured dense-capture VRAM/RAM/disk/timing overhead;
- GPU validation that dual-representation Output After Step leaves the ordinary final image unchanged;
- real-browser validation of Review Capture's preferred DOM screenshot path and matching dual-file download on the GTX 1650 Super setup;
- sampler sigma/timestep and multistep solver-state persistence in the trajectory manifest;
- exact/approximate resume classification;
- branch DAG persistence and lineage operations;
- trajectory viewer/jog/scrub UI;
- insert-edit operations and image/latent interventions;
- GIMP/GEGL bridge;
- Output After Step for DDIM/PLMS/adaptive samplers, img2img, or other Easy Diffusion engines;
- other Easy Diffusion engines or K80 support.

## Next gates

For the lightweight trajectory UI, continue fixed-seed SD1.4 sampler comparisons and use **Review capture** after each intentionally grouped test set. A valid review export should produce a same-basename `.png` and `.md`, include every visible task/request and output-step representation, and leave generation state untouched.

For the Output After Step prototype, use the SD1.4 `cyberpunk dinosaur mercenary` reference case in `OUTPUT_AFTER_STEP_MVP.md` with **Intermediate View = Both** and verify that:

1. each requested completed step emits a denoised estimate followed by its solver state;
2. the denoised estimate is visually clean enough to serve as the human-facing trajectory representation;
3. the final image matches the ordinary final render for the same request;
4. output labels/download names distinguish the paired representations;
5. dual CPU snapshot capture does not cause material GPU-memory growth or denoising stalls.

The broader persisted-artefact gate remains the fixed-seed GTX 1650 Super protocol in `TESTING.md`: baseline, sparse latent/preview/hybrid capture, dense capture, interruption and budget failure. Do not begin exact-resume work until that capture gate passes.
