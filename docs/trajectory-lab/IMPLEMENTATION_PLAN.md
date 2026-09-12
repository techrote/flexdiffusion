# FlexDiffusion Trajectory Lab — Implementation Plan

## Objective

Extend Easy Diffusion's SD1.x classic backend into a history-native diffusion research tool while preserving the familiar Easy Diffusion workflow. The first deliverable is trajectory capture and inspection. Exact branching follows only after sampler-state requirements are measured.

## Stage 0 — Baseline and guardrails

1. Record the upstream Easy Diffusion commit from which `trajectory-lab` starts.
2. Add deterministic baseline fixtures around classic SD1.x generation using a tiny/available test model where CI permits and a local full SD1.x fixture for GPU validation.
3. Record fixed-seed outputs/hashes or tolerant numerical/perceptual comparisons appropriate to the existing backend.
4. Add a trajectory capability structure rather than scattering feature checks through the UI.
5. Keep all trajectory settings default-off.

Acceptance:

- stock generation path is understood and regression-tested;
- trajectory-disabled execution has no intentional behavioural change;
- classic backend is explicitly the only trajectory-capable backend initially.

## Stage 1 — Persistent trajectory capture

### 1.1 Capture configuration

Extend render/task configuration with an optional trajectory specification containing:

- enabled;
- capture schedule;
- persistence mode (`preview`, `latent`, `hybrid`);
- output root/project name;
- preview format/quality;
- optional maximum checkpoint count / storage budget.

Compile schedule strings to exact step indices before the render begins.

### 1.2 Callback tap

Extend `sdkit_common.make_step_callback()` so the live latent can be handed to a TrajectoryRecorder independently of live preview.

Requirements:

- no change when trajectory recording is disabled;
- captured tensor must not keep the CUDA graph/live GPU buffer alive;
- use bounded writer queues so disk I/O cannot accumulate without limit;
- define backpressure policy explicitly (block, drop preview, or reject over-budget capture; never silently lose requested latent checkpoints);
- capture errors must be visible without corrupting the final normal render result.

### 1.3 Artefact format

Initial preferred latent format: a simple tensor container suitable for exact round-trip plus sidecar JSON metadata. Evaluate `safetensors` versus torch-native persistence; prefer the least coupled portable representation that preserves dtype and shape exactly.

Manifest should be written incrementally/atomically so an interrupted job remains inspectable.

### 1.4 Preview capture

Reuse the existing latent-to-image conversion path but persist only requested checkpoint previews. Do not make every callback perform a VAE decode.

### 1.5 Derived exports

Add:

- contact sheet;
- optional trajectory animation;
- CSV/JSON step timing table.

Acceptance:

- exact, range, and percentage schedules work;
- dense capture on the GTX 1650 Super does not leak VRAM;
- final image equals no-capture baseline for deterministic settings;
- interrupted trajectories remain self-describing.

## Stage 2 — Trajectory viewer

Implement a minimal UI extension in the existing Easy Diffusion style.

Features:

- trajectory toggle/settings near live-preview controls;
- checkpoint thumbnail strip;
- step/sigma/timing display;
- jog/scrub control;
- keyboard previous/next checkpoint;
- contact-sheet/export actions;
- no DAG/node UI yet.

The viewer reads the persisted manifest rather than relying solely on volatile browser state.

Acceptance:

- a completed run can be reopened and scrubbed after restarting Easy Diffusion;
- missing/corrupt individual previews degrade gracefully;
- latent-only checkpoints can be lazily decoded on request if practical.

## Stage 3 — Resume-state investigation

Before building arbitrary branching, instrument and classify the samplers actually used with SD1.x.

For each candidate sampler:

1. render a deterministic baseline;
2. capture at several steps;
3. stop/restart from the checkpoint using the minimal hypothesised state;
4. compare resumed latent/final output to uninterrupted baseline;
5. identify required additional state: sigma schedule, RNG state, prior denoiser outputs, solver history, Brownian state, etc.;
6. assign a resume-fidelity class.

Start with one deterministic sampler that gives the cleanest exact-resume path. Do not broaden until its semantics are proved.

Deliver a sampler compatibility matrix in-repo.

Acceptance:

- at least one sampler supports demonstrably exact or tolerance-bounded continuation from a saved checkpoint;
- unsupported/stateful samplers are labelled rather than approximated silently.

## Stage 4 — Branching core

Add immutable branch lineage.

### Operations

- `Continue from checkpoint` with identical settings;
- `Branch here` with prompt/negative-prompt delta;
- CFG/guidance changes where backend semantics permit;
- decoded-image/img2img fallback for samplers without exact latent resume.

### Data model

Each branch records:

- parent checkpoint UUID;
- edit recipe;
- inherited versus changed generation parameters;
- resume fidelity;
- resulting child run UUID.

Acceptance:

- parent history is never mutated;
- sibling branches are independently reproducible;
- branch comparison can align checkpoints by step/sigma/fraction of schedule.

## Stage 5 — History-native UX

Introduce the hidden-DAG/user-timeline interaction model.

Features:

- compact branch markers on trajectory rail;
- branch switcher;
- A/B comparison at same checkpoint or final output;
- undo/redo implemented as navigation through immutable history, not destructive mutation;
- bookmarks/favourites for promising states;
- optional advanced full graph inspector.

Acceptance:

- common workflows do not require understanding graph terminology;
- arbitrarily deep branch histories remain navigable.

## Stage 6 — Insert Edit v1

Implement explicit intervention nodes.

Initial operations:

1. prompt/conditioning change;
2. noise-strength perturbation;
3. mask + inpaint-style restart;
4. simple image-space operations such as curves/levels, blur/sharpen, colour transform and geometric crop/resize where semantics are clear;
5. simple latent arithmetic experiments behind an advanced/research flag.

Every intervention stores the operation and parameters, not just its output.

For decode -> image edit -> encode -> continue experiments, record:

- pre-edit checkpoint;
- decoded intermediate;
- filter recipe;
- re-encoded latent;
- noise/scheduler restoration method;
- measured divergence from unedited continuation.

Acceptance:

- interventions are reproducible;
- UI makes the edit location in diffusion time obvious;
- non-equivalent image-space round trips are labelled as such.

## Stage 7 — GEGL/GIMP filter bridge

Treat this as an optional adapter, not a dependency of the core tool.

1. Identify a deterministic subset of GEGL operations suitable for headless invocation.
2. Wrap them behind a small versioned filter recipe schema.
3. Expose curated operations in the `Insert Edit` dialog.
4. Record exact GEGL/GIMP version and filter parameters in branch metadata.
5. Avoid binding FlexDiffusion's history model to GIMP's own UI/history representation.

Candidate first operations:

- levels/curves;
- hue/saturation;
- gaussian blur;
- unsharp mask;
- edge/emboss-style filters for research;
- affine/warp only after coordinate semantics are solid.

## Stage 8 — Research tooling

Once branching is stable, add measurement rather than more novelty features first.

Possible experiments:

- quantify when macro composition stabilises across samplers;
- measure perceptual/structural change by step;
- determine latest step at which prompt concepts can alter object layout;
- compare prompt insertions at fixed sigma rather than raw step number;
- map which image-space edits survive subsequent denoising;
- cross-seed / cross-branch latent blending;
- branch fan-out sweeps from a single checkpoint;
- automatic branch scoring for structural similarity versus stylistic divergence.

Export experimental manifests and metrics separately from subjective user ratings.

## Stage 9 — Multi-GPU and async expansion

Only after the single-GPU semantics are correct:

- use existing Easy Diffusion multi-GPU scheduling for independent branch jobs;
- Quadro RTX 4000 as primary heavier worker;
- GTX 1650 Super as compatibility/minimum-VRAM worker;
- optional future remote worker protocol;
- K80 support only after its power/cooling/software stack exists and only where the classic SD1.x path remains practical.

No feature may require K80 support.

## Immediate implementation sequence

The next coding pass should be deliberately small:

1. add trajectory configuration types and schedule parser;
2. add pure unit tests for schedule compilation;
3. implement a `TrajectoryRecorder` with manifest-only/no-op storage path;
4. connect it to `sdkit_common.make_step_callback()` behind the default-off setting;
5. add latent/preview persistence;
6. test fixed-seed generation locally on GTX 1650 Super;
7. only then add the first UI controls.

This ordering isolates the most consequential backend semantics before UI work expands the surface area.
