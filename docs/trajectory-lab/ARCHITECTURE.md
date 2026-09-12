# FlexDiffusion Trajectory Lab — Architecture

## Purpose

FlexDiffusion is an Easy Diffusion fork focused initially on Stable Diffusion 1.x / classic `sdkit` generation. The first research target is to make the denoising trajectory a first-class editable object: capture intermediate states, inspect them, branch from them, and later insert controlled edits before continuing generation.

Primary development GPUs are GTX 1650 Super 4 GB and Quadro RTX 4000 8 GB. K80 support is explicitly non-blocking and deferred.

## Scope rule

The first implementation MUST NOT attempt to support every Easy Diffusion backend. Initial support is limited to the classic Easy Diffusion / `sdkit` path used for SD1.x-style models. Normal Easy Diffusion generation must remain unchanged when trajectory mode is disabled.

## Key architectural finding

The existing classic backend already exposes the hook we need. `ui/easydiffusion/backends/sdkit_common.py::make_step_callback()` receives the live latent tensor (`x_samples`) on every denoising callback, stores it as `context.partial_x_samples`, and optionally decodes it for Easy Diffusion's live preview.

This means Phase 1 trajectory capture can be implemented in the Easy Diffusion fork without immediately forking `sdkit` itself. The callback wrapper is the natural interception point for:

- exact step selection;
- asynchronous/off-thread persistence;
- latent snapshots;
- decoded previews;
- per-step metadata;
- future branch checkpoints.

A deeper `sdkit` fork should only be introduced if exact resume or sampler-specific state cannot be captured cleanly through the existing callback API.

## System layers

```text
Easy Diffusion UI
  |
  +-- normal generation controls
  +-- Trajectory Lab UI (opt-in)
        +-- capture schedule
        +-- scrub/jog timeline
        +-- branch here
        +-- insert edit (later)
        +-- compare branches

Easy Diffusion task system
  |
  +-- RenderTask
        |
        +-- classic backend adapter
              |
              +-- trajectory callback tap
                    +-- preview capture
                    +-- latent capture
                    +-- metadata capture
                    +-- branch manifest update

Trajectory store
  +-- immutable run manifest
  +-- checkpoint metadata
  +-- latent artefacts
  +-- decoded previews
  +-- edit recipes
  +-- history DAG
```

## Trajectory data model

A trajectory is immutable. Editing an earlier point never rewrites later history; it creates a descendant branch.

### Run

Required fields:

- stable run UUID;
- model identity/path plus file hash where practical;
- VAE identity;
- prompt / negative prompt;
- seed;
- sampler;
- step count;
- dimensions;
- CFG/guidance parameters;
- software/version identifiers;
- parent branch/edit reference if this run is derived.

### Checkpoint

Required fields:

- checkpoint UUID;
- run UUID;
- displayed step index;
- sampler timestep/sigma if available;
- total steps;
- latent tensor shape/dtype;
- latent artefact path/hash if persisted;
- preview artefact path/hash if persisted;
- parent checkpoint;
- resume fidelity classification.

### Edit node

Future edit nodes should record an immutable recipe, not only the resulting pixels. Candidate types:

- prompt/negative-prompt delta;
- CFG/guidance change;
- mask/inpaint operation;
- decoded image-space filter;
- latent-domain operation;
- noise injection;
- blend/cross-trajectory operation.

## Capture schedules

Support these syntaxes from the start:

1. exact steps, e.g. `5,10,15,20,30`;
2. ranges, e.g. `1-10:1,10-30:2,30-50:5`;
3. fractional checkpoints, e.g. `10%,25%,50%,75%,100%`.

The schedule must compile to an explicit set of step indices before generation begins so captured results are reproducible.

## Persistence modes

- `preview`: decoded image only;
- `latent`: latent only;
- `hybrid`: both.

Trajectory mode MUST avoid retaining a chain of GPU tensors. At a capture point, a latent selected for persistence should be detached/copied away from the live graph/device promptly, then written asynchronously where safe. The 4 GB GTX 1650 Super is the minimum-memory design target.

## Resume fidelity classes

Do not call every saved latent an "exact resume" checkpoint.

Different samplers carry different hidden state. A restart may require more than `(latent, step)`:

- deterministic single-step solvers may only require latent + remaining schedule;
- ancestral samplers may also require reproducible RNG/noise-sampler state;
- multistep solvers may require previous derivatives/model outputs/history;
- SDE samplers can require Brownian/noise state.

Each sampler must therefore be classified experimentally:

- `EXACT`: resumed continuation is bitwise or numerically equivalent within a declared tolerance;
- `STATEFUL_EXACT`: exact only when additional captured solver/RNG state is restored;
- `APPROXIMATE`: visually/semantically useful branch, not an identical continuation;
- `IMAGE_RESTART`: decoded image -> img2img fallback only.

Phase 1 does not promise exact resume. Phase 2 must prove it sampler by sampler.

## First exact-branch target

Prefer one deterministic, structurally simple SD1.x sampler for the first exact-resume proof. Do not begin with the broad sampler matrix. Once one reference path works and is tested, generalise deliberately.

## UI principle

The internal representation is a DAG; the default user interface is a timeline.

```text
0----5----10----15----20----25----30
               |
               +-- edit --20'---25'---30'
                        |
                        +-- branch B ...
```

Default UI should expose:

- checkpoint thumbnail strip;
- scrub/jog control;
- current branch;
- compact child-branch markers;
- `Branch here`;
- later, `Insert edit`.

The full history graph can be an advanced inspector. The project should not become a node-editor-first application.

## Image-space intervention research

A future image-space intervention may follow:

```text
latent at step N
  -> VAE decode
  -> image-space operation
  -> VAE encode
  -> restore appropriate noise level / continuation state
  -> continue denoising
```

This is deliberately treated as an experiment, not assumed semantics. Decode/re-encode loss, latent scale, scheduler noise level, and structural divergence must be measured.

GIMP/GEGL integration, if pursued, should initially expose a curated deterministic subset through immutable filter recipes rather than embedding the whole GIMP UI.

## Compatibility policy

### Primary

- SD1.4 / SD1.x-family checkpoints and merges;
- Easy Diffusion classic backend;
- GTX 1650 Super 4 GB;
- Quadro RTX 4000 8 GB.

### Deferred

- K80 / Kepler;
- Easy Diffusion Forge/v3.5;
- sdkit3/v4;
- SDXL/Flux/etc.;
- remote worker scheduling.

Deferred targets must not distort the initial design beyond keeping artefact formats and backend capability declarations extensible.

## Non-negotiable tests

1. Trajectory disabled: output remains identical to upstream for fixed model/seed/settings.
2. Capture enabled: final output remains identical unless a documented backend limitation proves otherwise.
3. Capture schedule parser has boundary/adversarial tests.
4. No unbounded GPU-memory growth with dense capture.
5. Metadata is sufficient to reproduce a run or explicitly states why it is not.
6. Resume claims are backed by numerical comparison, not visual similarity.
7. Interrupted runs leave recoverable manifests and never silently masquerade as complete trajectories.

## Upstream strategy

Keep `main` close to `easydiffusion/easydiffusion`. Development occurs on `trajectory-lab` and subsequent feature branches. Core modifications should remain narrow and clearly separated from Trajectory Lab modules so upstream rebases remain tractable.
