# FlexDiffusion Trajectory Lab — Testing Notes

## Fast unit test

The trajectory core deliberately keeps its schedule/manifest tests independent of a full diffusion installation. The artefact writer is exercised with injected test serializers, so CI does not need CUDA or PyTorch.

From the repository root:

```bash
python -m unittest discover -s tests -p "test_trajectory.py" -v
```

The lightweight GitHub Actions workflow installs only `pydantic<2` and runs this command.

## First GPU validation gate

Run the following first on the GTX 1650 Super with one fixed SD1.4/SD1.x model and Easy Diffusion's classic backend. Repeat the accepted set later on the Quadro RTX 4000.

Keep constant unless the individual test says otherwise:

- exact model file/hash and VAE;
- fixed prompt and negative prompt;
- fixed seed;
- one deterministic sampler for the primary equality check;
- fixed step count;
- 512x512, batch 1;
- no post-filters;
- browser live preview disabled unless a test explicitly enables it.

Record wall time, peak VRAM, peak host RAM, final image hash and trajectory directory size.

### A. Upstream-equivalent baseline

Trajectory disabled.

Expected:

- no trajectory directory is created;
- behaviour and final image match the upstream-equivalent classic path.

### B. Sparse latent-only capture

Example:

```json
{
  "trajectory": {
    "enabled": true,
    "capture_schedule": "1,5,10,50%,100%",
    "persistence_mode": "latent",
    "writer_queue_size": 2
  }
}
```

Expected:

- final output matches A for deterministic settings;
- every requested checkpoint has one `.safetensors` latent;
- manifest paths exist and SHA-256 hashes match;
- tensor shape/dtype/device-at-capture metadata is plausible;
- persisted safetensors reopen with key `latent` and expected CPU tensor shape/dtype;
- VRAM does not grow with checkpoint count.

### C. Sparse preview-only capture

Use the same schedule with `persistence_mode: preview`.

Expected:

- decoded files exist for every requested step;
- no latent files exist;
- browser live preview remains disabled;
- final image is unchanged for deterministic settings;
- timing cost reflects VAE decodes only at selected trajectory checkpoints.

### D. Hybrid capture and live-preview reuse

Use `persistence_mode: hybrid`. Run once with browser live preview disabled and once with its interval aligned to the capture schedule.

Expected:

- each checkpoint has both latent and preview metadata;
- aligned browser live preview and trajectory preview reuse one decoded image path in the backend rather than requesting two VAE decodes at the same callback;
- final image remains unchanged.

### E. Dense every-step capture

Use `1-<N>:1` and a deliberately small writer queue (`writer_queue_size: 1`).

Expected:

- queue backpressure may increase wall time but no requested checkpoint is silently dropped;
- peak GPU memory remains bounded because checkpoint tensors are detached to CPU immediately;
- host RAM remains bounded by the queue plus current render state rather than scaling with all steps;
- final manifest is internally complete.

### F. Interrupted capture

Cancel during a run after several checkpoints.

Expected:

- already queued artefacts finish writing before the recorder closes;
- manifest status is `interrupted` (or `complete_with_capture_errors` only when an actual capture error occurred);
- checkpoint files referenced by the manifest exist and hash correctly;
- no `.tmp` files remain.

### G. Storage-budget failure

Set an intentionally tiny positive `storage_budget_mb`.

Expected:

- a checkpoint that would exceed the budget is staged then rejected without committing a partial artefact set;
- manifest records the checkpoint artefact error explicitly;
- final image generation continues;
- overall trajectory status records capture errors;
- no orphan committed files from the rejected checkpoint remain.

### H. Browser live-preview independence

Repeat a sparse latent-only run with browser live preview on/off.

Expected:

- latent checkpoint hashes match between the two runs at the same callback steps for deterministic settings;
- browser preview settings do not change which trajectory checkpoints are selected.

## Resume gate (later)

A saved latent is **not** currently labelled restartable. Before branching is enabled, each supported sampler must be tested by resuming from recorded state and comparing against an uninterrupted reference numerically. Stateful/multistep samplers may require sigma/history/RNG information beyond the latent and callback index. Fidelity must be classified from evidence, not visual similarity.
