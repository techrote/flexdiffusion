# FlexDiffusion Trajectory Lab — Baseline Record

## Upstream starting point

The `trajectory-lab` work started from Easy Diffusion upstream commit:

- `9cd95259381921c9a8be433cf02c9f5db68ad7a7`
- `Fix missing ensure_torchruntime.py to startup scripts (#2060)`
- upstream date: 2026-09-11

The fork default branch remains intended to track upstream closely; trajectory work is isolated from it.

## Classic SD1.x interception point

For the initial SD1.4/SD1.x target, the classic backend path is deliberately preferred over Forge/v4.

The relevant call chain is:

```text
/render request
  -> RenderTaskData / GenerateImageRequest
  -> tasks/render_images.py
  -> ed_classic.generate_images
  -> sdkit_common.generate_images
  -> sdkit.generate.generate_images
  -> sampler callback(x_samples, i)
  -> sdkit_common.make_step_callback
```

`sdkit_common.make_step_callback()` already receives the live latent (`x_samples`) on each sampler callback. Existing Easy Diffusion live-preview behaviour optionally decodes that state. FlexDiffusion therefore does not need to fork `sdkit` merely to observe trajectory checkpoints.

## Callback indexing convention

The classic k-diffusion adapter forwards `info["i"]` from k-diffusion as the callback index. FlexDiffusion treats this callback index as zero-based and presents trajectory capture steps to users as one-based (`display_step = callback_index + 1`).

This convention is frozen for the Stage 1 capture format. Sampler-specific resume work must verify, rather than assume, that callback boundaries correspond to restartable solver states.

## Compatibility boundary

Initial trajectory capability is declared only by `ed_classic`.

Deferred backends must not receive trajectory-specific keyword arguments unless they explicitly declare a compatible capability. This keeps normal Easy Diffusion behaviour and upstream backend adapters isolated from the experiment.

## Current implementation boundary

The first recorder is intentionally **manifest-only**. It proves:

- request configuration plumbing;
- schedule compilation;
- callback interception;
- incremental atomic manifest updates;
- interruption/completion status handling;
- no retention of live GPU tensor references.

Latent/preview persistence, bounded writer queues, storage budgets, exact-resume semantics and UI controls remain subsequent work.
