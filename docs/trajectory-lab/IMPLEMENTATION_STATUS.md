# FlexDiffusion Trajectory Lab — Implementation Status

## Current branch state

The first narrow backend plumbing pass is implemented on the trajectory development branch.

Completed:

- default-off `TrajectoryData` render configuration;
- one-based exact/range/percentage capture schedule compiler;
- incremental atomic manifest recorder;
- classic-backend capability declaration;
- request plumbing that sends trajectory options only to capable backends;
- classic `sdkit` callback tap that observes live latent tensors without retaining them;
- interrupted/failed/completed manifest status handling;
- pure unit tests for schedules, config parsing and manifest recording;
- lightweight GitHub Actions unit-test workflow;
- upstream baseline/callback semantics recorded.

Not yet implemented:

- latent tensor persistence;
- preview persistence independent of Easy Diffusion live preview;
- bounded asynchronous writer queue/backpressure;
- storage budget enforcement beyond checkpoint count;
- UI controls/viewer;
- resume/branch semantics;
- sampler-state classification.

## Important semantic boundary

The current recorder is intentionally **manifest-only**. A captured checkpoint currently proves that the requested callback boundary was observed and records tensor shape/dtype/device metadata. It does not yet claim to contain a restartable latent artefact.

Normal generation takes the upstream path when `trajectory.enabled` is false. Trajectory-specific kwargs are not sent to backends unless they explicitly advertise trajectory capture capability.

## Next implementation pass

1. persist selected latents by detaching and copying them to CPU at capture points;
2. move disk writes to a bounded worker queue;
3. implement explicit backpressure/error policy;
4. persist requested decoded previews only at capture points;
5. add hashes/paths to checkpoint manifest records;
6. perform a fixed-seed no-capture vs capture GPU comparison on GTX 1650 Super;
7. measure dense-capture VRAM/RAM/disk behaviour before adding UI.
