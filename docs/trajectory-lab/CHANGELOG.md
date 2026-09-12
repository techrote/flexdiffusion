# Trajectory Lab changelog

## Bootstrap capture plumbing

- Added default-off trajectory request configuration.
- Added exact/range/percentage capture schedule compilation with one-based user semantics.
- Added an incremental atomic manifest recorder that stores checkpoint metadata without retaining CUDA tensors.
- Declared trajectory capability only on the classic Easy Diffusion backend.
- Routed trajectory settings to the backend only when enabled and supported.
- Tapped the classic `sdkit` denoising callback for selected checkpoint observation.
- Added interruption/completion/error status to manifests.
- Added unit tests and lightweight CI.
- Recorded the upstream baseline and first GPU validation protocol.

This bootstrap deliberately does **not** persist latents or previews yet and makes no resume-fidelity claims.
