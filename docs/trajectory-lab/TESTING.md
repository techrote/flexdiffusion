# FlexDiffusion Trajectory Lab — Testing Notes

## Fast unit test

The trajectory core deliberately avoids importing torch so schedule/config/manifest behaviour can be checked independently of a diffusion installation.

From the repository root:

```bash
python -m unittest discover -s tests -p "test_trajectory.py" -v
```

The lightweight GitHub Actions workflow installs only `pydantic<2` and runs this command.

## First GPU validation gate

Before latent persistence or UI work is considered stable, run the following on the GTX 1650 Super with the same SD1.x checkpoint and Easy Diffusion environment.

### A. Upstream-equivalent baseline

- classic backend;
- trajectory disabled;
- fixed model hash;
- fixed VAE;
- fixed prompt/negative prompt;
- fixed seed;
- fixed sampler and step count;
- 512x512, batch 1;
- no post-filters.

Record final image bytes/hash and generation timing.

### B. Manifest-only capture

Repeat the identical request with:

```json
{
  "trajectory": {
    "enabled": true,
    "capture_schedule": "1,5,10,50%,100%",
    "persistence_mode": "latent"
  }
}
```

The current implementation does not yet persist the latent despite the requested future persistence mode; it records manifest checkpoint metadata only.

Expected:

- final output is identical to A for deterministic settings;
- manifest contains the resolved one-based capture steps;
- each recorded checkpoint has the expected latent shape/dtype/device;
- no retained sequence of CUDA tensors exists;
- no meaningful VRAM increase beyond normal run-to-run noise.

### C. Dense metadata capture

Use `1-<N>:1` for the full step count. Verify:

- no GPU-memory growth across the run;
- manifest remains valid if generation is interrupted;
- ordinary live-preview behaviour remains independent of trajectory capture.

## Later gates

Latent persistence will add numerical round-trip checks and bounded queue/backpressure tests. Resume work must compare resumed latent/final output against uninterrupted reference runs and classify each sampler by demonstrated fidelity rather than appearance.
