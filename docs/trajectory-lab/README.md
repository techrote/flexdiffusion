# Trajectory Lab

FlexDiffusion's initial research track for making the SD1.x denoising trajectory a first-class inspectable and eventually editable object.

Read in this order:

- `ARCHITECTURE.md` — design boundaries and history model;
- `IMPLEMENTATION_PLAN.md` — staged roadmap;
- `BASELINE.md` — upstream starting point and callback semantics;
- `IMPLEMENTATION_STATUS.md` — what is implemented now;
- `TESTING.md` — unit and first GPU validation gates.

The current target is Easy Diffusion's classic `sdkit` path with SD1.4/SD1.x checkpoints on GTX 1650 Super and Quadro RTX 4000. Other backends and K80 support are deferred.
