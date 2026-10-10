# Circuit bend next pass

## Plan

Implement the bounded next pass agreed on 10 October 2026:

1. Extend Euler/Heun noise with a continuous memory control and fade envelopes.
   Retain exact legacy fresh/held recipes, private randomness, and the ordinary
   path when disabled or churn is zero. Resolve percentage windows against the
   actual native update count. Keep rectangular windows as the default.
2. Add a small comparison grid. Preflight a pair of parameter axes, then queue
   one normal task per cell and seed, without multiplying by Also run. Preserve
   all ordinary outputs and metadata. Show final results per seed and allow a
   cell's full settings to be restored. Export the grid recipe for replay.
3. Add an experimental PLMS history freeze with a blend strength. Freeze the
   three prior predictions after native warm-up and blend them into the inputs
   of later updates. Do not alter native history length or add model calls.
   Reuse the window, pulse and fade controls; strength zero is native PLMS.
4. Verify configuration/metadata round trips, variance-preserving noise memory,
   envelopes, history ownership, native equivalence, task routing and grid UI.
   Sample images use the render policy below when the Titan is available.

Branching, spatial noise masks, UniPC correction gain and DDIM eta are later
work; they are not included in this bounded pass.

## Sample render policy

Future artistic samples and feature-validation renders use the **Titan RTX at
512 x 512**, with **generation seeds 2, 3, 4 and 5**. Noise seed is an independent
parameter and must also be recorded. Use an existing local checkpoint.

Before GPU work, inspect active Codex chats for GPU jobs and check the Titan's
running processes/utilization. An idle moment during another chat's benchmark
or cooling pause does not make the GPU available. Do not start while another
chat is using or reserving the Titan. Do CPU and browser work first, then check
again. Do not stop another chat's process or change GPU power/clock settings.

## Acceptance

- Old recipes restore and render the same way; neutral settings equal native.
- Memory 0 is fresh noise, memory 1 is held noise; intermediate values preserve
  unit marginal variance in expectation, without per-sample renormalization.
- Fade and percentage behavior is identical in Python and the UI.
- PLMS freezes owned copies, keeps live history intact, and adds no model calls.
- A grid uses the same prompt/settings for all cells, varies only named axes
  and seeds, and can restore or export each recipe. Invalid grids enqueue nothing.
- Numerical divergence retains finite partial output and an explicit warning.

## Implemented

- `CircuitBendOptions` retains legacy fresh/held recipes and adds memory,
  percentage windows, fades, and the PLMS history mode. Both samplers continue
  through their existing per-render adapters.
- Noise uses an AR(1) process with variance-preserving coefficients. Endpoint
  values reproduce fresh and held noise exactly. Fades scale churn without
  introducing new global random draws.
- PLMS snapshots its three-entry history after warm-up, then blends frozen/live
  history only at selected pulses. Live history is not modified. Captures record
  the frozen-at iteration and effective history blend.
- The grid has two axes, explicit seeds, optional baseline cells, preflight,
  final-image cells, restore, and JSON recipe import/export. Grid membership and
  bend options survive request/image metadata and Review Capture.
- The two sample-validation CLIs default to 512×512/seeds 2–5, pin CUDA to the
  Titan UUID, require a recent Codex ownership check and refuse unknown compute
  processes. Explicitly verified idle validation servers are checked via `/ping`
  before each render. GPU power and clock settings are untouched.

## Validation on 10 October 2026

- 60 Python tests passed with the installed classic runtime. Coverage includes
  native neutral equivalence, model-call counts, repeatability, noise variance
  and correlation, private RNG, percentage/fade boundaries, owned frozen history,
  metadata persistence and the GPU ownership gate.
- JavaScript tests passed for grid preflight, independent task expansion,
  generation seeds, baseline handling, recipe round trips, examples, final-image
  selection and text metadata. Syntax and whitespace checks passed.
- Browser fixture checks passed: invalid grids queue zero tasks; a 28-task grid
  includes seeds 2–5; all simulated final cells appear; selecting a seed-4 cell
  restores its exact memory/noise/fades; PLMS 30-step percentages use 31 native
  updates; unsupported sampler requests omit bends; export and file import work.
  The fixture calls the real plugins but cannot start GPU work.
- No GPU sample renders were launched during the initial implementation. The active
  `Summarise FlexDiffusion progress` chat was benchmarking the Titan during
  implementation. Its Titan sweep later completed, but its model-server process
  remained resident. Existing first-pass GPU evidence does not validate the new
  memory/fade/history features; their full-model artistic results remain to be
  checked in a free Titan session, subsequently completed below.

## Titan follow-up: 512 x 512 validation

After the user confirmed that the Titan was free, active chats and GPU processes
were inspected again. The other project chat was reviewing V100 results. The
resident Titan validation server was Online (idle); it was left running and its
status was checked before every render. WDDM desktop connections were inspected
separately. No GPU clocks or power settings were changed.

Evidence directory: `C:/LocalRepos/flexdiff-runtime/evidence/circuit-bends-512-20261010`.
`results.json` contains requests, bend parameters, image hashes and observation
metadata; `preflight.json` records the GPU and ownership check. The runner exited
successfully. All saved final hashes and sample dimensions were independently
checked after completion.

- Stable Diffusion 1.4, classic backend, Titan RTX, 512 x 512, seeds **2, 3, 4, 5**.
- Prompt: `cyberpunk dinosaur mercenary`; empty negative prompt; CFG 7.5;
  20 requested updates; one image per call.
- Euler and Heun: churn 20, noise 1.25, memory **0 / 0.5 / 1**, separate noise
  seed **123**. PLMS: frozen-history blend **0.5 / 1**; no added noise seed.
- Every bend used window 25–75%, fade-in 20%, fade-out 25%, every iteration.
  Intermediate observations were saved at iterations 5, 10 and 15.
- **88 generation calls**, **44 final images**, **96 intermediate images** and
  three comparison sheets. All 12 neutral settings matched native final pixels;
  all 32 bent configurations changed native output and repeated exactly. No
  capture or numerical-divergence warnings occurred.
- **61 CPU tests passed**, including idle-server, busy-server and unreachable
  status checks. This validates the direct generation path; it does not add a
  full application HTTP/UI rendering test or a new DDIM/UniPC GPU sweep.

Visual inspection of all three comparison sheets suggests two useful ranges.
PLMS freeze retains scene composition while increasing contour contrast, fine
detail and local distortion, especially at full strength. Euler/Heun memory 0
leaves recognizable silhouettes under heavy interference. Memory 0.5 and 1 at
these strong settings largely replace the subject with saturated abstract
texture. Using noise seed 123 across generation seeds creates a recurring
texture pattern; generation-seed diversity becomes less visible. These outputs
are retained as intentional experimental options. A lower churn/noise sweep is
the next useful calibration for finding the transition from readable scenes to
abstraction; this run does not establish that threshold.

## PLMS all-step rerun

User-requested rerun on 10 October: same seeds 2–5, 512 x 512, prompt and
ordinary/half-freeze/full-freeze recipes. `validate_circuit_bends.py --samplers
plms --all-steps` exports steps 1–19 as clean estimates and the authoritative
final return as step 20. All 240 step PNGs were verified, and all twelve finals
match the preceding sample run exactly. Four neutral comparisons and eight
repeatability comparisons passed. Evidence, per-sequence contact sheets and a
local HTML viewer are in
`C:/LocalRepos/flexdiff-runtime/evidence/plms-all-steps-20261010`.

The requested 2000 MHz core / 1 V / 8400 MHz VRAM settings were not applied:
the GPU clock API returned Insufficient Permissions, memory locking returned
Not Supported, and the modern memory-offset setter also returned Insufficient
Permissions. Installed NVML controls expose no absolute 1 V setter. Rendering
continued with the existing settings; loaded telemetry reported 1200 MHz core,
6500 MHz memory, 240 W power limit, and peak sampled temperature 33 C. Attempts
and telemetry are retained alongside the images. No power setting was changed.

Use [SAMPLER_OUTPUT_AND_BENDS.md](SAMPLER_OUTPUT_AND_BENDS.md) for controls and
sample runner instructions. The existing server was not restarted, so backend
changes require a restart and UI changes require a refresh before use.
