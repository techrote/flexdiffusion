# Sampler intermediate output and experimental noise

Classic SD1.x txt2img now supports sparse intermediate output for DDIM, PLMS,
all five UniPC presets, and the existing nine fixed-step k-diffusion samplers.
Euler and Heun also have an opt-in **Circuit bend** noise panel, PLMS has
history freezing and second-seed mixing, and DDIM has seeded stochastic noise. These are
observations and artistic interventions; saved latents are not exact-resume
checkpoints.

Use **Capture iterations** as before, for example `2, 10, 19-20` with 20 inference
steps. This produces three clean estimates and the ordinary final image. The
optional endpoint checkbox adds raw states at the first and last selected
intermediate boundaries. A selected true final boundary uses the ordinary final
image instead of a duplicate solver-state image.

## Meaning of the images

| Family | Clean estimate | Raw endpoint | Iteration meaning |
| --- | --- | --- | --- |
| DDIM | Native model clean estimate | Input to the next update | N completed updates, before update N |
| PLMS | Native history-weighted solver clean estimate | Input to the next update | N completed updates, before update N |
| UniPC | Predictor clean estimate | Corrected solver state | Return from completed update N |
| k-diffusion | Native denoised estimate | Callback solver state | N completed updates, before update N; active churn may already have added noise |

UniPC's predictor estimate is evaluated before correction and does not describe
a fresh model evaluation on the corrected state. Capturing it adds no model
evaluation. The native UniPC callback changes phase after warm-up; the adapter
observes update returns instead. All families use the ordinary return for the
final image.

DDIM/PLMS retain the installed native timetable. A 30-step request uses **31
updates**. With capture `10,30`, both boundaries are retained and the final is
labelled `Step 31/31 · Final · 30 requested`. A lone capture value equal to the
requested inference count preserves the existing final-only shortcut. Some
native schedules (including 9 and 27) contain an invalid timestep and are
rejected with a suggested valid count. Counts 20, 25, 40 and 50 are straightforward
starting points. Alternatively, enable Circuit bend and choose **Exact steps**:
this produces exactly the requested number of unique bounded timesteps for DDIM
or PLMS, including 16 and 27. This is an explicit change of schedule; native
remains the default. Safe divisors such as 20 retain identical timesteps.

UniPC requires at least its preset's order: TU needs 2; SNR, TQ and TU 2 need 3;
SNR 2 needs 1. **UniPC SNR 2 is experimental and limited to one image per batch**
because the installed `vary_coeff` implementation has a batching defect. The
other presets support batching. The `_2` names do not consistently mean order 2.

## Circuit bend controls

Select classic txt2img **Euler** or **Heun**, tick **Experimental noise**, then
open **Bend controls**. Changes feed into subsequent iterations.

| Control | Range and behavior |
| --- | --- |
| Churn strength | 0–100; 0 follows ordinary upstream sampling exactly; initial value 10 |
| Noise amount | 0–4; initial value 1; larger values can fragment the result |
| First and last iteration | Inclusive, numbered from 1; blank last means final |
| Pulse every N iterations | Apply churn at the first selected iteration and every N iterations thereafter |
| Noise seed | 0–4294967295; separate generator from the image's initial noise |
| Noise memory | 0–1; 0 generates fresh noise at each pulse, 1 holds the same field, intermediate values evolve correlated noise |
| Window units/preset | Absolute iterations or percentages of the native schedule; Early, Middle, Late and Whole render presets |
| Fade in/out | Each 0–100% of the selected window span; linear ramps, 0 disables that fade |

Noise memory uses `next = memory * previous + sqrt(1 - memory^2) * fresh`.
This preserves unit marginal variance in expectation. It does not normalize
each tensor, and the fresh and held endpoints retain the original implementation's
noise sequence. Legacy recipes with `pattern: held` still restore as memory 1.
The noise field advances only when an active pulse adds noise.

Fades scale churn for Euler/Heun and history blend for PLMS. Their endpoints
reach zero; overlapping fades use the smaller weight. A one-iteration window is
a full pulse. Percentages use ceiling against actual native update count,
clamped to at least iteration 1. Thus 25–75% of a 30-request/31-actual PLMS render
resolves to iterations 8–24. The UI reports the window and weighted pulse count.

Select **PLMS** to expose **Experimental history / seed mixing**; choose
**Freeze own history** and adjust **Bend strength**. At the first selected iteration with three previous predictions, the
adapter takes owned copies of that history. A blend of 1 substitutes those
copies in later selected updates; 0 follows native PLMS; intermediate values
blend live and frozen history. Native live history continues to advance. Warm-up,
history length and model-call count are unchanged. A fade may start at zero
while the history is frozen at the selected boundary. A window entirely before
the fourth update has no history-freeze effect.

### PLMS second-seed mixing

Select **History source → Second seed**. The normal generation seed initializes
the main branch; Second seed initializes an independent PLMS branch with the
same prompt, guidance and schedule. Both advance together, requiring about
twice as many model calls. A zero-strength recipe skips the second branch.
Batched images use second seed + image index, matching the usual seed convention.

**History (glitch)** substitutes/blends the donor's previous predictions into
the main solver's history inputs. The live main history still advances normally.
Mismatched history can cause ghosting, abrupt structure changes and instability;
these finite results are kept. **Trajectory feedback (glitch)** blends the latent after the
update, its clean estimate and its returned prediction. This influences future
steps. It does not alternate independent finished images: blend zero resumes
the evolving main branch, including effects from earlier mixing. Constant full
trajectory mixing reproduces the second seed's trajectory.

**Independent trajectories** keeps both seed trajectories untouched. Each
exported step blends their corresponding latent/clean estimates without feeding
that blend back into either solver. This is the best choice for reliable A/B
interlacing: low/high 0/1 and block length 1 alternate the two seed sequences at
every update. Smooth oscillation can repeatedly return to either original seed.
Feedback remains available because its drift and eventual convergence onto the
donor path can also be useful artistically.

| Movement | Controls |
| --- | --- |
| Constant | Uses Ending / high blend throughout the window |
| Progressive | Interpolates from Starting / low to Ending / high across the window |
| Oscillate smoothly | Cosine movement between low and high; Cycle length is iterations per full cycle |
| Interlace hard blocks | Alternates low/high blocks; Block length 1 alternates every iteration |

Swap low/high to reverse the movement. Blend amount is multiplied by Bend
strength and the existing window, pulse and fade envelope. Start with history
strength 0.2–0.4 for a less disruptive experiment, or trajectory strength 1 for
obvious switching in independent mode. The first history-mixing update has no past predictions yet.
Settings restore/export with recipes; observations record donor seed, target,
curve and effective blend weight.

### DDIM stochastic controls

Select **DDIM**, enable **DDIM noise / exact steps**, and open Bend controls.
**Eta 0** is deterministic native DDIM; **eta 1** uses its full stochastic
variance. The window/pulses/fades scale eta at each update. **Noise amount**
scales only the injected noise, while **Noise memory** selects fresh, correlated
or held fields. Both are deliberately available beyond conventional DDIM;
noise amount zero with nonzero eta still changes the update direction. Use eta
zero (and the same schedule) for an ordinary deterministic reference.

The private Noise seed makes the stochastic branch repeatable. Memory advances
only when the selected update has nonzero variance. Native DDIM equations,
guidance and parameterization handling remain upstream. Sampler-local function
bindings replace only the noise provider and, for Exact steps, the timetable
builder; there is no shared module patch. The final step retains the native
return and all intermediate captures remain available.

The usual churn cap per update remains in place. A noise amount of zero still
changes the integration noise level when churn is nonzero; use **churn 0** or
untick the switch for an exact ordinary render. High amounts and held noise are
kept available even when they produce rough or abstract results. Non-finite
values stop the bend and return the last finite state with a visible warning;
already captured observations are retained. Cancellation also retains captured
observations. Interrupted results are labelled as partial states.

Other samplers in an **Also run** comparison receive no incompatible bend
settings from the UI. Direct API requests that enable bends on unsupported samplers are rejected.
The controls restore with task settings, survive request/JSON/text metadata
export, and appear in Review Capture. Observation metadata includes phase,
estimate basis, native timestep or sigma, and requested/actual step counts.

## Comparison grid

### Interlacing sampler types: PLMS ↔ DDIM

Select either **PLMS** or **DDIM**, enable Circuit bend, then tick **Interlace
sampler types: PLMS ↔ DDIM** inside Bend controls. This selects `sampler_mix`,
separately from the earlier second-seed experiments. It starts with Exact steps
and DDIM eta 0. Both solvers share one model, prompt, generation seed and alpha
timetable. This first pair supports classic epsilon-prediction txt2img models;
other parameterizations/backends and img2img are rejected.

- **Alternate one live latent**: with first sampler PLMS and block length 1,
  PLMS makes update 1, DDIM receives that latent for update 2, PLMS receives the
  result for update 3, and so on. Block length 2 gives PLMS, PLMS, DDIM, DDIM.
  Reversing the first sampler reverses the order. No second-seed run is involved.
- **Coupled parallel trajectories**: both solvers advance their own latent from
  the same initial noise at every timestep. After each update they simultaneously
  exchange a fraction `q` of their new latents: `P'=(1-q)P+qD`,
  `D'=(1-q)D+qP`. At 0 they remain independent; at 0.5 they join; at 1 they swap.
  Sampling calls execute sequentially on one GPU, with both trajectories held
  in memory. There is no requirement for concurrent GPU jobs or a second model.

Coupling can stay constant, increase progressively, or oscillate between zero
and its selected amount. Cycle iterations sets the oscillation period. Window,
pulse and fade controls multiply exchange strength and DDIM eta. In alternate
mode the selected first sampler runs outside active pulses; switching is discrete
even with fades. Export the PLMS branch, DDIM branch, or equal latent blend in
coupled mode. Branch output selection does not change how they interact.

**PLMS memory** deliberately exposes three behaviours:

| Policy | Alternating | Coupled |
| --- | --- | --- |
| Shared predictions | Append epsilon predictions from every update, including DDIM | Mix the current branch predictions by the exchange amount before appending to PLMS history |
| Keep stale PLMS history | Keep only predictions from PLMS updates, even across skipped timesteps | Retain PLMS's prediction despite exchanging its latent |
| Reset | Clear history at a sampler switch; PLMS repeats warm-up when returning | Clear history after nonzero exchange; PLMS repeats warm-up next time |

Shared uses an instance-local proxy to observe DDIM's existing model prediction;
it does not add an inference call. Stale history intentionally violates the
usual spacing assumption and is retained for glitch art. Reset costs additional
PLMS warm-up evaluations and can be less visibly glitchy. Coupled mode costs
roughly twice the sampling work. These interventions alter subsequent denoising;
they are not blends of final decoded images.

Mixed captures use the state **after update N**, plus the clean estimate computed
by that update (or its exchanged counterpart). Each carries `active_sampler`,
`sampler_iteration`, `history_length`, `mix_history` and `exchange_weight`.
Ordinary reference captures retain their legacy input-boundary estimate phase;
compare their final outputs directly rather than assuming intermediate clean
estimates have identical prediction phases. The viewer labels mixed updates
with the sampler that ran. The final is the authoritative returned latent.

[Sampler alternation grid recipe](../../benchmarks/plms-ddim-sampler-grid.json)
compares block sizes and DDIM eta at seeds 2–5. Coupled grids expose exchange
amount and period. The ordinary grid reference retains the selected schedule
and the sampler in the main sampler dropdown.

### Grid controls

Open **Compare bends → Parameter grid**, choose two parameter axes and
comma-separated values, then **Queue grid**. Defaults use generation seeds
2, 3, 4 and 5. Each cell is a single ordinary task; an optional ordinary reference
is added for each seed. The grid does not multiply by **Also run** selections.
All cells are checked before queueing; limits are five values per axis, eight
seeds and 64 tasks. Use one resolved prompt and classic txt2img.

Final results appear under each seed and in normal task history. Click a finished
cell to restore its image settings and bend recipe. **Export recipe** saves the
full request, axes and seeds; **Load recipe** restores it without queueing.
Completed grid panels are session-local; export recipes before refreshing.
Example recipes: [Euler noise memory](../../benchmarks/euler-noise-memory-grid.json)
and [PLMS history](../../benchmarks/plms-history-grid.json), both 512×512 with the
four agreed seeds. These files use the bend grid's Load recipe button, not the
separate benchmark importer.

New examples: [PLMS seed oscillation](../../benchmarks/plms-seed-mixing-grid.json)
and [DDIM noise](../../benchmarks/ddim-noise-grid.json). Exact-step grids keep the
exact schedule in their neutral reference cells, so comparisons do not silently
change iteration count. The grid also supports eta, blend bounds, period and
second seed as appropriate axes.

## Implementation and verification

`sampler_adapters.py` installs pass-through sdkit dispatchers once and selects
the active adapter using a per-render ContextVar. Sampler objects, history and
noise generators are local to that render. DDIM/PLMS and UniPC retain upstream
integration code. The active Euler/Heun bend loop follows k-diffusion 0.0.12
equations, with an explicit noise generator and iteration window. Disabled or
zero-churn sampling uses the upstream loop.

Validation on 10 October 2026 used sdkit 2.1.1, stable-diffusion-sdkit 2.1.5,
k-diffusion 0.0.12, SD1.4, Titan RTX, high-VRAM mode, 256×256, seed 2 and CFG 7.5:

- 49 Python tests passed, including native solver CPU tests, unchanged final
  tensors/model-call counts, schedule resolution, batch limits, routing cleanup,
  thread isolation, private noise RNG, finite-state recovery and task-response metadata.
- Nine 20-step GPU comparisons (DDIM, PLMS, five UniPC presets, Euler, Heun)
  passed exact final RGB equality against ordinary sdkit generation. Each
  captured clean estimates at 2, 10 and 19 and the raw state at 2.
- DDIM and PLMS at requested 30 passed final RGB equality, reported 31 updates,
  and retained both clean/raw observations at 10 and 30.
- Fresh and held noise for both Euler and Heun repeated exactly, changed the
  ordinary output, and passed the zero-churn equality check. Experiment settings:
  churn 20, noise 1.5, iterations 5–15, every 2, noise seed 123.
- Browser fixture checks passed request construction, restore, sampler gating,
  native schedule output counts and predictor/final labels. Full application
  HTTP orchestration and the V100/low-VRAM matrix were not rerun for this change.

For new GPU checks, `scripts/validate_sampler_adapters.py` now uses Titan-only
512×512 samples at seeds 2–5. `scripts/validate_circuit_bends.py` covers the new
memory/fade/history variants. Both require `--model <local checkpoint> --output
<new evidence directory> --thread-check <fresh check.json>`; see
[`sample_render_policy.py`](../../scripts/sample_render_policy.py) for the check
record. Verify active Codex chats and Titan processes before creating that record.
Windows desktop-only graphics connections must be individually verified if
listed as desktop exceptions. A separately inspected idle validation server can
be recorded under `idle_servers` with its PID, exact process name and `/ping`
status URL; the runner requires `Online` again before every image. This does
not override another chat's reservation. On WDDM, verified desktop graphics
activity can be recorded with `desktop_utilization_verified`; utilization alone
cannot distinguish desktop composition from inference. Unknown processes and
unreachable or busy servers still stop the run.
The original 256×256 evidence above is historical. Set `HF_HOME` to
an existing model cache if needed; the script uses offline model loading, saves
PNG/JSON evidence, and leaves server and GPU settings alone. CPU tests run with
`python -m unittest discover -s tests -p "test_trajectory*.py"`; native solver
tests skip when the classic runtime is absent. `node tests/test_circuit_bend_ui.js`
checks control validation and output counts.

Adaptive samplers, Stability DPM Solver, img2img capture, other backends, PLMS
spatial history mixing and UniPC correction gain remain later work. The
[next-pass plan and validation record](CIRCUIT_BEND_NEXT_PASS.md) covers the
memory/envelope/grid/history additions.

## Seed mixing and DDIM validation — 10 October 2026

- 67 Python tests passed with the installed classic runtime. New checks cover
  all exact timetable sizes 1–1000, native equivalence at safe counts, two-seed
  batching, zero-strength/no-extra-call behavior, full-donor endpoints, repeated
  trajectories, independent interlacing and thread-local DDIM overrides.
- JavaScript validation/grid/recipe tests and the optional linkedom form test
  passed, including new controls, sampler gating, saved-setting restore and
  changing grid axes. Run the form test with `node tests/test_circuit_bend_dom.js`
  with linkedom available through `NODE_PATH`; it skips if that optional test
  dependency is absent. The step-player/GIF tests also passed.
- Titan RTX / SD1.4 / CFG 10 / 512×512 / seeds 2–5 / exact 16 steps: **36
  sequences and 576 exported frames**. Nine seed-2 variants were each rerendered
  and every exported frame matched exactly. Second seed was 5; DDIM noise seed
  was 123. All renders completed with finite values and no sampling warnings.
- Independent mixing matched **76 exported A/B reference endpoints exactly**.
  With both seeds equal to 5, all PLMS variants matched the reference at every
  exported step. All saved frame dimensions and pixel hashes were verified.
- History oscillation at strength 0.35 produced soft smearing and silhouettes.
  Fresh DDIM noise at eta 0.8 produced readable variants. Held noise at eta 0.8
  produced saturated static-like imagery; this finite glitch remains available.
- Evidence and player: `C:/LocalRepos/flexdiff-runtime/evidence/seed-bends-ddim-20261010/index.html`.
  Recipes, checks and hashes are in `results.json` and `verification.json`.
  Reproduce using `scripts/render_seed_bends.py` with the model/output/thread-check
  arguments described above. `--resume` verifies saved hashes and recipes before
  retaining completed sequences after a GPU guard pause.

The other project chat was idle and the resident Titan server was Online before
each launch. The guard stopped one run when Windows replaced ShellHost with a
new PID; that desktop process was inspected before resuming. Clocks and power
were unchanged. The rendering processes exited and released the Titan. Existing
resident servers were left running; restart the backend and refresh its UI to
load these source changes. Full application HTTP orchestration and other GPU
or backend combinations were not rerun in this pass.

## Sampler-type interlacing validation — 10 October 2026

- 72 Python tests passed. A manual native-update reference verifies that PLMS
  and DDIM receive each other's updated latent in alternation. Keeping one
  sampler for all steps matches native final tensors and model-call counts.
  Coupling 0 preserves separate native results, 0.5 joins branch tensors exactly,
  and 1 swaps them simultaneously. Tests also cover warm-up policies, timestep
  alignment, noise repeatability, finite-state recovery and unsupported models.
- JavaScript validation/grid tests and the optional form test passed for the
  new toggle, memory policies, coupling controls, grid axes and recipe restore.
  Player tests cover the per-update sampler label as well as playback/GIF timing.
- Titan RTX / SD1.4 / 512×512 / CFG 10 / exact 16 steps / seeds 2–5: **28
  sequences, 448 frames**. Eight ordinary reference sequences and 20 mixed
  sequences compare shared/stale/reset alternation, constant 50% coupling and
  oscillating coupling. All seven seed-2 variants repeated exactly at every
  exported step. All frame dimensions, hashes and mixed sampler labels passed
  independent verification. No numerical-stop warnings occurred.
- Every mixed sequence uses one generation seed and the same initial noise
  for both solvers. Eta was 0 to isolate sampler interaction: recipe noise seed
  123 is recorded but unused. There is no second/donor seed in this experiment.
- Shared-prediction alternation produced particularly strong structural changes
  in this batch. Stale-history steps showed alternating sharpness and shape;
  coupling produced different intermediate/final compositions. These finite
  artistic deviations are preserved, without assuming one history policy is
  uniformly smoother or better than another.

Evidence/player: `C:/LocalRepos/flexdiff-runtime/evidence/sampler-interlace-20261010/index.html`.
`results.json` preserves all recipes and per-step metadata; `verification.json`
records the independent checks. Reproduce with `scripts/render_seed_bends.py
--experiment sampler` and the usual model/output/thread-check arguments. The
other Codex chat was idle; GPU processes and the resident server's Online status
were checked. GPU clocks/power settings were unchanged. Existing resident
servers were not restarted; restart the backend and refresh its UI to load the
new source. No full application HTTP or non-Titan matrix was run in this pass.
