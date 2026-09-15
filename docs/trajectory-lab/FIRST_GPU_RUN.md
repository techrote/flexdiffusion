# First GTX 1650 Super GPU run

This is the first point where Trajectory Lab needs real hardware. The code/CI gates are passing; this run validates that checkpoint capture is numerically non-invasive and bounded under an actual SD1.x render.

## 1. Prepare a normal Easy Diffusion install

Use an existing working Easy Diffusion installation if possible. Confirm it can generate an SD1.x image normally, then close Easy Diffusion completely.

## 2. Clone the FlexDiffusion test branch

Example PowerShell:

```powershell
git clone https://github.com/techrote/flexdiffusion.git C:\flexdiffusion
cd C:\flexdiffusion
git checkout trajectory-artifacts
```

If the repository is already cloned:

```powershell
cd C:\flexdiffusion
git fetch origin
git checkout trajectory-artifacts
git pull --ff-only origin trajectory-artifacts
```

## 3. Link the installed Easy Diffusion UI/server to this checkout

The helper is reversible. Replace the install path with the real existing Easy Diffusion directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_flexdiffusion_dev.ps1 -EasyDiffusionPath "C:\EasyDiffusion"
```

It:

- refuses to run while something is listening on port 9000;
- backs up the two Easy Diffusion startup scripts once;
- disables only the startup copy operations that would overwrite the development UI/server;
- replaces the installed `ui` directory with a junction to this repository's `ui` directory;
- leaves the installed models/environment intact.

To undo it later:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_flexdiffusion_dev.ps1 -EasyDiffusionPath "C:\EasyDiffusion" -Restore
```

## 4. Start Easy Diffusion and select the classic engine

Start Easy Diffusion normally from its installed launcher.

In Settings:

1. set **Engine to use** to **v2.0** (`ed_classic`);
2. save;
3. restart Easy Diffusion;
4. confirm a normal SD1.4/SD1.x image still renders.

Trajectory Lab deliberately rejects other backends at this stage.

## 5. Run a short sanity pair

From the FlexDiffusion checkout, replace the model value with the exact SD1.x model filename/path as Easy Diffusion knows it:

```powershell
py .\scripts\trajectory_smoke.py --model "YOUR-SD1X-MODEL.safetensors" --trajectory-root "C:\FlexDiffusionTrajectories" --steps 12 --modes baseline,latent
```

Expected result:

- baseline and latent-capture final PNG hashes are identical;
- one trajectory manifest is created for the latent run;
- selected `.safetensors` latent files exist and match manifest SHA-256 values;
- script exits with `PASS`.

If this fails, stop here and preserve the complete console/server log.

## 6. Run the full capture matrix

After the sanity pair passes:

```powershell
py .\scripts\trajectory_smoke.py --model "YOUR-SD1X-MODEL.safetensors" --trajectory-root "C:\FlexDiffusionTrajectories"
```

The default matrix runs:

- baseline;
- sparse latent-only capture;
- sparse preview-only capture;
- sparse hybrid capture;
- dense every-callback hybrid capture with queue depth 1;
- deliberate tiny storage-budget failure.

The budget case is expected to report capture/storage errors in its manifest while preserving the final render. The harness treats that as success only when the failure is explicit and no partial checkpoint artefacts remain.

## 7. Hardware observations

During the dense run, record or screenshot:

- peak GPU VRAM;
- peak host RAM;
- approximate GPU utilization;
- any obvious long stalls.

The automated harness records render wall times and trajectory byte counts but does not yet collect NVIDIA telemetry itself.

## 8. What not to infer yet

Passing this gate proves capture/persistence behavior. It does **not** prove that a saved latent is restartable.

The current k-diffusion callback is an internal solver boundary before the current integration update. Exact continuation is a separate sampler-specific research gate documented in `SAMPLER_RESUME_RESEARCH.md`.
