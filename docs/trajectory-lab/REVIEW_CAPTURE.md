# Review Capture

FlexDiffusion's review-capture workflow turns a visual experiment into a paired,
editable evidence bundle that can be zipped and returned later without losing the
exact generation context.

## User workflow

After one or more result tasks are visible, click **Review capture** next to
**Download images**. FlexDiffusion downloads two files with the same local-time
basename:

```text
2026-09-13_17-33-04.png
2026-09-13_17-33-04.md
```

The PNG captures all currently visible result-task cards. The Markdown sidecar is
intended to be opened immediately and edited by the reviewer. Its first section is
reserved for rating, keep/reject/investigate status, preferred sampler, comments,
notable artifacts/useful edge cases, and follow-up ideas.

A browser may ask once for permission to allow multiple downloads. Both files are
ordinary independent downloads so the Markdown remains easy to edit before the
whole feedback set is eventually zipped.

## Captured metadata

The sidecar records enough structured context to reconstruct and compare an
experiment without relying on the screenshot alone:

- capture time, local timezone and ISO timestamp;
- Easy Diffusion UI/backend/update-branch information;
- relevant app configuration including low-VRAM mode when present;
- browser/environment hints useful for hardware/UI diagnosis;
- current primary sampler and any **Also run** sampler selections;
- explicit seed and current trajectory controls;
- every visible result task's complete request body;
- prompt, negative prompt, model, sampler, dimensions, CFG, step count, VAE,
  LoRA/hypernetwork/control settings and other request parameters carried by Easy
  Diffusion;
- task status, batch counters and Easy Diffusion timing/result text;
- each visible output's seed, completed-step metadata and
  `denoised`/`solver_state` representation label when available.

The human-readable summary is followed by a fenced JSON block containing the
machine-readable capture. That JSON is authoritative for later automated analysis.

## Large embedded values

Image inputs and other very large strings can make an editable sidecar unusably
large. Review Capture therefore replaces data URLs and strings longer than 4096
characters with a provenance descriptor containing:

- original character length;
- SHA-256 when Web Crypto is available;
- a short prefix identifying the omitted value.

This preserves identity/comparison information without copying megabytes of base64
into every review note.

## Screenshot implementation

The first path clones all visible `.imageTaskContainer` result cards, expands their
result areas, removes transient controls, inlines current visual styles and image
sources, and renders the result through an SVG `foreignObject` into a PNG canvas.
No remote screenshot service or CDN dependency is used.

Browsers differ in `foreignObject` behaviour. If that path fails, FlexDiffusion
falls back to a deterministic review-sheet renderer that draws task metadata and
all visible output images directly onto a canvas. The sidecar records
`capture.screenshot_mode` and, for a fallback, the original screenshot error.

## Scope

Review Capture intentionally exports **all currently visible result tasks**. This
matches sampler-sweep experiments where each selected sampler is already a separate
Easy Diffusion task. Use **Clear All** before a new experiment when a capture should
contain only that test set.

The export is observational only: it does not modify sampler state, task requests,
trajectory artifacts, generated images, seeds, or model state.
