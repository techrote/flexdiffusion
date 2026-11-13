"""Save noise-memory/envelope and PLMS-freeze samples on an available Titan.

All samples are 512x512 at generation seeds 2, 3, 4, 5. Requires --thread-check
from a fresh Codex app check. Refuses to start if the Titan has other compute
processes; checks again before every render. Does not alter GPU settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from sample_render_policy import SAMPLE_SEEDS, SAMPLE_SIZE, assert_no_other_compute, prepare_titan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thread-check", required=True, type=Path)
    parser.add_argument("--samplers", nargs="+", choices=["euler", "heun", "plms"], default=["euler", "heun", "plms"])
    parser.add_argument("--all-steps", action="store_true", help="Export steps 1-19 and the authoritative final as step 20")
    args = parser.parse_args()
    preflight = prepare_titan(args.thread_check)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "temp").mkdir()
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    tempfile.tempdir = str(args.output / "temp")
    ui = Path(__file__).resolve().parents[1] / "ui"
    sys.path.insert(0, str(ui))
    os.environ.update(SD_UI_PATH=str(ui), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    from sdkit import Context
    from sdkit.models import load_model
    from sdkit.generate import generate_images as native_generate
    from sdkit.utils import base64_str_to_img
    from easydiffusion import app
    from easydiffusion.backends import sdkit_common as backend
    from PIL import Image, ImageDraw

    context = Context()
    context.device = "cuda:0"
    context.test_diffusers = False
    context.vram_usage_level = "high"
    context.model_paths["stable-diffusion"] = str(args.model.resolve())
    assert_no_other_compute()
    load_model(context, "stable-diffusion")
    backend.set_options(context, output_format="png", output_quality=100, output_lossless=True,
                        stream_image_progress=False)
    results, thumbnails = [], []
    def digest(image): return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    def render(req, bend=None, native=False):
        assert_no_other_compute()
        if native: return native_generate(context, **req)[0]
        return backend.generate_images(context, circuit_bend=bend, output_after_step=1 if args.all_steps else 5,
            intermediate_representation="denoised" if args.all_steps else "sparse:5,10,15,20|endpoints=0", **req)[0]
    def export_steps(name, image, observations):
        if args.all_steps:
            assert [o["step"] for o in observations] == list(range(1, 20)), "Incomplete step capture"
        for observation in observations:
            base64_str_to_img(observation["images"][0]).save(args.output / f"{name}-step{observation['step']}.png")
        if args.all_steps:
            image.save(args.output / f"{name}-step20.png")
        metadata = [{k: v for k, v in observation.items() if k != "images"} for observation in observations]
        if args.all_steps:
            metadata.append(dict(step=20, representation="final", metadata={"observation_phase": "final_return"}))
        return metadata
    def save(name, image, req, bend, extra):
        image.save(args.output / f"{name}.png")
        results.append(dict(name=name, request=req, circuit_bend=bend, sha256=digest(image), **extra))
        (args.output / "results.json").write_text(json.dumps({"preflight": preflight, "results": results}, indent=2), encoding="utf-8")
        thumbnails.append((name, image.copy()))
        print(json.dumps(results[-1]), flush=True)

    for seed in SAMPLE_SEEDS:
        for sampler in args.samplers:
            req = dict(prompt="cyberpunk dinosaur mercenary", negative_prompt="", seed=seed,
                width=SAMPLE_SIZE, height=SAMPLE_SIZE, num_outputs=1, num_inference_steps=20,
                guidance_scale=7.5, init_image=None, control_image=None, sampler_name=sampler)
            baseline = render(req, native=True)
            neutral = dict(enabled=True, mode="plms_history", history_strength=0) if sampler == "plms" else dict(enabled=True, churn=0)
            same = render(req, neutral)
            assert digest(baseline) == digest(same), (sampler, seed, "neutral changed output")
            name = f"{sampler}-seed{seed}-ordinary"
            extra = {"neutral_matches_native": True}
            if args.all_steps:
                assert not context.output_step_warnings, context.output_step_warnings
                extra["observations"] = export_steps(name, same, context.output_step_results)
            save(name, baseline, req, None, extra)
            common = dict(enabled=True, window_unit="percent", start=25, end=75, fade_in=20, fade_out=25)
            variants = [("freeze-half", dict(common, mode="plms_history", history_strength=.5)),
                        ("freeze-full", dict(common, mode="plms_history", history_strength=1))] if sampler == "plms" else [
                        (f"memory-{memory:g}", dict(common, churn=20, noise=1.25, memory=memory, seed=123)) for memory in (0, .5, 1)]
            for label, bend in variants:
                image = render(req, bend)
                observations = list(context.output_step_results)
                warnings = list(context.output_step_warnings)
                repeated = render(req, bend)
                assert digest(image) == digest(repeated), (sampler, seed, label, "not repeatable")
                assert not warnings and not context.output_step_warnings, (sampler, seed, warnings)
                name = f"{sampler}-seed{seed}-{label}"
                metadata = export_steps(name, image, observations)
                save(name, image, req, bend, {"repeat_exact": True, "changes_output": digest(image) != digest(baseline),
                    "observations": metadata})
    # Samples stay 512x512; contact sheets use thumbnails only for inspection.
    for sampler in args.samplers:
        chosen = [(name, image) for name, image in thumbnails if name.startswith(sampler + "-")]
        columns = 3 if sampler == "plms" else 4
        sheet = Image.new("RGB", (columns * 256, 4 * 286), "#202020")
        draw = ImageDraw.Draw(sheet)
        for index, (name, image) in enumerate(chosen):
            x, y = index % columns * 256, index // columns * 286
            sheet.paste(image.resize((256, 256)), (x, y))
            draw.text((x + 5, y + 262), name, fill="white")
        sheet.save(args.output / f"{sampler}-comparison.png")


if __name__ == "__main__":
    main()
