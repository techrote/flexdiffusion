"""Compare native renders with intermediate capture and repeatable noise bends.

Uses an existing local checkpoint. Writes PNGs and JSON evidence to --output;
does not start/restart a server or change GPU settings.
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0"])
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE, choices=[SAMPLE_SIZE])
    parser.add_argument("--thread-check", type=Path, required=True)
    args = parser.parse_args()
    preflight = prepare_titan(args.thread_check)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "temp").mkdir()
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    tempfile.tempdir = str(args.output / "temp")
    ui = Path(__file__).resolve().parents[1] / "ui"
    sys.path.insert(0, str(ui))
    os.environ["SD_UI_PATH"] = str(ui)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from sdkit import Context
    from sdkit.models import load_model
    from sdkit.generate import generate_images as upstream_generate
    from sdkit.utils import base64_str_to_img
    from easydiffusion import app  # Match the application's normal import order.
    from easydiffusion.backends import sdkit_common as backend

    context = Context()
    context.device = args.device
    context.test_diffusers = False
    context.vram_usage_level = "high"
    context.model_paths["stable-diffusion"] = str(args.model.resolve())
    assert_no_other_compute()
    load_model(context, "stable-diffusion")
    backend.set_options(context, output_format="png", output_quality=100, output_lossless=True,
                        stream_image_progress=False)
    def native_generate(*args, **kwargs):
        assert_no_other_compute()
        return upstream_generate(*args, **kwargs)

    def captured_generate(*args, **kwargs):
        assert_no_other_compute()
        return backend.generate_images(*args, **kwargs)

    results = []
    samples = []

    def pixels(image):
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()

    def save(name, image):
        image.save(args.output / f"seed{current_seed}-{name}.png")
        return pixels(image)

    def record(row):
        row["seed"] = current_seed
        results.append(row)
        (args.output / "results.json").write_text(json.dumps({
            "model": str(args.model), "device": args.device, "size": args.size, "results": results,
        }, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    for current_seed in SAMPLE_SEEDS:
        base = dict(prompt="cyberpunk dinosaur mercenary", negative_prompt="", seed=current_seed,
                    width=args.size, height=args.size, num_outputs=1, num_inference_steps=20,
                    guidance_scale=7.5, init_image=None, control_image=None)
        for name in ["ddim", "plms", "unipc_tu", "unipc_snr", "unipc_tq", "unipc_tu_2", "unipc_snr_2", "euler", "heun"]:
            req = dict(base, sampler_name=name)
            expected = native_generate(context, **req)[0]
            captured = captured_generate(context, output_after_step=2,
                        intermediate_representation="sparse:2,10,19-20|endpoints=1", **req)[0]
            observations = context.output_step_results
            row = {"sampler": name, "same_final_pixels": pixels(expected) == pixels(captured),
                   "metadata": context.sampling_metadata, "observations": []}
            row["final_hash"] = save(f"{name}-final", captured)
            for observation in observations:
                image = base64_str_to_img(observation["images"][0])
                filename = f"{name}-{observation['step']}-{observation['representation']}"
                row["observations"].append({"step": observation["step"], "representation": observation["representation"],
                                            "hash": save(filename, image), **observation["metadata"]})
                if observation["representation"] == "denoised":
                    samples.append((f"s{current_seed}-{filename}", image))
            assert row["same_final_pixels"], row
            assert [(o["step"], o["representation"]) for o in observations] == [
                (2, "denoised"), (2, "solver_state"), (10, "denoised"), (19, "denoised")]
            assert not context.output_step_warnings, context.output_step_warnings
            record(row)
        for name in ["ddim", "plms"]:
            req = dict(base, sampler_name=name, num_inference_steps=30)
            expected = native_generate(context, **req)[0]
            captured = captured_generate(context, output_after_step=10,
                        intermediate_representation="sparse:10,30|endpoints=1", **req)[0]
            assert pixels(expected) == pixels(captured)
            assert context.sampling_metadata["actual_steps"] == 31
            assert [(o["step"], o["representation"]) for o in context.output_step_results] == [
                (10, "denoised"), (10, "solver_state"), (30, "denoised"), (30, "solver_state")]
            record({"sampler": name, "requested_steps": 30, "actual_steps": 31, "same_final_pixels": True})
        for name in ["euler", "heun"]:
            req = dict(base, sampler_name=name)
            baseline = native_generate(context, **req)[0]
            zero = captured_generate(context, circuit_bend={"enabled": True, "churn": 0}, **req)[0]
            assert pixels(baseline) == pixels(zero)
            for pattern in ["fresh", "held"]:
                bend = dict(enabled=True, churn=20, noise=1.5, start=5, end=15, every=2, seed=123, pattern=pattern)
                bent = captured_generate(context, circuit_bend=bend, **req)[0]
                repeated = captured_generate(context, circuit_bend=bend, **req)[0]
                assert pixels(bent) == pixels(repeated)
                assert pixels(bent) != pixels(baseline)
                record({"sampler": name, "circuit_bend": bend, "same_repeat_pixels": True,
                        "zero_churn_matches_native": True, "changes_output": True,
                        "hash": save(f"{name}-bend-{pattern}", bent)})
                samples.append((f"s{current_seed}-{name}-bend-{pattern}", bent))
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (args.size * 4, (args.size + 30) * ((len(samples) + 3) // 4)), "#171717")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(samples):
        x, y = (index % 4) * args.size, (index // 4) * (args.size + 30)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + args.size + 7), name, fill="white")
    sheet.save(args.output / "contact-sheet.png")


if __name__ == "__main__":
    main()
