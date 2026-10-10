"""Validate/export seed bends or PLMS/DDIM sampler interlacing on an idle Titan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from sample_render_policy import SAMPLE_SEEDS, SAMPLE_SIZE, prepare_titan, assert_no_other_compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--thread-check', required=True, type=Path)
    parser.add_argument('--resume', action='store_true', help='Verify and retain completed sequences after a guard pause')
    parser.add_argument('--experiment', choices=['seed', 'sampler'], default='seed')
    args = parser.parse_args()
    preflight = prepare_titan(args.thread_check)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / 'temp').mkdir(exist_ok=args.resume)
    tempfile.tempdir = str((args.output / 'temp').resolve())
    preflight_file = f"preflight-resume-{int(preflight['checked_at'])}.json" if args.resume else 'preflight.json'
    (args.output / preflight_file).write_text(json.dumps(preflight, indent=2), encoding='utf-8')
    ui = Path(__file__).resolve().parents[1] / 'ui'
    sys.path.insert(0, str(ui))
    os.environ.update(SD_UI_PATH=str(ui), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from sdkit import Context
    from sdkit.models import load_model
    from sdkit.utils import base64_str_to_img
    from easydiffusion import app
    from easydiffusion.backends import sdkit_common as backend
    from easydiffusion.circuit_bend import CircuitBendOptions
    from render_plms_character_grid import CYBORG_PROMPT, CYBORG_NEGATIVE
    from PIL import Image, ImageDraw

    context = Context()
    context.device = 'cuda:0'
    context.test_diffusers = False
    context.vram_usage_level = 'high'
    context.model_paths['stable-diffusion'] = str(args.model.resolve())
    assert_no_other_compute()
    load_model(context, 'stable-diffusion')
    backend.set_options(context, output_format='png', output_quality=100, output_lossless=True,
                        stream_image_progress=False)
    common = dict(enabled=True, schedule='exact')
    plms = dict(common, mode='plms_history', history_source='seed', donor_seed=5)
    variants = [
        ('plms-reference', dict(plms, history_strength=0)),
        ('plms-history-oscillate', dict(plms, blend_curve='oscillate', blend_period=8, history_strength=.35)),
        ('plms-step-progressive', dict(plms, blend_target='trajectory', blend_curve='progressive')),
        ('plms-step-interlace', dict(plms, blend_target='trajectory', blend_curve='interlace', blend_period=2)),
        ('ddim-reference', dict(common, mode='ddim', eta=0)),
        ('ddim-fresh', dict(common, mode='ddim', eta=.8, seed=123, memory=0)),
        ('ddim-held', dict(common, mode='ddim', eta=.8, seed=123, memory=1)),
        ('plms-independent-interlace', dict(plms, blend_target='independent', blend_curve='interlace', blend_period=1)),
        ('plms-independent-oscillate', dict(plms, blend_target='independent', blend_curve='oscillate', blend_period=8)),
    ]
    if args.experiment == 'sampler':
        hybrid = dict(common, mode='sampler_mix', eta=0, seed=123)
        variants = [
            ('plms-reference', dict(common, mode='plms_history', history_strength=0)),
            ('ddim-reference', dict(common, mode='ddim', eta=0)),
            ('plms-alternate-shared', dict(hybrid, mix_history='shared')),
            ('plms-alternate-stale', dict(hybrid, mix_history='stale')),
            ('plms-alternate-reset', dict(hybrid, mix_history='reset')),
            ('plms-coupled-half', dict(hybrid, mix_method='coupled', mix_exchange=.5, mix_output='plms')),
            ('plms-coupled-oscillate', dict(hybrid, mix_method='coupled', mix_exchange=.5,
                mix_output='plms', mix_curve='oscillate', mix_period=8)),
        ]
    results, thumbnails = [], []
    def digest(image): return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
    if args.resume:
        results = json.loads((args.output / 'results.json').read_text(encoding='utf-8'))['results']
        for row in results:
            for step, expected in enumerate(row['step_sha256'], 1):
                with Image.open(args.output / f"{row['name']}-step{step}.png") as saved:
                    assert digest(saved) == expected, 'Saved frame changed before resume'
            with Image.open(args.output / f"{row['name']}.png") as saved:
                assert digest(saved) == row['sha256']
                thumbnails.append((row['name'], saved.copy()))
    def render(req, bend):
        assert_no_other_compute()
        image = backend.generate_images(context, circuit_bend=bend, output_after_step=1,
            intermediate_representation='denoised', **req)[0]
        assert not context.output_step_warnings, context.output_step_warnings
        assert context.sampling_metadata['actual_steps'] == 16
        assert [o['step'] for o in context.output_step_results] == list(range(1, 16))
        frames = [base64_str_to_img(o['images'][0]).convert('RGB') for o in context.output_step_results] + [image]
        observations = [{k: v for k, v in o.items() if k != 'images'} for o in context.output_step_results]
        final_meta = {'observation_phase': 'final_return'}
        if bend['mode'] == 'sampler_mix':
            final_meta.update(sampler_iteration=16, active_sampler='plms+ddim' if bend['mix_method'] == 'coupled'
                else ('ddim' if bend['mix_first'] == 'plms' else 'plms') if (15 // bend['mix_block']) % 2 else bend['mix_first'])
        observations.append(dict(step=16, representation='final', metadata=final_meta))
        return image, frames, observations, dict(context.sampling_metadata)
    for seed in SAMPLE_SEEDS:
        for label, options in variants:
            name = f'{label}-seed{seed}'
            bend = CircuitBendOptions(**options).dict()
            req = dict(prompt=CYBORG_PROMPT, negative_prompt=CYBORG_NEGATIVE, seed=seed,
                width=SAMPLE_SIZE, height=SAMPLE_SIZE, num_outputs=1, num_inference_steps=16,
                guidance_scale=10, init_image=None, control_image=None, sampler_name=label.split('-')[0])
            existing = next((row for row in results if row['name'] == name), None)
            if existing:
                assert existing['request'] == req and CircuitBendOptions(**existing['circuit_bend']).dict() == bend, 'Recipe changed before resume'
                continue
            image, frames, observations, metadata = render(req, bend)
            hashes = [digest(frame) for frame in frames]
            repeated = None
            if seed == SAMPLE_SEEDS[0]:
                _, repeat_frames, _, _ = render(req, bend)
                repeated = hashes == [digest(frame) for frame in repeat_frames]
                assert repeated, f'{name} not repeatable'
            image.save(args.output / f'{name}.png')
            sheet = Image.new('RGB', (4 * 256, 4 * 276), '#202020')
            draw = ImageDraw.Draw(sheet)
            for step, frame in enumerate(frames, 1):
                frame.save(args.output / f'{name}-step{step}.png')
                x, y = (step - 1) % 4 * 256, (step - 1) // 4 * 276
                sheet.paste(frame.resize((256, 256)), (x, y))
                draw.text((x + 5, y + 258), f'Step {step}', fill='white')
            sheet.save(args.output / f'{name}-steps.jpg', quality=90)
            results.append(dict(name=name, request=req, circuit_bend=bend, sha256=digest(image),
                noise_seed=bend['seed'] if bend['mode'] in ('ddim', 'sampler_mix') and bend['eta'] else None,
                donor_seed=bend['donor_seed'] if bend['mode'] == 'plms_history' and bend['history_source'] == 'seed' and bend['history_strength'] else None,
                repeat_all_steps_exact=repeated, step_sha256=hashes, observations=observations, sampling_metadata=metadata))
            (args.output / 'results.json').write_text(json.dumps(dict(preflight=preflight, results=results), indent=2), encoding='utf-8')
            thumbnails.append((name, image.copy()))
            print(f'SAVED {len(results)}/{len(SAMPLE_SEEDS)*len(variants)} {name}', flush=True)
    ordering = {f'{label}-seed{seed}': i for i, (seed, (label, _)) in enumerate(
        (seed, variant) for seed in SAMPLE_SEEDS for variant in variants)}
    thumbnails.sort(key=lambda row: ordering[row[0]])
    results.sort(key=lambda row: ordering[row['name']])
    (args.output / 'results.json').write_text(json.dumps(dict(preflight=preflight, results=results), indent=2), encoding='utf-8')
    overview = Image.new('RGB', (len(variants)*224, len(SAMPLE_SEEDS)*254), '#202020')
    draw = ImageDraw.Draw(overview)
    for i, (name, image) in enumerate(thumbnails):
        x, y = i % len(variants)*224, i // len(variants)*254
        overview.paste(image.resize((224, 224)), (x, y))
        draw.text((x+3, y+226), name, fill='white')
    overview.save(args.output / 'overview-seed-bends.jpg', quality=92)
    print('Rendering complete; building viewer', flush=True)
    from build_step_viewer import build
    build(args.output)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
