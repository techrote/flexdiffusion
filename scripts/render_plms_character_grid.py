"""Art sample batch: explicit 16-update PLMS schedule, all steps, CFG/seed grid.

Schedule overrides live only in this dedicated process, never in the UI server.
Native PLMS equations and coefficient construction are retained. No GPU tuning.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from sample_render_policy import prepare_titan, assert_no_other_compute

PROMPT = ('Anthropomorphic velociraptor soldier, unhelmeted green scaly dinosaur head, short snout, amber eyes, '
          'upright human-shaped body, long arms and legs. Holding a massive futuristic kinetic anti-materiel '
          'rifle in both hands, long barrel, box magazine, telescopic scope. Black tactical armor, full-body '
          'character portrait, neon city street, detailed science fiction painting.')
NEGATIVE = 'helmet, mask, human face, quadruped, tiny arms, extra limbs, extra fingers, blurry, text, watermark'
CYBORG_PROMPT = ('A single humanoid dinosaur cyborg mercenary standing upright, two long robotic legs, '
                'two mechanical arms, armored torso, green scaly reptilian head and toothed snout. '
                'Holding a futuristic kinetic anti-materiel rifle in both clawed hands, long barrel and scope. '
                'Full-length centered character design, entire head and feet visible, neon city, science fiction illustration.')
CYBORG_NEGATIVE = ('human face, human skin, man, woman, quadruped, multiple characters, duplicate bodies, '
                   'crowd, helmet, mask, cropped head, cropped feet, close-up, extra limbs, text, watermark')


def exact_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=False):
    import numpy as np
    if ddim_discr_method != 'uniform' or num_ddim_timesteps != 16 or num_ddpm_timesteps != 1000:
        raise ValueError('This experimental batch explicitly requires uniform 16 / 1000')
    return 1 + np.floor(np.arange(16) * (1000 / 16)).astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thread-check', type=Path, required=True)
    parser.add_argument('--preset', choices=['character','cyborg'], default='character')
    parser.add_argument('--height', type=int, choices=[512,1024], default=512)
    parser.add_argument('--guidance', nargs='+', type=int, choices=range(1,17), default=[9,10,11,12])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1,9)))
    parser.add_argument('--variants', nargs='+', choices=['ordinary','freeze-half','freeze-full'], default=['ordinary','freeze-half','freeze-full'])
    args = parser.parse_args()
    prompt, negative = (CYBORG_PROMPT,CYBORG_NEGATIVE) if args.preset=='cyborg' else (PROMPT,NEGATIVE)
    preflight = prepare_titan(args.thread_check)
    preflight['seeds'] = args.seeds
    preflight['guidance_scales'] = args.guidance
    preflight['size'] = [512,args.height]
    preflight['steps'] = 16
    preflight['schedule'] = {'name': 'experimental_exact_uniform_16', 'formula': '1 + floor(i * 1000 / 16), i=0..15'}
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        previous = json.loads((out/'results.json').read_text())['results']
        if any(r['request']['prompt']!=prompt or r['request']['negative_prompt']!=negative or r['request']['height']!=args.height for r in previous):
            raise ValueError('Existing batch has different prompt or dimensions; use a new directory')
    (out/'temp').mkdir(exist_ok=True); tempfile.tempdir = str(out/'temp')
    (out/'preflight.json').write_text(json.dumps(preflight, indent=2))
    ui = Path(__file__).resolve().parents[1] / 'ui'; sys.path.insert(0,str(ui))
    os.environ.update(SD_UI_PATH=str(ui), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from sdkit import Context
    from sdkit.models import load_model
    from sdkit.utils import base64_str_to_img
    from easydiffusion import app
    from easydiffusion.backends import sdkit_common as backend
    import ldm.modules.diffusionmodules.util as util
    import ldm.models.diffusion.plms as plms
    from PIL import Image, ImageDraw
    util.make_ddim_timesteps = plms.make_ddim_timesteps = exact_timesteps
    preflight['schedule']['ascending_timesteps'] = exact_timesteps('uniform',16,1000).tolist()
    (out/'preflight.json').write_text(json.dumps(preflight,indent=2))
    context=Context(); context.device='cuda:0'; context.test_diffusers=False; context.vram_usage_level='high'
    context.model_paths['stable-diffusion']=str(args.model.resolve())
    assert_no_other_compute(); load_model(context,'stable-diffusion')
    backend.set_options(context,output_format='png',output_quality=100,output_lossless=True,stream_image_progress=False)
    results_path=out/'results.json'
    results=json.loads(results_path.read_text())['results'] if results_path.exists() else []
    completed={r['name'] for r in results}
    variants=[(name,{'ordinary':None,'freeze-half':.5,'freeze-full':1}[name]) for name in args.variants]
    total=len(args.seeds)*len(args.guidance)*len(variants)
    thumb_width=round(256*512/args.height)
    for seed in args.seeds:
        for cfg in args.guidance:
            for label,strength in variants:
                name=f'plms-seed{seed}-cfg{cfg}-{label}'
                if name in completed: continue
                assert_no_other_compute()
                req=dict(prompt=prompt,negative_prompt=negative,seed=seed,width=512,height=args.height,
                         num_outputs=1,num_inference_steps=16,guidance_scale=cfg,
                         init_image=None,control_image=None,sampler_name='plms')
                bend=None if strength is None else dict(enabled=True,mode='plms_history',history_strength=strength,
                         window_unit='percent',start=25,end=75,fade_in=20,fade_out=25)
                final=backend.generate_images(context,circuit_bend=bend,output_after_step=1,
                         intermediate_representation='denoised',**req)[0]
                observations=list(context.output_step_results)
                assert not context.output_step_warnings, context.output_step_warnings
                assert context.sampling_metadata['actual_steps']==16
                assert [o['step'] for o in observations]==list(range(1,16))
                final.save(out/f'{name}.png')
                frames=[base64_str_to_img(o['images'][0]) for o in observations]+[final]
                hashes=[];sheet=Image.new('RGB',(4*thumb_width,1104),'#202020');draw=ImageDraw.Draw(sheet)
                for step,image in enumerate(frames,1):
                    image.save(out/f'{name}-step{step}.png')
                    hashes.append(hashlib.sha256(image.convert('RGB').tobytes()).hexdigest())
                    x=(step-1)%4*thumb_width;y=(step-1)//4*276
                    sheet.paste(image.resize((thumb_width,256)),(x,y));draw.text((x+5,y+258),f'Step {step}',fill='white')
                sheet.save(out/f'{name}-steps.jpg',quality=93)
                metadata=[{k:v for k,v in o.items() if k!='images'} for o in observations]
                metadata.append(dict(step=16,representation='final',metadata={'observation_phase':'final_return'}))
                results.append(dict(name=name,request=req,circuit_bend=bend,noise_seed=None,
                    sha256=hashes[-1],step_sha256=hashes,observations=metadata,sampling_metadata=context.sampling_metadata))
                temporary=out/'results.tmp';temporary.write_text(json.dumps(dict(preflight=preflight,results=results),indent=2));temporary.replace(results_path)
                print(f'SAVED {len(results)}/{total} {name}',flush=True)
    for label,_ in variants:
        sheet=Image.new('RGB',(len(args.guidance)*thumb_width,len(args.seeds)*276),'#202020');draw=ImageDraw.Draw(sheet)
        for row,seed in enumerate(args.seeds):
            for col,cfg in enumerate(args.guidance):
                name=f'plms-seed{seed}-cfg{cfg}-{label}';x=col*thumb_width;y=row*276
                with Image.open(out/f'{name}.png') as image:sheet.paste(image.resize((thumb_width,256)),(x,y))
                draw.text((x+5,y+258),f'Seed {seed} / CFG {cfg}',fill='white')
        sheet.save(out/f'overview-{label}.jpg',quality=94)
    print(f'COMPLETE {total} sequences / {total*16} steps',flush=True)


if __name__=='__main__': main()
