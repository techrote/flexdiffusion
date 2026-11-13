"""Build a file:// step player with local GIF export data from a render manifest."""
import argparse
import base64
import html
import json
from pathlib import Path
import shutil
from PIL import Image


def build(directory):
    rows = json.loads((directory / 'results.json').read_text(encoding='utf-8'))['results']
    counts = {max(o['step'] for o in row['observations']) for row in rows}
    if len(counts) != 1 or min(counts) < 2:
        raise ValueError('Viewer requires a shared sequence length of at least two steps')
    count = counts.pop()
    sizes = {(row['request']['width'], row['request']['height']) for row in rows}
    if len(sizes) != 1:
        raise ValueError('Viewer requires shared frame dimensions')
    width, height = sizes.pop()
    seeds = ', '.join(str(s) for s in sorted({r['request']['seed'] for r in rows}))
    guidance = ', '.join(str(s) for s in sorted({r['request']['guidance_scale'] for r in rows}))
    samplers = ' / '.join(sorted({r['request'].get('sampler_name', 'plms').upper() for r in rows}))
    details = html.escape(f'SD 1.4 · Titan RTX · {width}×{height} · seeds {seeds} · CFG {guidance}')
    prompt = html.escape(rows[0]['request']['prompt'])
    overviews = ' · '.join(f'<a href="{path.name}">{html.escape(path.stem.replace("overview-", "").replace("-", " "))} overview</a>'
                          for path in sorted(directory.glob('overview-*.jpg')))
    default_name = next((r['name'] for r in rows if (r.get('circuit_bend') or {}).get('mode') == 'sampler_mix'), rows[0]['name'])
    options = ''.join('<option' + (' selected' if row['name'] == default_name else '') + '>' + html.escape(row['name']) + '</option>' for row in rows)
    sequence_meta = json.dumps({r['name']: {'bend': r.get('circuit_bend'),
        'steps': {str(o['step']): o.get('metadata', {}) for o in r['observations']}} for r in rows}).replace('<', '\\u003c')
    for row in rows:
        name = row['name']
        if Path(name).name != name:
            raise ValueError('Invalid sequence name')
        target = directory / f'{name}-gif-data.js'
        if target.exists():
            continue
        frames = []
        for step in range(1, count + 1):
            with Image.open(directory / f'{name}-step{step}.png') as image:
                if image.size != (width, height):
                    raise ValueError(f'Expected {width}x{height} frames')
                indexed = image.convert('RGB').quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                palette = bytes(indexed.getpalette()[:768]).ljust(768, b'\0')
                frames.append(dict(pixels=base64.b64encode(indexed.tobytes()).decode(), palette=base64.b64encode(palette).decode()))
        target.write_text('window.stepGifData=' + json.dumps(dict(name=name, width=width, height=height, frames=frames), separators=(',', ':')) + ';', encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('step_viewer.js'), directory / 'step_viewer.js')
    (directory / 'index.html').write_text('''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAMPLERS steps 1–COUNT</title><style>
body{background:#191b20;color:#eee;font:16px system-ui;margin:24px;max-width:1050px}h1{margin-bottom:8px}p{line-height:1.5;color:#bcc5d5}
.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:16px 0}button,select{background:#303746;color:#fff;border:1px solid #56647a;border-radius:8px;padding:10px;font:inherit}button{cursor:pointer}button:disabled{opacity:.5}input{accent-color:#80b9ff}#step{width:min(65vw,420px)}#frame{max-width:100%;width:512px;display:block;border-radius:8px}a{color:#99caff}#status{min-height:24px}label{display:inline-flex;gap:8px;align-items:center}
</style><h1>SAMPLERS steps 1–COUNT</h1><p>DETAILS<br>Steps 1–BEFORE are clean-image estimates; step COUNT is the final output.</p><details><summary>Character prompt</summary><p>PROMPT</p></details>
<label for="recipe">Sequence</label> <select id="recipe">OPTIONS</select>
<p id="sequenceSettings"></p>
<div class="controls"><button id="play">Play</button><label><input id="continuous" type="checkbox">Continuous</label><label for="speed">Speed <input id="speed" type="range" min="1" max="20" value="8"></label><output id="speedLabel"></output><button id="export">Export to GIF</button></div>
<p>Forward → pause 1 second → backward → pause 1 second. Continuous repeats the cycle.<br>GIF export uses the complete sequence, current speed and continuous setting. GIF timing is rounded to 10 ms; the player uses that same timing.</p>
<div class="controls"><label for="step">Step</label><input id="step" type="range" min="1" max="COUNT" value="1"><output id="label"></output></div>
<p id="stepDetails"></p><script type="application/json" id="sequenceMetadata">SEQUENCE_META</script>
<img id="frame" alt="Selected render step"><p id="status" role="status" aria-live="polite"></p>
<p><a id="sheet">All COUNT steps contact sheet</a> · <a href="results.json">Recipes and metadata</a> · <a href="preflight.json">Render setup</a></p><p>OVERVIEWS</p><script src="step_viewer.js"></script></html>'''.replace('COUNT', str(count)).replace('BEFORE', str(count - 1)).replace('SAMPLERS', html.escape(samplers)).replace('DETAILS', details).replace('PROMPT', prompt).replace('OPTIONS', options).replace('OVERVIEWS', overviews).replace('SEQUENCE_META', sequence_meta), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    build(parser.parse_args().directory.resolve())
