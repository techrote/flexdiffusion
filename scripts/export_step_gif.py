"""Export an existing viewer sequence, using its indexed pixels and JS timeline.

Requires Pillow and Node. No model loading or GPU work. The JSON sidecar records
the request, observation phases, source RGB hashes and exact playback timeline.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image


def export(directory, name, output, fps=8, continuous=False):
    if Path(name).name != name:
        raise ValueError('Sequence name must not contain a path')
    if not 1 <= fps <= 20:
        raise ValueError('Speed must be between 1 and 20')
    rows = json.loads((directory / 'results.json').read_text(encoding='utf-8'))['results']
    row = next(r for r in rows if r['name'] == name)
    source = (directory / f'{name}-gif-data.js').read_text(encoding='utf-8')
    prefix = 'window.stepGifData='
    if not source.startswith(prefix) or not source.rstrip().endswith(';'):
        raise ValueError('Expected viewer GIF data, not executable JavaScript')
    data = json.loads(source[len(prefix):].rstrip()[:-1])
    size = (data['width'], data['height'])
    frames = []
    for step, raw in enumerate(data['frames'], 1):
        with Image.open(directory / f'{name}-step{step}.png') as png:
            digest = hashlib.sha256(png.convert('RGB').tobytes()).hexdigest()
            if png.size != size or digest != row['step_sha256'][step - 1]:
                raise ValueError(f'Source frame {step} does not match the render manifest')
        frame = Image.frombytes('P', size, base64.b64decode(raw['pixels'], validate=True))
        frame.putpalette(base64.b64decode(raw['palette'], validate=True))
        frames.append(frame)
    if len(frames) != len(row['step_sha256']) or len(frames) < 2:
        raise ValueError('Incomplete sequence')
    # One timing implementation for browser playback, browser GIFs and this CLI.
    player = Path(__file__).with_name('step_viewer.js').resolve()
    program = 'process.stdout.write(JSON.stringify(require(process.argv[1]).timeline(+process.argv[2],+process.argv[3],process.argv[4]==="true")))'
    timeline = json.loads(subprocess.check_output(
        ['node', '-e', program, str(player), str(len(frames)), str(fps), str(continuous).lower()], text=True))
    ordered = [frames[item['step'] - 1] for item in timeline]
    durations = [item['delay'] * 10 for item in timeline]
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered[0].save(output, save_all=True, append_images=ordered[1:],
                    duration=durations, disposal=1, optimize=False,
                    **({'loop': 0} if continuous else {}))
    # Pillow may combine identical neighbouring frames. Compare the full timed
    # RGB stream, allowing only that lossless duration aggregation.
    expected = [(image.convert('RGB').tobytes(), duration) for image, duration in zip(ordered, durations)]
    with Image.open(output) as gif:
        if gif.size != size or gif.info.get('loop') != (0 if continuous else None):
            raise AssertionError('GIF dimensions or looping changed')
        cursor = 0
        remaining = expected[0][1]
        for index in range(gif.n_frames):
            gif.seek(index)
            pixels = gif.convert('RGB').tobytes()
            delay = gif.info['duration']
            while delay:
                if cursor >= len(expected) or pixels != expected[cursor][0]:
                    raise AssertionError('GIF changed indexed frame pixels or ordering')
                elapsed = min(delay, remaining)
                delay -= elapsed
                remaining -= elapsed
                if remaining == 0:
                    cursor += 1
                    if cursor < len(expected):
                        remaining = expected[cursor][1]
        if cursor != len(expected):
            raise AssertionError('GIF duration changed')
        encoded_frames = gif.n_frames
    record = dict(schema='flexdiffusion-step-gif/v1', source_evidence=directory.name,
                  render=row, playback=dict(speed_slider=fps, continuous=continuous,
                  endpoint_hold_ms=1000, duration_ms=sum(durations), timeline=timeline),
                  export=dict(file=output.name, bytes=output.stat().st_size,
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  frames=encoded_frames, width=size[0], height=size[1],
                  palette='Existing viewer 256-colour per-frame palette',
                  encoder='Pillow LZW; decoded RGB and timing verified against viewer data'))
    output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f'{output.name}: {encoded_frames} frames, {sum(durations)} ms, {output.stat().st_size:,} bytes; verified')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('name')
    parser.add_argument('output', type=Path)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--continuous', action='store_true')
    args = parser.parse_args()
    export(args.directory, args.name, args.output, args.fps, args.continuous)
