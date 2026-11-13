/* Standalone local-file player and dependency-free GIF writer. */
(function () {
    const frameDelay = fps => Math.max(1, Math.round(100 / fps)); // GIF centiseconds
    function timeline(count, fps, continuous = false) {
        const steps = Array.from({length: count}, (_, i) => i + 1);
        for (let i = count - 1; i >= 1; i--) steps.push(i);
        if (continuous) steps.pop(); // Hold step 1 once across the loop boundary.
        return steps.map((step, i) => ({step, delay: step === count || (continuous ? i === 0 : i === steps.length - 1) ? 100 : frameDelay(fps)}));
    }
    function gifHeader(width, height, continuous) {
        const bytes = [...Array.from('GIF89a', c => c.charCodeAt(0)), width & 255, width >> 8, height & 255, height >> 8, 0x70, 0, 0];
        if (continuous) bytes.push(0x21, 0xff, 11, ...Array.from('NETSCAPE2.0', c => c.charCodeAt(0)), 3, 1, 0, 0, 0);
        return Uint8Array.from(bytes);
    }
    function gifFrame(width, height, pixels, palette, delay) {
        // Clear every 250 literals keeps LZW codes at nine bits. Larger files,
        // bounded memory and no dependency/worker/CORS requirement on file://.
        const packed = []; let bits = 0, bitCount = 0;
        const code = value => { bits |= value << bitCount; bitCount += 9; while (bitCount >= 8) { packed.push(bits & 255); bits >>>= 8; bitCount -= 8; } };
        for (let i = 0; i < pixels.length; i++) { if (i % 250 === 0) code(256); code(pixels[i]); }
        code(257); if (bitCount) packed.push(bits & 255);
        const bytes = [0x21, 0xf9, 4, 4, delay & 255, delay >> 8, 0, 0, 0x2c, 0, 0, 0, 0, width & 255, width >> 8, height & 255, height >> 8, 0x87, ...palette, 8];
        for (let i = 0; i < packed.length; i += 255) { const block = packed.slice(i, i + 255); bytes.push(block.length, ...block); }
        bytes.push(0); return Uint8Array.from(bytes);
    }
    if (typeof module !== 'undefined') module.exports = {timeline, gifHeader, gifFrame};
    if (typeof document === 'undefined') return;
    const $ = id => document.getElementById(id);
    const recipe = $('recipe'), step = $('step'), speed = $('speed'), continuous = $('continuous');
    const count = Number(step.max) || 20;
    const metadata = $('sequenceMetadata') ? JSON.parse($('sequenceMetadata').textContent) : {};
    let schedule = timeline(count, +speed.value), cursor = 0, timer, playing = false, finished = false;
    const show = () => {
        $('frame').src = recipe.value + '-step' + step.value + '.png'; $('label').textContent = 'Step ' + step.value + ' / ' + count; $('sheet').href = recipe.value + '-steps.jpg';
        const info = metadata[recipe.value], observation = info?.steps?.[step.value], bend = info?.bend;
        if ($('stepDetails')) $('stepDetails').textContent = observation?.active_sampler
            ? `Update ${observation.sampler_iteration}: ${observation.active_sampler.toUpperCase()}${observation.exchange_weight != null ? ' · latent exchange ' + Number(observation.exchange_weight.toFixed(3)) : ''}` : '';
        if ($('sequenceSettings')) $('sequenceSettings').textContent = bend?.mode === 'sampler_mix'
            ? `Sampler sharing: ${bend.mix_method} · PLMS memory: ${bend.mix_history} · ${bend.mix_method === 'alternate' ? 'block: ' + bend.mix_block + ', first: ' + bend.mix_first.toUpperCase() : 'exchange: ' + bend.mix_exchange + ' (' + bend.mix_curve + '), output: ' + bend.mix_output} · DDIM eta: ${bend.eta} · one generation seed`
            : '';
    };
    function pause() { playing = false; clearTimeout(timer); $('play').textContent = 'Play'; }
    function tick() {
        timer = setTimeout(() => {
            if (++cursor === schedule.length) {
                if (!continuous.checked) { finished = true; pause(); return; }
                cursor = 0;
            }
            step.value = schedule[cursor].step; show(); tick();
        }, schedule[cursor].delay * 10);
    }
    $('play').onclick = () => {
        if (playing) return pause();
        if (finished) { cursor = 0; step.value = 1; show(); finished = false; }
        playing = true; $('play').textContent = 'Pause'; tick();
    };
    step.oninput = () => { pause(); finished = false; cursor = +step.value - 1; show(); };
    recipe.onchange = () => { pause(); finished = false; cursor = 0; step.value = 1; show(); preload(); };
    function setSpeed() {
        schedule = timeline(count, +speed.value, continuous.checked);
        if (cursor >= schedule.length) cursor = 0;
        $('speedLabel').textContent = (100 / frameDelay(+speed.value)).toFixed(1) + ' frames/sec';
        if (playing) { clearTimeout(timer); tick(); }
    }
    speed.oninput = setSpeed;
    continuous.onchange = setSpeed;
    let preloaded = [];
    function preload() { preloaded = Array.from({length: count}, (_, i) => { const image = new Image(); image.src = recipe.value + '-step' + (i + 1) + '.png'; return image; }); }
    const decode = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
    $('export').onclick = async () => {
        const button = $('export'), status = $('status');
        const selected = recipe.value, fps = +speed.value, loop = continuous.checked, frames = timeline(count, fps, loop);
        button.disabled = true; status.textContent = 'Loading GIF frames…';
        try {
            const data = await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = selected + '-gif-data.js';
                script.onload = () => { const value = window.stepGifData; window.stepGifData = null; script.remove(); value && value.name === selected ? resolve(value) : reject(new Error('Wrong sequence data')); };
                script.onerror = () => { script.remove(); reject(new Error('GIF data could not be loaded. Keep the viewer and its data files together.')); };
                document.head.appendChild(script);
            });
            const chunks = [gifHeader(data.width, data.height, loop)];
            for (let i = 0; i < frames.length; i++) {
                const frame = frames[i], source = data.frames[frame.step - 1];
                chunks.push(gifFrame(data.width, data.height, decode(source.pixels), decode(source.palette), frame.delay));
                status.textContent = `Encoding GIF ${i + 1} / ${frames.length}…`;
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            chunks.push(Uint8Array.of(0x3b));
            const url = URL.createObjectURL(new Blob(chunks, {type: 'image/gif'}));
            const link = document.createElement('a'); link.href = url; link.download = `${selected}-${(100 / frameDelay(fps)).toFixed(1)}fps-${loop ? 'continuous' : 'once'}.gif`;
            document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 60000);
            status.textContent = `GIF exported: ${loop ? 'continuous loop' : 'one cycle'}, ${(100 / frameDelay(fps)).toFixed(1)} frames/sec, 1-second end pauses.`;
        } catch (error) { status.textContent = 'Export failed: ' + error.message; }
        finally { button.disabled = false; }
    };
    setSpeed(); show(); preload();
})();
