;(function () {
    "use strict"
    const root = typeof window !== "undefined" ? window : globalThis
    const defaults = {
        enabled: false, mode: "noise", churn: 10, noise: 1, start: 1, end: null,
        window_unit: "iterations", every: 1, seed: 0, pattern: "fresh", memory: null,
        fade_in: 0, fade_out: 0, history_strength: 1,
        schedule: "native", eta: .5, history_source: "freeze", blend_target: "history", donor_seed: 3,
        blend_curve: "constant", blend_start: 0, blend_end: 1, blend_period: 4,
        mix_method: "alternate", mix_first: "plms", mix_block: 1, mix_history: "shared",
        mix_exchange: .5, mix_output: "blend", mix_curve: "constant", mix_period: 8,
    }

    function resolveWindow(value, steps) {
        return value.window_unit === "percent"
            ? [Math.max(1, Math.ceil(steps * value.start / 100)), Math.max(1, Math.ceil(steps * (value.end ?? 100) / 100))]
            : [value.start, value.end ?? steps]
    }

    function envelope(value, iteration, steps) {
        const [first, last] = resolveWindow(value, steps)
        if (iteration < first || iteration > last || (iteration - first) % value.every) return 0
        if (first === last) return 1
        const position = (iteration - first) / (last - first)
        return Math.max(0, Math.min(1, value.fade_in ? position * 100 / value.fade_in : 1,
            value.fade_out ? (1 - position) * 100 / value.fade_out : 1))
    }

    function blendWeight(value, iteration, steps) {
        const [first, last] = resolveWindow(value, steps)
        const offset = iteration - first
        let phase = 1
        if (value.blend_curve === "progressive") phase = Math.min(1, Math.max(0, offset / Math.max(1, last - first)))
        if (value.blend_curve === "oscillate") phase = (1 - Math.cos(2 * Math.PI * offset / value.blend_period)) / 2
        if (value.blend_curve === "interlace") phase = Math.floor(offset / value.blend_period) % 2
        const mix = value.blend_start + phase * (value.blend_end - value.blend_start)
        return value.history_strength * mix * envelope(value, iteration, steps)
    }

    function validate(options, steps) {
        const value = { ...defaults, ...options }
        if (!["noise", "plms_history", "ddim", "sampler_mix"].includes(value.mode)) throw new Error("Unknown circuit bend mode")
        for (const [key, choices] of Object.entries({ schedule: ["native", "exact"], history_source: ["freeze", "seed"],
            mix_method: ["alternate", "coupled"], mix_first: ["plms", "ddim"], mix_history: ["shared", "stale", "reset"],
            mix_output: ["plms", "ddim", "blend"], mix_curve: ["constant", "progressive", "oscillate"],
            blend_target: ["history", "trajectory", "independent"], blend_curve: ["constant", "progressive", "oscillate", "interlace"] })) {
            if (!choices.includes(value[key])) throw new Error(`Unknown circuit bend ${key}`)
        }
        if (!["iterations", "percent"].includes(value.window_unit)) throw new Error("Unknown circuit bend window unit")
        for (const [key, min, max, integer] of [
            ["churn", 0, 100, false], ["noise", 0, 4, false],
            ["every", 1, 1000, true], ["seed", 0, 4294967295, true],
            ["fade_in", 0, 100, false], ["fade_out", 0, 100, false], ["history_strength", 0, 1, false],
            ["eta", 0, 1, false], ["donor_seed", 0, 4294967295, true], ["blend_period", 1, 1000, true],
            ["blend_start", 0, 1, false], ["blend_end", 0, 1, false],
            ["mix_block", 1, 1000, true], ["mix_period", 1, 1000, true], ["mix_exchange", 0, 1, false],
        ]) {
            if (!Number.isFinite(value[key]) || value[key] < min || value[key] > max || (integer && !Number.isInteger(value[key]))) {
                throw new Error(`Circuit bend ${key} must be ${integer ? "an integer " : ""}between ${min} and ${max}`)
            }
        }
        const percent = value.window_unit === "percent"
        const maximum = percent ? 100 : steps
        for (const key of ["start", "end"]) {
            if (key === "end" && value.end === null) continue
            if (!Number.isFinite(value[key]) || value[key] < (percent ? 0 : 1) || value[key] > maximum || (!percent && !Number.isInteger(value[key]))) {
                throw new Error(`Circuit bend ${key} must be ${percent ? "within 0–100%" : "an iteration within Inference Steps"}`)
            }
        }
        if (value.end !== null && value.end < value.start) throw new Error("Circuit bend end must be at or after start")
        if (!["fresh", "held"].includes(value.pattern)) {
            throw new Error("Unknown circuit bend noise pattern")
        }
        if (value.memory !== null && (!Number.isFinite(value.memory) || value.memory < 0 || value.memory > 1)) {
            throw new Error("Noise memory must be between 0 and 1")
        }
        return value
    }

    function describe(b) {
        if (b.mode === "sampler_mix") return `PLMS ↔ DDIM ${b.mix_method}; first ${b.mix_first}, block ${b.mix_block}; history ${b.mix_history}; exchange ${b.mix_exchange} ${b.mix_curve}, period ${b.mix_period}; output ${b.mix_output}; DDIM eta ${b.eta}, noise ${b.noise}, memory ${b.memory ?? 0}, noise seed ${b.seed}; ${b.schedule} schedule; window ${b.start}–${b.end ?? "final"} ${b.window_unit}, pulse ${b.every}, fades ${b.fade_in}/${b.fade_out}%`
        const memory = b.memory ?? (b.pattern === "held" ? 1 : 0)
        const effect = b.mode === "plms_history" ? (b.history_source === "seed"
            ? `PLMS seed ${b.donor_seed} ${b.blend_target}, ${b.blend_curve} ${b.blend_start}→${b.blend_end}, period ${b.blend_period}, strength ${b.history_strength}`
            : `PLMS freeze ${b.history_strength ?? 1}`) : b.mode === "ddim" ? `DDIM eta ${b.eta}, noise ${b.noise}, memory ${memory}` : `churn ${b.churn}, noise ${b.noise}, memory ${memory}`
        return `${effect}; ${b.start}–${b.end ?? "final"}${b.window_unit === "percent" ? "%" : " iterations"} / ${b.every}; fades ${b.fade_in ?? 0}/${b.fade_out ?? 0}%${b.mode === "plms_history" ? "" : `; noise seed ${b.seed}`}${b.mode === "noise" ? "" : `; ${b.schedule ?? "native"} schedule`}`
    }

    root.__flexDiffusionCircuitBendTest = { validate, defaults, resolveWindow, envelope, blendWeight, describe }
    if (typeof document === "undefined") return

    function install() {
        const sampler = document.querySelector("#sampler_name")
        const steps = document.querySelector("#num_inference_steps")
        if (!sampler || !steps || typeof getCurrentUserRequest !== "function") {
            root.setTimeout(install, 50)
            return
        }
        if (document.querySelector("#circuit_bend_row")) return
        const row = document.createElement("tr")
        row.id = "circuit_bend_row"
        row.className = "pl-5"
        row.innerHTML = `
            <td><label for="circuit_bend_enabled">Circuit bend:</label></td>
            <td>
                <label><input type="checkbox" id="circuit_bend_enabled"> <span id="circuit_bend_title">Experimental noise</span></label>
                <details id="circuit_bend_controls" style="margin-top:4px">
                    <summary>Bend controls</summary>
                    <div id="circuit_bend_mix_toggle"><label><input id="circuit_bend_mix_enabled" type="checkbox"> Interlace sampler types: PLMS ↔ DDIM</label></div>
                    <div id="circuit_bend_mix_fields" style="display:none">
                        <label for="circuit_bend_mix_method">Sampler sharing</label><select id="circuit_bend_mix_method"><option value="alternate">Alternate one live latent</option><option value="coupled">Coupled parallel trajectories</option></select><br>
                        <label for="circuit_bend_mix_first">First / outside-window sampler</label><select id="circuit_bend_mix_first"><option value="plms">PLMS</option><option value="ddim">DDIM</option></select>
                        <label for="circuit_bend_mix_block">Iterations per block</label><input id="circuit_bend_mix_block" type="number" min="1" max="1000" step="1" value="1" style="width:5em"><br>
                        <label for="circuit_bend_mix_history">PLMS memory</label><select id="circuit_bend_mix_history"><option value="shared">Shared predictions</option><option value="stale">Keep stale PLMS history (glitch)</option><option value="reset">Reset at handoff / exchange</option></select>
                        <div id="circuit_bend_coupling_fields">
                            <label for="circuit_bend_mix_exchange">Latent exchange</label><input id="circuit_bend_mix_exchange" type="number" min="0" max="1" step="0.05" value="0.5" style="width:5em">
                            <label for="circuit_bend_mix_curve">Exchange movement</label><select id="circuit_bend_mix_curve"><option value="constant">Constant</option><option value="progressive">Progressive 0 → amount</option><option value="oscillate">Oscillate 0 ↔ amount</option></select>
                            <label for="circuit_bend_mix_period">Cycle iterations</label><input id="circuit_bend_mix_period" type="number" min="1" max="1000" value="8" style="width:5em"><br>
                            <label for="circuit_bend_mix_output">Export trajectory</label><select id="circuit_bend_mix_output"><option value="blend">Equal blend</option><option value="plms">PLMS branch</option><option value="ddim">DDIM branch</option></select>
                        </div>
                        <small>One generation seed, one prompt, a shared timestep schedule. Alternate: PLMS step 1 → DDIM step 2 → PLMS step 3 when block=1. Coupled: both advance each step, then exchange latents simultaneously; 0 leaves them separate, 0.5 joins them, 1 swaps them. Coupled uses roughly twice the sampling work. Shared memory follows both predictors; stale/reset modes deliberately bend solver history.</small>
                    </div>
                    <div id="circuit_bend_schedule_fields"><label for="circuit_bend_schedule">Step schedule</label>
                        <select id="circuit_bend_schedule"><option value="native">Native (legacy step counts)</option><option value="exact">Exact steps (supports 16, 27, etc.)</option></select></div>
                    <div id="circuit_bend_ddim_fields"><label for="circuit_bend_eta">DDIM eta</label>
                        <input id="circuit_bend_eta" type="number" min="0" max="1" step="0.05" value="0.5">
                        <br><small>0 deterministic; 1 full stochastic DDIM. Window and fades scale eta. Noise above 1 and held noise deliberately bend the usual distribution.</small></div>
                    <div id="circuit_bend_noise_fields" style="display:grid;grid-template-columns:auto 7em;gap:6px;margin-top:8px;max-width:350px">
                        <label for="circuit_bend_churn">Churn strength</label><input id="circuit_bend_churn" type="number" min="0" max="100" step="1" value="10">
                        <label for="circuit_bend_noise">Noise amount</label><input id="circuit_bend_noise" type="number" min="0" max="4" step="0.05" value="1">
                        <label for="circuit_bend_memory">Noise memory</label><input id="circuit_bend_memory" type="range" min="0" max="1" step="0.05" value="0">
                        <small>0 fresh · 1 held</small><output id="circuit_bend_memory_value">0</output>
                        <label for="circuit_bend_seed">Noise seed</label><input id="circuit_bend_seed" type="number" min="0" max="4294967295" step="1" value="0">
                    </div>
                    <div id="circuit_bend_history_fields" style="display:none;margin-top:8px">
                        <label for="circuit_bend_history_source">History source</label>
                        <select id="circuit_bend_history_source"><option value="freeze">Freeze own history</option><option value="seed">Second seed</option></select><br>
                        <label for="circuit_bend_history_strength">Bend strength</label>
                        <input id="circuit_bend_history_strength" type="number" min="0" max="1" step="0.05" value="1" style="width:7em">
                        <br><small>Freeze holds three predictions after warm-up. Second seed follows an independent trajectory and costs about twice as much sampling. Strength 0 disables mixing.</small>
                        <div id="circuit_bend_donor_fields" style="display:grid;grid-template-columns:auto 10em;gap:6px;margin-top:8px;max-width:420px">
                            <label for="circuit_bend_donor_seed">Second seed</label><input id="circuit_bend_donor_seed" type="number" min="0" max="4294967295" step="1" value="3">
                            <label for="circuit_bend_blend_target">Mix into</label><select id="circuit_bend_blend_target"><option value="history">History (glitch)</option><option value="independent">Independent trajectories</option><option value="trajectory">Trajectory feedback (glitch)</option></select>
                            <label for="circuit_bend_blend_curve">Movement</label><select id="circuit_bend_blend_curve"><option value="constant">Constant</option><option value="progressive">Progressive</option><option value="oscillate">Oscillate smoothly</option><option value="interlace">Interlace hard blocks</option></select>
                            <label for="circuit_bend_blend_start">Starting / low blend</label><input id="circuit_bend_blend_start" type="number" min="0" max="1" step="0.05" value="0">
                            <label for="circuit_bend_blend_end">Ending / high blend</label><input id="circuit_bend_blend_end" type="number" min="0" max="1" step="0.05" value="1">
                            <label for="circuit_bend_blend_period">Cycle / block length</label><input id="circuit_bend_blend_period" type="number" min="1" max="1000" step="1" value="4">
                        </div>
                        <small id="circuit_bend_donor_help">Independent trajectories: blend 0 uses seed A, 1 uses seed B; neither branch is altered. History and feedback modes carry changes into later steps. Progressive spans the window; constant uses high blend. Cycle length is iterations per smooth cycle, or per A/B block. Swapping low/high reverses direction.</small>
                    </div>
                    <div style="display:grid;grid-template-columns:auto 7em;gap:6px;margin-top:8px;max-width:350px">
                        <label for="circuit_bend_preset">Window preset</label><select id="circuit_bend_preset"><option value="custom">Custom</option><option value="early">Early</option><option value="middle">Middle</option><option value="late">Late</option><option value="all">Whole render</option></select>
                        <label for="circuit_bend_window_unit">Window units</label><select id="circuit_bend_window_unit"><option value="iterations">Iterations</option><option value="percent">Percent</option></select>
                        <label for="circuit_bend_start">First iteration</label><input id="circuit_bend_start" type="number" min="1" step="1" value="1">
                        <label for="circuit_bend_end">Last iteration</label><input id="circuit_bend_end" type="number" min="1" step="1" placeholder="Final">
                        <label for="circuit_bend_every">Pulse every N iterations</label><input id="circuit_bend_every" type="number" min="1" max="1000" step="1" value="1">
                        <label for="circuit_bend_fade_in">Fade in (% of window)</label><input id="circuit_bend_fade_in" type="number" min="0" max="100" step="5" value="0">
                        <label for="circuit_bend_fade_out">Fade out (% of window)</label><input id="circuit_bend_fade_out" type="number" min="0" max="100" step="5" value="0">
                    </div>
                    <small id="circuit_bend_window_summary" style="display:block;margin-top:6px"></small>
                    <small>Fades start/end at zero; a one-iteration window remains a full pulse. High noise and frozen history are experimental.</small>
                </details>
                <small id="circuit_bend_status" style="display:block"></small>
            </td>`
        sampler.closest("tr").insertAdjacentElement("afterend", row)
        const field = (name) => row.querySelector(`#circuit_bend_${name}`)
        const backend = document.querySelector("#backend")
        const init = document.querySelector("#init_image_preview_container")
        const samplerMode = () => ["plms", "ddim"].includes(sampler.value) && field("mix_enabled").checked ? "sampler_mix" : sampler.value === "plms" ? "plms_history" : sampler.value === "ddim" ? "ddim" : "noise"
        let mode = samplerMode()
        function available() {
            return (!backend || backend.value === "ed_classic") && !init?.classList.contains("has-image")
                && (mode === "sampler_mix" ? ["plms", "ddim"].includes(sampler.value) : mode === "plms_history" ? sampler.value === "plms" : mode === "ddim" ? sampler.value === "ddim" : ["euler", "heun"].includes(sampler.value))
        }
        function read() {
            const result = { enabled: field("enabled").checked, mode, window_unit: field("window_unit").value }
            for (const key of ["churn", "noise", "start", "every", "seed", "memory", "fade_in", "fade_out", "history_strength", "eta", "donor_seed", "blend_start", "blend_end", "blend_period", "mix_block", "mix_exchange", "mix_period"]) {
                result[key] = field(key).value.trim() === "" ? NaN : Number(field(key).value)
            }
            result.pattern = result.memory === 1 ? "held" : "fresh"
            for (const key of ["schedule", "history_source", "blend_target", "blend_curve", "mix_method", "mix_first", "mix_history", "mix_output", "mix_curve"]) result[key] = field(key).value
            result.end = field("end").value.trim() === "" ? null : Number(field("end").value)
            return validate(result, Number(steps.value))
        }
        function update() {
            mode = samplerMode()
            const supported = available()
            row.style.display = !backend || backend.value === "ed_classic" ? "" : "none"
            field("enabled").disabled = !supported
            field("controls").style.display = supported ? "" : "none"
            field("title").textContent = mode === "sampler_mix" ? "Experimental PLMS ↔ DDIM" : mode === "plms_history" ? "Experimental history / seed mixing" : mode === "ddim" ? "DDIM noise / exact steps" : "Experimental noise"
            field("mix_toggle").style.display = ["plms", "ddim"].includes(sampler.value) ? "" : "none"
            field("mix_fields").style.display = mode === "sampler_mix" ? "" : "none"
            field("coupling_fields").style.display = field("mix_method").value === "coupled" ? "" : "none"
            field("mix_first").disabled = field("mix_block").disabled = mode === "sampler_mix" && field("mix_method").value === "coupled"
            field("noise_fields").style.display = mode !== "plms_history" ? "grid" : "none"
            field("schedule_fields").style.display = mode !== "noise" ? "" : "none"
            field("ddim_fields").style.display = ["ddim", "sampler_mix"].includes(mode) ? "" : "none"
            field("churn").style.display = mode === "noise" ? "" : "none"
            row.querySelector('label[for="circuit_bend_churn"]').style.display = field("churn").style.display
            field("history_fields").style.display = mode === "plms_history" ? "" : "none"
            field("donor_fields").style.display = field("history_source").value === "seed" ? "grid" : "none"
            field("donor_help").style.display = field("history_source").value === "seed" ? "" : "none"
            field("memory_value").value = field("memory").value
            field("status").textContent = supported ? "" : "Available with Euler, Heun, DDIM or PLMS in classic txt2img; off for this sampler."
            const percent = field("window_unit").value === "percent"
            for (const key of ["start", "end"]) {
                row.querySelector(`label[for="circuit_bend_${key}"]`).textContent = `${key === "start" ? "First" : "Last"} ${percent ? "percent" : "iteration"}`
                field(key).min = percent ? "0" : "1"
                field(key).step = percent ? "0.1" : "1"
            }
            try {
                const options = read()
                const requested = Number(steps.value)
                const actual = mode !== "noise" && !(options.enabled && options.schedule === "exact") && requested >= 1 && requested <= 1000 ? Math.ceil(1000 / Math.floor(1000 / requested)) : requested
                const [first, last] = resolveWindow(options, actual)
                const donor = mode === "plms_history" && options.history_source === "seed"
                const freeze = mode === "plms_history" && !donor
                const pulses = Array.from({length: Math.min(actual, 1000)}, (_, i) => i + 1).filter(i =>
                    i >= (freeze ? 4 : donor && options.blend_target === "history" ? 2 : 1)
                    && (donor ? blendWeight(options, i, actual) : envelope(options, i, actual)) > 0).length
                field("window_summary").textContent = `Iterations ${first}–${last} of ${actual}; ${pulses} selected updates; ${freeze ? "freeze starts after native warm-up" : donor ? "second seed advances throughout the render" : "memory advances on active noise pulses"}.`
                if (mode === "sampler_mix") field("window_summary").textContent = options.mix_method === "alternate"
                    ? `${actual} shared-latent updates; switch every ${options.mix_block} iterations in ${first}–${last}. Outside active pulses use ${options.mix_first.toUpperCase()}. Fades scale DDIM noise; switching stays discrete.`
                    : `${actual} updates per branch; exchange ${options.mix_exchange} (${options.mix_curve}) in ${first}–${last}; export ${options.mix_output}.`
            } catch (error) {
                field("window_summary").textContent = ""
                if (supported && field("enabled").checked) field("status").textContent = error.message
            }
            document.dispatchEvent(new Event("circuit_bend_change"))
        }
        row.addEventListener("input", update)
        row.addEventListener("change", update)
        field("mix_enabled").addEventListener("change", () => {
            if (field("mix_enabled").checked) { field("eta").value = "0"; field("schedule").value = "exact" }
            update()
        })
        field("preset").addEventListener("change", () => {
            const presets = { early: [0, 35], middle: [25, 75], late: [65, 100], all: [0, 100] }
            const value = presets[field("preset").value]
            if (value) {
                field("window_unit").value = "percent"
                field("start").value = value[0]
                field("end").value = value[1]
            }
            update()
        })
        for (const key of ["start", "end", "window_unit"]) field(key).addEventListener("input", () => { field("preset").value = "custom" })
        sampler.addEventListener("change", () => { mode = samplerMode(); update() })
        steps.addEventListener("input", update)
        backend?.addEventListener("change", update)
        if (init) new MutationObserver(update).observe(init, { attributes: true, attributeFilter: ["class"] })
        const originalRequest = getCurrentUserRequest
        root.__flexDiffusionCircuitBendUI = { available, read }
        root.__flexDiffusionReadCircuitBend = () => available() && field("enabled").checked ? read() : null
        getCurrentUserRequest = function (...args) {
            const task = originalRequest.apply(this, args)
            if (available() && field("enabled").checked) task.reqBody.circuit_bend = read()
            else delete task.reqBody.circuit_bend
            return task
        }
        if (typeof restoreTaskToUI === "function") {
            const originalRestore = restoreTaskToUI
            restoreTaskToUI = function (...args) {
                const result = originalRestore.apply(this, args)
                if (!args[0]?.reqBody || args[1]?.includes("circuit_bend")) return result
                let saved = args[0]?.reqBody?.circuit_bend
                if (typeof saved === "string") saved = JSON.parse(saved)
                const value = { ...defaults, ...saved }
                mode = saved ? value.mode : samplerMode()
                field("mix_enabled").checked = saved?.mode === "sampler_mix"
                field("enabled").checked = value.enabled
                field("memory").value = value.memory ?? (value.pattern === "held" ? 1 : 0)
                for (const key of ["churn", "noise", "start", "end", "every", "seed", "window_unit", "fade_in", "fade_out", "history_strength", "schedule", "eta", "history_source", "blend_target", "donor_seed", "blend_curve", "blend_start", "blend_end", "blend_period", "mix_method", "mix_first", "mix_block", "mix_history", "mix_exchange", "mix_output", "mix_curve", "mix_period"]) field(key).value = value[key] ?? ""
                field("preset").value = "custom"
                update()
                return result
            }
        }
        if (typeof taskConfigSetup !== "undefined") {
            taskConfigSetup.taskConfig.circuit_bend = {
                label: "Circuit bend (experimental)",
                visible: ({ reqBody }) => !!reqBody?.circuit_bend?.enabled,
                value: ({ reqBody }) => describe(reqBody.circuit_bend),
            }
        }
        update()
    }
    install()
})()
