;(function () {
    "use strict"
    const root = typeof window !== "undefined" ? window : globalThis
    const SCHEMA = "flexdiffusion-bend-grid/v1"
    const AXES = {
        memory: "Noise memory", noise: "Noise amount", churn: "Churn strength",
        history_strength: "History / seed strength", fade_in: "Fade in", fade_out: "Fade out",
        eta: "DDIM eta", blend_start: "Low blend", blend_end: "High blend", blend_period: "Cycle / block length", donor_seed: "Second seed",
        start: "Window start", end: "Window end", every: "Pulse interval",
        mix_block: "Sampler block length", mix_exchange: "Latent exchange", mix_period: "Exchange cycle",
    }
    const clone = (value) => JSON.parse(JSON.stringify(value))
    function axesFor(bend) {
        const shared = ["fade_in", "fade_out", "start", "end", "every"]
        if (bend.mode === "sampler_mix") return [...(bend.mix_method === "coupled" ? ["mix_exchange", "mix_period"] : ["mix_block"]), "eta", "memory", "noise", ...shared]
        return [...(bend.mode === "plms_history" ? ["history_strength", ...(bend.history_source === "seed"
            ? ["blend_start", "blend_end", "blend_period", "donor_seed"] : [])]
            : bend.mode === "ddim" ? ["eta", "memory", "noise"] : ["memory", "noise", "churn"]), ...shared]
    }

    function numbers(text, label, maximum = 5) {
        const tokens = String(text).split(",").map((value) => value.trim())
        if (!tokens.length || tokens.some((value) => !value)) throw new Error(`${label}: enter comma-separated numbers`)
        const values = tokens.map(Number)
        if (values.length > maximum || values.some((value) => !Number.isFinite(value)) || new Set(values).size !== values.length) {
            throw new Error(`${label}: use up to ${maximum} distinct finite numbers`)
        }
        return values
    }

    function buildGrid(recipe, runId) {
        const helper = root.__flexDiffusionCircuitBendTest
        if (recipe.schema !== SCHEMA) throw new Error("Unknown bend grid recipe version")
        const { request, x, y, seeds } = recipe
        if (!request || !request.circuit_bend?.enabled) throw new Error("Enable a circuit bend before building a grid")
        const history = request.circuit_bend.mode === "plms_history"
        const ddim = request.circuit_bend.mode === "ddim"
        const hybrid = request.circuit_bend.mode === "sampler_mix"
        const ldm = history || ddim || hybrid
        if (!(hybrid ? ["plms", "ddim"] : history ? ["plms"] : ddim ? ["ddim"] : ["euler", "heun"]).includes(request.sampler_name)) throw new Error("Sampler does not support this bend")
        if (request.init_image || request.control_image || request.mask || request.ref_images?.length) throw new Error("Bend grids currently support plain txt2img")
        if (!Number.isInteger(request.num_inference_steps) || request.num_inference_steps < 1) throw new Error("Inference Steps must be a positive integer")
        if (ldm && request.circuit_bend.schedule !== "exact") {
            const stride = Math.floor(1000 / request.num_inference_steps)
            if (!stride || (Math.ceil(1000 / stride) - 1) * stride + 1 >= 1000) throw new Error("PLMS native schedule is invalid for this step count; try 20, 25, 40 or 50")
        }
        if (!Number.isInteger(request.width) || !Number.isInteger(request.height) || request.width < 64 || request.height < 64) throw new Error("Invalid image dimensions")
        if (typeof request.prompt !== "string") throw new Error("Grid recipe needs one resolved prompt")
        if (!x || !y || !Object.hasOwn(AXES, x.key) || !Object.hasOwn(AXES, y.key) || x.key === y.key) throw new Error("Choose two different grid axes")
        if (ldm && request.num_inference_steps > 1000) throw new Error("DDIM/PLMS support up to 1000 steps")
        const allowed = axesFor(request.circuit_bend)
        if (![x.key, y.key].every((key) => allowed.includes(key))) throw new Error("Grid axis is not available for this bend")
        if (![x.values, y.values].every((values) => Array.isArray(values) && values.length > 0 && values.length <= 5 && values.every(Number.isFinite) && new Set(values).size === values.length)) throw new Error("Each axis needs 1–5 distinct finite values")
        if (!Array.isArray(seeds) || !seeds.length || seeds.length > 8 || seeds.some((seed) => !Number.isInteger(seed) || seed < 0 || seed > 4294967295) || new Set(seeds).size !== seeds.length) throw new Error("Use 1–8 distinct generation seeds within 0–4294967295")
        if (seeds.length * (x.values.length * y.values.length + (recipe.baseline ? 1 : 0)) > 64) throw new Error("Limit a grid to 64 tasks")
        const entries = []
        function entry(seed, row, column, bend, label) {
            const reqBody = clone(request)
            reqBody.seed = seed
            reqBody.used_random_seed = false
            reqBody.num_outputs = 1
            reqBody.circuit_bend = bend
            reqBody.bend_grid = { schema: SCHEMA, run_id: runId, cell_id: `${seed}:${row}:${column}`, seed, row, column, label }
            entries.push({ seed, numOutputsTotal: 1, batchCount: 1, batchesDone: 0, reqBody })
        }
        for (const seed of seeds) {
            if (recipe.baseline) {
                const exact = ldm && request.circuit_bend.schedule === "exact"
                entry(seed, -1, -1, exact ? helper.validate({ enabled: true, mode: request.sampler_name === "plms" ? "plms_history" : "ddim",
                    schedule: "exact", history_strength: 0, eta: 0 }, request.num_inference_steps) : null,
                    exact ? "Exact-step reference (no bend)" : "Ordinary reference")
            }
            y.values.forEach((yv, row) => x.values.forEach((xv, column) => {
                const bend = helper.validate({ ...request.circuit_bend, [x.key]: xv, [y.key]: yv }, request.num_inference_steps)
                entry(seed, row, column, bend, `${AXES[x.key]} ${xv} · ${AXES[y.key]} ${yv}`)
            }))
        }
        return entries
    }

    function finalOutput(outputs) {
        const finals = (outputs || []).filter((output) => !output.is_intermediate)
        return finals.find((output) => output.intermediate_representation === "final") || finals[0]
    }

    root.__flexDiffusionBendGridTest = { SCHEMA, AXES, numbers, buildGrid, finalOutput }
    if (typeof document === "undefined") return

    function install() {
        const controls = root.__flexDiffusionCircuitBendUI
        const bendRow = document.querySelector("#circuit_bend_row")
        if (!controls || !bendRow || typeof createTask !== "function" || typeof getPrompts !== "function") {
            root.setTimeout(install, 100)
            return
        }
        if (document.querySelector("#bend_grid_row")) return
        const row = document.createElement("tr")
        row.id = "bend_grid_row"
        row.innerHTML = `<td>Compare bends:</td><td>
            <details><summary>Parameter grid</summary>
            <div style="display:grid;grid-template-columns:5em minmax(90px,1fr) minmax(110px,1.2fr);gap:6px;margin:8px 0;max-width:560px">
                <label for="bend_grid_x">Columns</label><select id="bend_grid_x"></select><input id="bend_grid_x_values" aria-label="Column values" value="0, 0.5, 1">
                <label for="bend_grid_y">Rows</label><select id="bend_grid_y"></select><input id="bend_grid_y_values" aria-label="Row values" value="0.75, 1.25">
                <label for="bend_grid_seeds">Generation seeds</label><input id="bend_grid_seeds" value="2, 3, 4, 5" style="grid-column:span 2">
            </div>
            <label><input id="bend_grid_baseline" type="checkbox" checked> Include ordinary reference per seed</label>
            <p style="margin:6px 0"><button id="bend_grid_queue" type="button" class="secondaryButton">Queue grid</button>
            <button id="bend_grid_export" type="button" class="tertiaryButton">Export recipe</button>
            <button id="bend_grid_import" type="button" class="tertiaryButton">Load recipe</button>
            <input id="bend_grid_file" type="file" accept=".json,application/json" hidden></p>
            <small>One task per cell and seed, using the current image settings. Also run selections are ignored. Click a finished cell to restore its settings.</small>
            <small id="bend_grid_status" role="status" style="display:block"></small>
            </details><div id="bend_grid_results"></div></td>`
        bendRow.insertAdjacentElement("afterend", row)
        row.querySelectorAll("input:not([type=checkbox]),select").forEach((field) => { field.style.minWidth = "0"; field.style.width = "100%" })
        const field = (key) => row.querySelector(`#bend_grid_${key}`)
        const runs = new Map()
        let previousMode
        function updateAxes() {
            row.style.display = controls.available() ? "" : "none"
            let bend
            try { bend = controls.read() } catch (_) { return }
            const mode = bend.mode
            const signature = `${mode}:${bend.history_source}:${bend.mix_method}`
            if (signature === previousMode) return
            previousMode = signature
            const allowed = axesFor(bend)
            for (const axis of ["x", "y"]) {
                field(axis).replaceChildren(...allowed.map((key) => new Option(AXES[key], key)))
            }
            field("x").value = allowed[0]
            field("y").value = mode === "sampler_mix" ? "eta" : mode === "plms_history" ? "fade_out" : "noise"
            field("x_values").value = mode === "sampler_mix" && bend.mix_method === "alternate" ? "1, 2, 4" : "0, 0.5, 1"
            field("y_values").value = mode === "plms_history" ? "0, 50" : "0.75, 1.25"
            if (mode === "sampler_mix") field("y_values").value = "0, 0.5"
        }
        function recipeFromUI() {
            if (!controls.available()) throw new Error("Choose a supported classic txt2img sampler")
            const prompts = getPrompts()
            if (prompts.length !== 1) throw new Error("Use one prompt for a comparison grid")
            const task = getCurrentUserRequest()
            task.reqBody.prompt = prompts[0]
            if (typeof setEmbeddings === "function") setEmbeddings(task)
            delete task.reqBody.bend_grid
            return { schema: SCHEMA, request: clone(task.reqBody),
                x: { key: field("x").value, values: numbers(field("x_values").value, "Columns") },
                y: { key: field("y").value, values: numbers(field("y_values").value, "Rows") },
                seeds: numbers(field("seeds").value, "Seeds", 8), baseline: field("baseline").checked }
        }
        function status(message) { field("status").textContent = message }
        function checked(action) {
            try { action() } catch (error) { status(error.message || String(error)) }
        }
        function drawRun(id, recipe, tasks) {
            const details = document.createElement("details")
            details.open = true
            const summary = document.createElement("summary")
            summary.textContent = `${recipe.request.sampler_name} · ${tasks.length} tasks · ${recipe.request.width}×${recipe.request.height} · ${id}`
            details.append(summary)
            const cells = new Map()
            for (const seed of recipe.seeds) {
                const title = document.createElement("p")
                title.textContent = `Generation seed ${seed}`
                details.append(title)
                const grid = document.createElement("div")
                grid.style.cssText = `display:grid;grid-template-columns:repeat(${recipe.x.values.length},minmax(110px,1fr));gap:6px;margin-bottom:12px;max-width:780px`
                for (const task of tasks.filter((task) => task.seed === seed)) {
                    const meta = task.reqBody.bend_grid
                    const cell = document.createElement("button")
                    cell.type = "button"
                    cell.disabled = true
                    cell.className = "tertiaryButton"
                    cell.style.cssText = "min-width:0;white-space:normal;padding:5px"
                    if (meta.row === -1) cell.style.gridColumn = "1 / -1"
                    const label = document.createElement("span")
                    label.textContent = `${meta.label} · queued`
                    cell.append(label)
                    const saved = clone(task)
                    cell.addEventListener("click", () => {
                        restoreTaskToUI(saved)
                        status(`Restored seed ${seed}: ${meta.label}`)
                    })
                    cells.set(meta.cell_id, { cell, label, task: saved })
                    grid.append(cell)
                }
                details.append(grid)
            }
            field("results").prepend(details)
            runs.set(id, cells)
            return cells
        }
        field("queue").addEventListener("click", () => checked(() => {
            if (typeof SD !== "undefined" && !SD.isServerAvailable()) throw new Error("The server is not available")
            const recipe = recipeFromUI()
            const id = `bend-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
            const tasks = buildGrid(recipe, id) // Validate every cell before enqueueing anything.
            const cells = drawRun(id, recipe, tasks)
            let queued = 0
            try {
                for (const task of tasks) {
                    createTask(task)
                    // Core createTask uses Date.now(); several grid cells can
                    // otherwise receive the same DOM id in one synchronous batch.
                    const element = task.outputContainer?.closest(".imageTaskContainer")
                    if (element) element.id = `imageTaskContainer-${id}-${task.reqBody.bend_grid.cell_id.replaceAll(":", "-")}`
                    queued++
                }
            } catch (error) {
                for (const task of tasks.slice(queued)) cells.get(task.reqBody.bend_grid.cell_id).label.textContent += " · not queued"
                throw new Error(`Queued ${queued}/${tasks.length} tasks: ${error.message}`)
            }
            status(`Queued ${tasks.length} tasks. Outputs also appear in ordinary task history.`)
            if (typeof updateInitialText === "function") updateInitialText()
        }))
        field("export").addEventListener("click", () => checked(() => {
            const recipe = recipeFromUI()
            buildGrid(recipe, "preflight")
            saveAs(new Blob([JSON.stringify(recipe, null, 2)], { type: "application/json" }), "circuit-bend-grid.json")
            status("Exported grid recipe, including prompt, image settings and both axes.")
        }))
        field("import").addEventListener("click", () => field("file").click())
        field("file").addEventListener("change", async () => {
            try {
                const file = field("file").files[0]
                if (!file) return
                const recipe = JSON.parse(await file.text())
                buildGrid(recipe, "preflight")
                restoreTaskToUI({ reqBody: recipe.request, seed: recipe.request.seed, numOutputsTotal: 1, batchCount: 1 })
                for (const axis of ["x", "y"]) {
                    field(axis).value = recipe[axis].key
                    field(`${axis}_values`).value = recipe[axis].values.join(", ")
                }
                field("seeds").value = recipe.seeds.join(", ")
                field("baseline").checked = !!recipe.baseline
                status("Recipe loaded. Queue grid to render it.")
            } catch (error) { status(error.message || String(error)) }
            finally { field("file").value = "" }
        })
        document.addEventListener("on_render_task_success", (event) => {
            const { reqBody, stepUpdate } = event.detail
            const meta = reqBody?.bend_grid
            const target = runs.get(meta?.run_id)?.get(meta?.cell_id)
            if (!target) return
            const output = finalOutput(stepUpdate?.output)
            const src = output?.data || output?.path
            if (!src) { target.label.textContent = `${meta.label} · no final image`; return }
            const image = document.createElement("img")
            image.src = src
            image.alt = `Seed ${meta.seed}, ${meta.label}`
            image.style.cssText = "display:block;width:100%;max-width:256px;height:auto;margin:auto"
            target.cell.querySelector("img")?.remove()
            target.cell.prepend(image)
            target.label.textContent = `${meta.label}${output.sampling_status === "interrupted" ? " · partial" : ""}`
            target.cell.disabled = false
        })
        document.addEventListener("on_render_task_fail", (event) => {
            const meta = (event.detail.reqBody || event.detail.task?.reqBody)?.bend_grid
            const target = runs.get(meta?.run_id)?.get(meta?.cell_id)
            if (target) target.label.textContent = `${meta.label} · failed`
        })
        document.addEventListener("circuit_bend_change", updateAxes)
        if (typeof taskConfigSetup !== "undefined") taskConfigSetup.taskConfig.bend_grid = {
            label: "Bend grid", visible: ({ reqBody }) => !!reqBody?.bend_grid,
            value: ({ reqBody }) => reqBody.bend_grid.label,
        }
        updateAxes()
    }
    install()
})()
