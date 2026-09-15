;(function () {
    "use strict"

    const root = typeof window !== "undefined" ? window : globalThis
    const SCHEMA = "flexdiffusion-benchmark/v1"
    const MAX_SEED = 2 ** 32 - 1
    const SETTING_KEYS = new Set([
        "prompt",
        "negative_prompt",
        "seed",
        "model",
        "width",
        "height",
        "inference_steps",
        "guidance_scale",
        "output_format",
        "output_quality",
        "capture_iterations",
        "endpoint_solver_states",
        "num_outputs",
        "num_outputs_parallel",
        "clip_skip",
        "vae_model",
        "scheduler",
        "vram_usage_level",
        "stream_image_progress",
        "block_nsfw",
        "show_only_filtered_image",
    ])

    const BUILT_IN_EXAMPLE = {
        schema: SCHEMA,
        name: "Cyberdino sparse sampler baseline",
        description:
            "SD1.4 seed-2 sampler sweep using sparse denoised trajectory observations. " +
            "Designed as a compact reproducible benchmark/review archive.",
        defaults: {
            prompt: "cyberpunk dinosaur mercenary",
            negative_prompt: "",
            seed: 2,
            model: "sd-v1-4",
            width: 512,
            height: 512,
            inference_steps: 20,
            guidance_scale: 7.5,
            output_format: "png",
            output_quality: 75,
            capture_iterations: "10, 16, 18-20",
            endpoint_solver_states: false,
            num_outputs: 1,
            num_outputs_parallel: 1,
            clip_skip: false,
            vae_model: "",
            vram_usage_level: "low",
            stream_image_progress: false,
            block_nsfw: false,
            show_only_filtered_image: false,
        },
        samplers: [
            "dpmpp_2m",
            "heun",
            "euler",
            "euler_a",
            "dpm2",
            "dpm2_a",
            "lms",
            "dpmpp_2s_a",
            "dpmpp_sde",
        ],
        cases: [{ id: "baseline", label: "SD1.4 seed 2 sparse sampler sweep" }],
    }

    function isPlainObject(value) {
        return !!value && typeof value === "object" && !Array.isArray(value)
    }

    function cloneJson(value) {
        return JSON.parse(JSON.stringify(value))
    }

    function nonEmptyString(value, label) {
        if (typeof value !== "string" || !value.trim()) {
            throw new Error(`${label} must be a non-empty string`)
        }
        return value.trim()
    }

    function integer(value, label, min, max) {
        if (!Number.isInteger(value) || value < min || (max !== undefined && value > max)) {
            throw new Error(`${label} must be an integer >= ${min}${max === undefined ? "" : ` and <= ${max}`}`)
        }
        return value
    }

    function finiteNumber(value, label, min) {
        if (typeof value !== "number" || !Number.isFinite(value) || (min !== undefined && value < min)) {
            throw new Error(`${label} must be a finite number${min === undefined ? "" : ` >= ${min}`}`)
        }
        return value
    }

    function normalizeSamplers(value, label) {
        if (!Array.isArray(value) || !value.length) {
            throw new Error(`${label} must contain at least one sampler`)
        }
        const ordered = []
        const seen = new Set()
        value.forEach((item, index) => {
            const name = nonEmptyString(item, `${label}[${index}]`)
            if (!seen.has(name)) {
                seen.add(name)
                ordered.push(name)
            }
        })
        return ordered
    }

    function normalizeSettings(value, label) {
        if (value === undefined || value === null) return {}
        if (!isPlainObject(value)) throw new Error(`${label} must be an object`)

        const out = {}
        Object.entries(value).forEach(([key, item]) => {
            if (!SETTING_KEYS.has(key)) throw new Error(`${label} contains unknown setting '${key}'`)
            switch (key) {
                case "prompt":
                case "negative_prompt":
                case "model":
                case "vae_model":
                case "scheduler":
                    if (typeof item !== "string") throw new Error(`${label}.${key} must be a string`)
                    out[key] = item
                    break
                case "seed":
                    out[key] = integer(item, `${label}.seed`, 0, MAX_SEED)
                    break
                case "width":
                case "height":
                    out[key] = integer(item, `${label}.${key}`, 64)
                    if (out[key] % 64 !== 0) throw new Error(`${label}.${key} must be a multiple of 64`)
                    break
                case "inference_steps":
                    out[key] = integer(item, `${label}.inference_steps`, 1)
                    break
                case "guidance_scale":
                    out[key] = finiteNumber(item, `${label}.guidance_scale`, 0)
                    break
                case "output_format": {
                    const format = nonEmptyString(item, `${label}.output_format`).toLowerCase()
                    if (!["png", "jpeg", "webp"].includes(format)) {
                        throw new Error(`${label}.output_format must be png, jpeg, or webp`)
                    }
                    out[key] = format
                    break
                }
                case "output_quality":
                    out[key] = integer(item, `${label}.output_quality`, 10, 95)
                    break
                case "capture_iterations":
                    out[key] = nonEmptyString(item, `${label}.capture_iterations`)
                    break
                case "endpoint_solver_states":
                case "clip_skip":
                case "stream_image_progress":
                case "block_nsfw":
                case "show_only_filtered_image":
                    if (typeof item !== "boolean") throw new Error(`${label}.${key} must be boolean`)
                    out[key] = item
                    break
                case "num_outputs":
                case "num_outputs_parallel":
                    out[key] = integer(item, `${label}.${key}`, 1)
                    break
                case "vram_usage_level": {
                    const mode = nonEmptyString(item, `${label}.vram_usage_level`).toLowerCase()
                    if (!["low", "balanced", "high"].includes(mode)) {
                        throw new Error(`${label}.vram_usage_level must be low, balanced, or high`)
                    }
                    out[key] = mode
                    break
                }
                default:
                    throw new Error(`Unsupported benchmark setting '${key}'`)
            }
        })
        return out
    }

    function validateResolvedSettings(settings, label) {
        const required = [
            "prompt",
            "seed",
            "model",
            "width",
            "height",
            "inference_steps",
            "guidance_scale",
            "output_format",
            "capture_iterations",
        ]
        required.forEach((key) => {
            if (settings[key] === undefined) throw new Error(`${label} is missing required setting '${key}'`)
        })
        if ((settings.num_outputs_parallel || 1) > (settings.num_outputs || 1)) {
            throw new Error(`${label}.num_outputs_parallel cannot exceed num_outputs`)
        }
        return settings
    }

    function normalizeBenchmarkConfig(input) {
        let value = input
        if (typeof value === "string") {
            try {
                value = JSON.parse(value)
            } catch (error) {
                throw new Error(`Benchmark JSON is invalid: ${error.message}`)
            }
        }
        if (!isPlainObject(value)) throw new Error("Benchmark must be a JSON object")
        if (value.schema !== SCHEMA) throw new Error(`Benchmark schema must be '${SCHEMA}'`)

        const name = nonEmptyString(value.name, "Benchmark name")
        const description = value.description === undefined ? "" : String(value.description)
        const defaults = normalizeSettings(value.defaults, "defaults")
        const rootSamplers = value.samplers === undefined ? null : normalizeSamplers(value.samplers, "samplers")
        const rawCases = value.cases === undefined ? [{ id: "default", label: "Default" }] : value.cases
        if (!Array.isArray(rawCases) || !rawCases.length) throw new Error("cases must contain at least one case")

        const ids = new Set()
        const cases = rawCases.map((rawCase, index) => {
            if (!isPlainObject(rawCase)) throw new Error(`cases[${index}] must be an object`)
            const id = nonEmptyString(rawCase.id ?? `case-${index + 1}`, `cases[${index}].id`)
            if (!/^[A-Za-z0-9._-]+$/.test(id)) {
                throw new Error(`cases[${index}].id may contain only letters, numbers, dot, underscore, and hyphen`)
            }
            if (ids.has(id)) throw new Error(`Duplicate benchmark case id '${id}'`)
            ids.add(id)
            const label = rawCase.label === undefined ? id : nonEmptyString(rawCase.label, `cases[${index}].label`)
            const override = normalizeSettings(rawCase.settings, `cases[${index}].settings`)
            const samplers = rawCase.samplers === undefined ? rootSamplers : normalizeSamplers(rawCase.samplers, `cases[${index}].samplers`)
            if (!samplers) throw new Error(`cases[${index}] has no sampler list and no root samplers are defined`)
            return {
                id,
                label,
                settings: validateResolvedSettings({ ...defaults, ...override }, `cases[${index}]`),
                samplers: [...samplers],
            }
        })
        return { schema: SCHEMA, name, description, cases }
    }

    function benchmarkSummary(config, sparseHelper) {
        const normalized = normalizeBenchmarkConfig(config)
        let tasks = 0
        let outputs = 0
        normalized.cases.forEach((testCase) => {
            tasks += testCase.samplers.length
            if (!sparseHelper) return
            const steps = sparseHelper.parseCaptureExpression(
                testCase.settings.capture_iterations,
                testCase.settings.inference_steps
            )
            const count = sparseHelper.outputCountSummary(
                steps,
                testCase.settings.inference_steps,
                !!testCase.settings.endpoint_solver_states
            )
            outputs += count.totalOutputs * testCase.samplers.length * (testCase.settings.num_outputs || 1)
        })
        return { cases: normalized.cases.length, tasks, outputs: sparseHelper ? outputs : null }
    }

    root.__flexDiffusionBenchmarkRunTest = {
        schema: SCHEMA,
        normalizeBenchmarkConfig,
        benchmarkSummary,
        builtInExample: cloneJson(BUILT_IN_EXAMPLE),
    }

    if (typeof document === "undefined") return
    if (root.__flexDiffusionBenchmarkRunInstalled) return

    function install() {
        const makeImageButton = document.querySelector("#makeImage")
        const samplerField = document.querySelector("#sampler_name")
        const captureField = document.querySelector("#output_after_step")
        const endpointField = document.querySelector("#capture_endpoint_solver_states")
        const ready =
            makeImageButton &&
            samplerField &&
            document.querySelector("#sampler_compare_options") &&
            captureField &&
            endpointField &&
            document.querySelector("#vram_usage_level") &&
            typeof makeImage === "function" &&
            typeof getCurrentUserRequest === "function" &&
            root.__flexDiffusionSparseCaptureTest
        if (!ready) {
            root.setTimeout(install, 100)
            return
        }
        if (root.__flexDiffusionBenchmarkRunInstalled) return
        root.__flexDiffusionBenchmarkRunInstalled = true

        let activeContext = null
        const sparseHelper = root.__flexDiffusionSparseCaptureTest
        const originalGetCurrentUserRequest = getCurrentUserRequest
        getCurrentUserRequest = function (...args) {
            const task = originalGetCurrentUserRequest.apply(this, args)
            if (activeContext && task?.reqBody) {
                Object.assign(task.reqBody, {
                    benchmark_schema: SCHEMA,
                    benchmark_name: activeContext.name,
                    benchmark_run_id: activeContext.runId,
                    benchmark_case_id: activeContext.caseId,
                    benchmark_case_label: activeContext.caseLabel,
                })
            }
            return task
        }

        if (typeof taskConfigSetup !== "undefined" && taskConfigSetup.taskConfig) {
            taskConfigSetup.taskConfig.benchmark_name = {
                label: "Benchmark",
                visible: ({ reqBody }) => !!reqBody?.benchmark_name,
            }
            taskConfigSetup.taskConfig.benchmark_case_id = {
                label: "Benchmark case",
                visible: ({ reqBody }) => !!reqBody?.benchmark_case_id,
            }
            taskConfigSetup.taskConfig.benchmark_run_id = {
                label: "Benchmark run",
                visible: ({ reqBody }) => !!reqBody?.benchmark_run_id,
            }
        }

        const dialog = document.createElement("dialog")
        dialog.id = "flex-benchmark-dialog"
        dialog.innerHTML = `
            <div style="min-width:min(760px,90vw);max-width:90vw">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">
                    <h3 style="margin:0">Benchmark run</h3>
                    <button type="button" id="flex-benchmark-close" class="tertiaryButton smallButton">Close</button>
                </div>
                <p>Paste a <code>${SCHEMA}</code> JSON block or load a JSON/text file. The complete run is validated before anything is queued.</p>
                <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
                    <label class="button tertiaryButton smallButton" style="position:relative;overflow:hidden">
                        Load file
                        <input id="flex-benchmark-file" type="file" accept=".json,.txt,application/json,text/plain" style="position:absolute;inset:0;opacity:0;cursor:pointer">
                    </label>
                    <button type="button" id="flex-benchmark-example" class="tertiaryButton smallButton">Load Cyberdino baseline</button>
                    <span id="flex-benchmark-file-name"><small></small></span>
                </div>
                <textarea id="flex-benchmark-text" spellcheck="false" style="width:100%;height:360px;box-sizing:border-box;font-family:monospace"></textarea>
                <div id="flex-benchmark-status" style="margin-top:8px;min-height:1.4em"></div>
                <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:10px">
                    <button type="button" id="flex-benchmark-validate" class="tertiaryButton">Validate</button>
                    <button type="button" id="flex-benchmark-queue" class="primaryButton">Queue benchmark</button>
                </div>
            </div>`
        document.body.appendChild(dialog)

        const openButton = document.createElement("button")
        openButton.type = "button"
        openButton.id = "flex-benchmark-open"
        openButton.className = "tertiaryButton"
        openButton.innerHTML = '<i class="fa-solid fa-flask"></i> Load benchmark…'
        makeImageButton.insertAdjacentElement("beforebegin", openButton)

        const textField = dialog.querySelector("#flex-benchmark-text")
        const fileField = dialog.querySelector("#flex-benchmark-file")
        const fileNameField = dialog.querySelector("#flex-benchmark-file-name small")
        const statusField = dialog.querySelector("#flex-benchmark-status")
        const queueButton = dialog.querySelector("#flex-benchmark-queue")
        textField.value = JSON.stringify(BUILT_IN_EXAMPLE, null, 2)

        function setStatus(message, error) {
            statusField.textContent = message
            statusField.style.color = error ? "var(--error-color, #b33)" : ""
        }

        function parseText() {
            const normalized = normalizeBenchmarkConfig(textField.value)
            normalized.cases.forEach((testCase) =>
                sparseHelper.parseCaptureExpression(
                    testCase.settings.capture_iterations,
                    testCase.settings.inference_steps
                )
            )
            return normalized
        }

        function summaryForNormalized(normalized) {
            let tasks = 0
            let outputs = 0
            normalized.cases.forEach((testCase) => {
                tasks += testCase.samplers.length
                const steps = sparseHelper.parseCaptureExpression(
                    testCase.settings.capture_iterations,
                    testCase.settings.inference_steps
                )
                const count = sparseHelper.outputCountSummary(
                    steps,
                    testCase.settings.inference_steps,
                    !!testCase.settings.endpoint_solver_states
                )
                outputs += count.totalOutputs * testCase.samplers.length * (testCase.settings.num_outputs || 1)
            })
            return { tasks, outputs }
        }

        function validateText() {
            try {
                const normalized = parseText()
                const summary = summaryForNormalized(normalized)
                setStatus(
                    `Valid — ${normalized.name}: ${normalized.cases.length} case${normalized.cases.length === 1 ? "" : "s"}, ` +
                        `${summary.tasks} sampler task${summary.tasks === 1 ? "" : "s"}, ~${summary.outputs} output image${summary.outputs === 1 ? "" : "s"}.`,
                    false
                )
                return normalized
            } catch (error) {
                setStatus(error.message || String(error), true)
                return null
            }
        }

        const ordinarySelectors = {
            prompt: "#prompt",
            negative_prompt: "#negative_prompt",
            seed: "#seed",
            width: "#width",
            height: "#height",
            inference_steps: "#num_inference_steps",
            guidance_scale: "#guidance_scale",
            output_format: "#output_format",
            output_quality: "#output_quality",
            capture_iterations: "#output_after_step",
            endpoint_solver_states: "#capture_endpoint_solver_states",
            num_outputs: "#num_outputs_total",
            num_outputs_parallel: "#num_outputs_parallel",
            clip_skip: "#clip_skip",
            scheduler: "#scheduler_name",
            vram_usage_level: "#vram_usage_level",
            stream_image_progress: "#stream_image_progress",
            block_nsfw: "#block_nsfw",
            show_only_filtered_image: "#show_only_filtered_image",
        }

        function fire(field) {
            field.dispatchEvent(new Event("input", { bubbles: true }))
            field.dispatchEvent(new Event("change", { bubbles: true }))
        }

        function modelValue(kind) {
            if (kind === "model" && typeof stableDiffusionModelField !== "undefined") return stableDiffusionModelField.value
            if (kind === "vae_model" && typeof vaeModelField !== "undefined") return vaeModelField.value
            const selector = kind === "model" ? "#stable_diffusion_model" : "#vae_model"
            return document.querySelector(selector)?.dataset?.path || ""
        }

        function setModelValue(kind, value) {
            if (value === undefined) return
            if (kind === "model" && typeof stableDiffusionModelField !== "undefined") {
                stableDiffusionModelField.value = String(value)
                stableDiffusionModelField.dispatchEvent(new Event("change", { bubbles: true }))
                return
            }
            if (kind === "vae_model" && typeof vaeModelField !== "undefined") {
                vaeModelField.value = String(value)
                vaeModelField.dispatchEvent(new Event("change", { bubbles: true }))
                return
            }
            const selector = kind === "model" ? "#stable_diffusion_model" : "#vae_model"
            const field = document.querySelector(selector)
            if (!field) throw new Error(`Benchmark setting requires unavailable UI field ${selector}`)
            field.dataset.path = String(value)
            field.value = String(value)
            fire(field)
        }

        function setOrdinaryValue(selector, value) {
            if (value === undefined) return
            const field = document.querySelector(selector)
            if (!field) throw new Error(`Benchmark setting requires unavailable UI field ${selector}`)
            if (field.type === "checkbox") field.checked = !!value
            else field.value = String(value)
            fire(field)
        }

        function applySettings(settings) {
            Object.entries(ordinarySelectors).forEach(([key, selector]) => setOrdinaryValue(selector, settings[key]))
            setModelValue("model", settings.model)
            setModelValue("vae_model", settings.vae_model ?? "")
        }

        function selectedExtraSamplers() {
            return Array.from(document.querySelectorAll('#sampler_compare_options input[type="checkbox"]:checked')).map(
                (input) => input.value
            )
        }

        function setSamplerSelection(samplers) {
            const requested = normalizeSamplers(samplers, "samplers")
            const field = document.querySelector("#sampler_name")
            const available = new Set(Array.from(field.options || []).map((option) => option.value).filter(Boolean))
            requested.forEach((name) => {
                if (!available.has(name)) throw new Error(`Sampler '${name}' is not available in the current backend`)
            })

            selectedExtraSamplers().forEach((name) => {
                const checkbox = Array.from(document.querySelectorAll('#sampler_compare_options input[type="checkbox"]')).find(
                    (item) => item.value === name
                )
                if (checkbox) {
                    checkbox.checked = false
                    checkbox.dispatchEvent(new Event("change", { bubbles: true }))
                }
            })

            field.value = requested[0]
            field.dispatchEvent(new Event("change", { bubbles: true }))
            requested.slice(1).forEach((name) => {
                const checkbox = Array.from(document.querySelectorAll('#sampler_compare_options input[type="checkbox"]')).find(
                    (item) => item.value === name
                )
                if (!checkbox) throw new Error(`Sampler '${name}' could not be selected for comparison`)
                checkbox.checked = true
                checkbox.dispatchEvent(new Event("change", { bubbles: true }))
            })
        }

        function snapshotUi() {
            const values = {}
            Object.entries(ordinarySelectors).forEach(([key, selector]) => {
                const field = document.querySelector(selector)
                if (!field) return
                values[key] = field.type === "checkbox" ? !!field.checked : field.value
            })
            return {
                values,
                model: modelValue("model"),
                vae_model: modelValue("vae_model"),
                sampler: document.querySelector("#sampler_name")?.value || "",
                extras: selectedExtraSamplers(),
            }
        }

        function restoreUi(snapshot) {
            Object.entries(ordinarySelectors).forEach(([key, selector]) => {
                if (snapshot.values[key] !== undefined) setOrdinaryValue(selector, snapshot.values[key])
            })
            setModelValue("model", snapshot.model)
            setModelValue("vae_model", snapshot.vae_model)
            if (snapshot.sampler) setSamplerSelection([snapshot.sampler, ...snapshot.extras])
        }

        function assertPlainTxt2ImgRequest(task, caseId) {
            const req = task?.reqBody || {}
            const forbidden = [
                "init_image",
                "mask",
                "control_image",
                "use_controlnet_model",
                "use_face_correction",
                "use_upscale",
                "use_hypernetwork_model",
                "use_lora_model",
                "ref_images",
                "active_tags",
                "inactive_tags",
            ].filter((key) => {
                const value = req[key]
                if (Array.isArray(value)) return value.length > 0
                return value !== undefined && value !== null && value !== false && value !== ""
            })
            if (forbidden.length) {
                throw new Error(
                    `Benchmark case '${caseId}' requires plain txt2img for v1; clear/disable current UI features: ${forbidden.join(", ")}`
                )
            }
            if (document.querySelector("#process_order_toggle")?.checked) {
                throw new Error("Benchmark runs require 'Process newest jobs first' to be OFF so sampler order remains reproducible")
            }
        }

        function makeRunId(name) {
            const slug = name
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, "-")
                .replace(/^-|-$/g, "")
                .slice(0, 48) || "benchmark"
            return `${slug}-${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}`
        }

        function queueBenchmark() {
            const normalized = validateText()
            if (!normalized) return
            if (typeof SD !== "undefined" && SD?.activeTasks?.size > 0) {
                setStatus("Benchmark not queued: wait for currently active generation tasks to finish first.", true)
                return
            }

            const snapshot = snapshotUi()
            const runId = makeRunId(normalized.name)
            let queued = 0
            queueButton.disabled = true
            try {
                // Preflight all cases before any call to makeImage().
                normalized.cases.forEach((testCase) => {
                    applySettings(testCase.settings)
                    setSamplerSelection(testCase.samplers)
                    assertPlainTxt2ImgRequest(getCurrentUserRequest(), testCase.id)
                })

                normalized.cases.forEach((testCase) => {
                    applySettings(testCase.settings)
                    setSamplerSelection(testCase.samplers)
                    activeContext = {
                        name: normalized.name,
                        runId,
                        caseId: testCase.id,
                        caseLabel: testCase.label,
                    }
                    makeImage()
                    activeContext = null
                    queued += testCase.samplers.length
                })
                setStatus(
                    `Queued '${normalized.name}' as ${runId}: ${queued} sampler task${queued === 1 ? "" : "s"}.`,
                    false
                )
            } catch (error) {
                activeContext = null
                setStatus(`Benchmark not queued: ${error.message || error}`, true)
            } finally {
                try {
                    restoreUi(snapshot)
                } catch (restoreError) {
                    console.error("FlexDiffusion benchmark loader could not fully restore UI", restoreError)
                }
                queueButton.disabled = false
            }
        }

        openButton.addEventListener("click", () => {
            validateText()
            dialog.showModal()
        })
        dialog.querySelector("#flex-benchmark-close").addEventListener("click", () => dialog.close())
        dialog.querySelector("#flex-benchmark-example").addEventListener("click", () => {
            textField.value = JSON.stringify(BUILT_IN_EXAMPLE, null, 2)
            fileNameField.textContent = "built-in baseline"
            validateText()
        })
        dialog.querySelector("#flex-benchmark-validate").addEventListener("click", validateText)
        queueButton.addEventListener("click", queueBenchmark)
        fileField.addEventListener("change", async () => {
            const file = fileField.files?.[0]
            if (!file) return
            try {
                textField.value = await file.text()
                fileNameField.textContent = file.name
                validateText()
            } catch (error) {
                setStatus(`Could not read benchmark file: ${error.message || error}`, true)
            }
        })
        textField.addEventListener("input", () => setStatus("Edited — validate before queueing.", false))
        dialog.addEventListener("click", (event) => {
            if (event.target === dialog) dialog.close()
        })
        validateText()
    }

    install()
})()
