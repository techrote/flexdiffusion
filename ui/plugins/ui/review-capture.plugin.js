;(function () {
    "use strict"

    const root = typeof window !== "undefined" ? window : globalThis
    const REVIEW_CAPTURE_VERSION = "1"
    const SOURCE_BRANCH_HINT = "trajectory-artifacts"
    const FEATURE_BASE_COMMIT = "4fcdb8670b4050883c0919f43b5cc20b301ba8e9"

    function pad2(value) {
        return String(value).padStart(2, "0")
    }

    function pad3(value) {
        return String(value).padStart(3, "0")
    }

    function makeTimestamp(date) {
        const d = date instanceof Date ? date : new Date()
        return (
            `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}` +
            `_${pad2(d.getHours())}-${pad2(d.getMinutes())}-${pad2(d.getSeconds())}`
        )
    }

    function scalar(value) {
        if (value === undefined || value === null || value === "") {
            return ""
        }
        if (Array.isArray(value)) {
            return value.join(", ")
        }
        if (typeof value === "object") {
            return JSON.stringify(value)
        }
        return String(value)
    }

    function markdownValue(value) {
        return scalar(value).replace(/\r?\n/g, " ↵ ")
    }

    function taskSummaryLines(task, index) {
        const req = task?.request || {}
        const outputs = task?.outputs || []
        const representations = Array.from(
            new Set(outputs.map((output) => output.intermediate_representation).filter(Boolean))
        )
        const outputSteps = Array.from(
            new Set(outputs.map((output) => output.output_step).filter((value) => value !== null && value !== undefined))
        )

        return [
            `### Task ${index + 1}`,
            `- Prompt: ${markdownValue(req.prompt ?? task?.prompt)}`,
            `- Negative prompt: ${markdownValue(req.negative_prompt)}`,
            `- Seed: ${markdownValue(req.seed ?? task?.seed)}`,
            `- Model: ${markdownValue(req.use_stable_diffusion_model)}`,
            `- Sampler: ${markdownValue(req.sampler_name)}`,
            `- Dimensions: ${markdownValue(req.width)}x${markdownValue(req.height)}`,
            `- Inference steps: ${markdownValue(req.num_inference_steps)}`,
            `- Output after step: ${markdownValue(req.output_after_step)}`,
            `- Intermediate view: ${markdownValue(req.intermediate_representation)}`,
            `- CFG: ${markdownValue(req.guidance_scale)}`,
            `- VRAM mode: ${markdownValue(req.vram_usage_level)}`,
            `- Output steps present: ${markdownValue(outputSteps)}`,
            `- Representations present: ${markdownValue(representations)}`,
            `- UI status: ${markdownValue(task?.status)}`,
            `- Timing/result text: ${markdownValue(task?.output_message)}`,
            "",
        ]
    }

    function buildReviewMarkdown(metadata, screenshotName) {
        const tasks = Array.isArray(metadata?.tasks) ? metadata.tasks : []
        const capture = metadata?.capture || {}
        const app = metadata?.app || {}
        const ui = metadata?.ui_state || {}
        const lines = [
            "# FlexDiffusion Review Capture",
            "",
            "## Review",
            "- Rating (0-10):",
            "- Keep / Reject / Investigate:",
            "- Preferred sampler:",
            "- Comments:",
            "- Notable artifacts / useful edge cases:",
            "- Follow-up ideas:",
            "",
            "## Capture",
            `- Timestamp: ${markdownValue(capture.local_time)}`,
            `- Screenshot: ${markdownValue(screenshotName)}`,
            `- Task count: ${tasks.length}`,
            `- FlexDiffusion review-capture version: ${REVIEW_CAPTURE_VERSION}`,
            `- Source branch hint: ${markdownValue(app.source_branch_hint || SOURCE_BRANCH_HINT)}`,
            `- Feature base commit: ${markdownValue(app.feature_base_commit || FEATURE_BASE_COMMIT)}`,
            `- Easy Diffusion UI version: ${markdownValue(app.ui_version)}`,
            `- Backend: ${markdownValue(app.backend)}`,
            `- Config update branch: ${markdownValue(app.update_branch)}`,
            `- Current UI sampler: ${markdownValue(ui.primary_sampler)}`,
            `- Additional selected samplers: ${markdownValue(ui.additional_samplers)}`,
            `- Explicit seed: ${markdownValue(ui.seed)}`,
            "",
            "## Tasks",
            "",
        ]

        tasks.forEach((task, index) => lines.push(...taskSummaryLines(task, index)))

        lines.push(
            "## Notes on interpretation",
            "",
            "The raw JSON below is the authoritative machine-readable capture. Large embedded image/data strings are replaced by SHA-256/length descriptors so the sidecar remains editable while retaining provenance.",
            "",
            "## Raw Metadata",
            "",
            "```json",
            JSON.stringify(metadata, null, 2),
            "```",
            ""
        )
        return lines.join("\n")
    }

    root.__flexDiffusionReviewCaptureTest = {
        makeTimestamp,
        buildReviewMarkdown,
        taskSummaryLines,
    }

    if (typeof document === "undefined") {
        return
    }
    if (root.__flexDiffusionReviewCaptureInstalled) {
        return
    }

    const COMPUTED_STYLE_PROPERTIES = [
        "display",
        "position",
        "box-sizing",
        "width",
        "height",
        "min-width",
        "max-width",
        "min-height",
        "max-height",
        "margin",
        "padding",
        "border",
        "border-top",
        "border-right",
        "border-bottom",
        "border-left",
        "border-radius",
        "background",
        "background-color",
        "background-image",
        "background-size",
        "background-position",
        "color",
        "font-family",
        "font-size",
        "font-weight",
        "font-style",
        "line-height",
        "letter-spacing",
        "text-align",
        "text-decoration",
        "white-space",
        "word-break",
        "overflow",
        "overflow-x",
        "overflow-y",
        "vertical-align",
        "opacity",
        "transform",
        "transform-origin",
        "flex",
        "flex-basis",
        "flex-grow",
        "flex-shrink",
        "flex-direction",
        "flex-wrap",
        "justify-content",
        "align-items",
        "align-content",
        "gap",
        "row-gap",
        "column-gap",
        "grid-template-columns",
        "grid-template-rows",
        "grid-auto-flow",
        "object-fit",
        "object-position",
        "list-style",
        "visibility",
    ]

    function visibleTaskEntries() {
        return Array.from(document.querySelectorAll(".imageTaskContainer")).filter((element) => {
            const style = root.getComputedStyle(element)
            return style.display !== "none" && style.visibility !== "hidden"
        })
    }

    function selectedAdditionalSamplers() {
        return Array.from(document.querySelectorAll('#sampler_compare_options input[type="checkbox"]:checked'))
            .map((input) => input.value)
            .filter(Boolean)
    }

    function taskObjectFor(element) {
        if (typeof htmlTaskMap === "undefined" || !htmlTaskMap?.get) {
            return null
        }
        return htmlTaskMap.get(element) || null
    }

    function imageRequestFor(img) {
        const counter = img?.dataset?.imagecounter
        if (!counter || typeof imageRequest === "undefined") {
            return null
        }
        return imageRequest[counter] || null
    }

    function imageMetadata(img, index) {
        const request = imageRequestFor(img) || {}
        return {
            index,
            seed: request.seed ?? img.dataset.seed ?? null,
            output_step: request.output_step ?? parseOptionalInt(img.dataset.outputStep),
            total_steps: request.total_steps ?? parseOptionalInt(img.dataset.totalSteps),
            intermediate_representation:
                request.intermediate_representation ?? img.dataset.intermediateRepresentation ?? null,
            width: img.naturalWidth || parseOptionalInt(img.getAttribute("width")),
            height: img.naturalHeight || parseOptionalInt(img.getAttribute("height")),
            source_kind: img.src?.startsWith("data:") ? "embedded-data-url" : img.src?.startsWith("blob:") ? "blob-url" : "url",
            source_url: img.src?.startsWith("data:") || img.src?.startsWith("blob:") ? null : img.src || null,
        }
    }

    function parseOptionalInt(value) {
        if (value === undefined || value === null || value === "") {
            return null
        }
        const parsed = parseInt(value)
        return Number.isFinite(parsed) ? parsed : null
    }

    function parseTaskCreatedAt(element) {
        const match = String(element.id || "").match(/imageTaskContainer-(\d+)/)
        if (!match) {
            return null
        }
        const millis = Number(match[1])
        if (!Number.isFinite(millis)) {
            return null
        }
        return new Date(millis).toISOString()
    }

    function collectRawTaskMetadata(element, index) {
        const task = taskObjectFor(element)
        const reqBody = task?.reqBody ? { ...task.reqBody } : {}
        const imgs = Array.from(element.querySelectorAll(".imgItem img")).filter((img) => {
            const item = img.closest(".imgItem")
            return item && item.style.display !== "none"
        })

        return {
            index,
            dom_id: element.id || null,
            created_at: parseTaskCreatedAt(element),
            prompt: task?.previewPrompt?.textContent?.trim() || reqBody.prompt || element.querySelector(".preview-prompt")?.textContent?.trim() || "",
            seed: task?.seed ?? reqBody.seed ?? null,
            status: task?.taskStatusLabel?.textContent?.trim() || element.querySelector(".taskStatusLabel")?.textContent?.trim() || null,
            output_message: task?.outputMsg?.textContent?.trim() || element.querySelector(".outputMsg")?.textContent?.trim() || null,
            batches_done: task?.batchesDone ?? null,
            batch_count: task?.batchCount ?? null,
            num_outputs_total: task?.numOutputsTotal ?? null,
            request: reqBody,
            outputs: imgs.map(imageMetadata),
        }
    }

    async function sha256Text(text) {
        if (!root.crypto?.subtle || typeof TextEncoder === "undefined") {
            return null
        }
        const bytes = new TextEncoder().encode(text)
        const digest = await root.crypto.subtle.digest("SHA-256", bytes)
        return Array.from(new Uint8Array(digest))
            .map((byte) => byte.toString(16).padStart(2, "0"))
            .join("")
    }

    async function sanitizeMetadata(value, seen) {
        if (value === null || value === undefined || typeof value === "number" || typeof value === "boolean") {
            return value
        }
        if (typeof value === "string") {
            if (value.startsWith("data:") || value.length > 4096) {
                return {
                    omitted_long_value: true,
                    length: value.length,
                    sha256: await sha256Text(value),
                    prefix: value.slice(0, 96),
                }
            }
            return value
        }
        if (typeof value !== "object") {
            return String(value)
        }

        const visited = seen || new WeakSet()
        if (visited.has(value)) {
            return "[circular]"
        }
        visited.add(value)

        if (Array.isArray(value)) {
            const result = []
            for (const item of value) {
                result.push(await sanitizeMetadata(item, visited))
            }
            return result
        }

        const result = {}
        for (const [key, item] of Object.entries(value)) {
            if (typeof item === "function") {
                continue
            }
            result[key] = await sanitizeMetadata(item, visited)
        }
        return result
    }

    async function fetchAppConfig() {
        try {
            const response = await fetch("/get/app_config")
            if (!response.ok) {
                return { error: `HTTP ${response.status}` }
            }
            return await response.json()
        } catch (error) {
            return { error: `${error?.name || "Error"}: ${error?.message || error}` }
        }
    }

    async function collectMetadata(taskEntries, timestamp) {
        const rawTasks = taskEntries.map(collectRawTaskMetadata)
        const appConfig = await fetchAppConfig()
        const now = new Date()
        const metadata = {
            schema: "flexdiffusion-review-capture/v1",
            capture: {
                basename: timestamp,
                local_time: now.toLocaleString(),
                iso_time: now.toISOString(),
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || null,
                timezone_offset_minutes: now.getTimezoneOffset(),
            },
            app: {
                repository: "techrote/flexdiffusion",
                source_branch_hint: SOURCE_BRANCH_HINT,
                feature_base_commit: FEATURE_BASE_COMMIT,
                review_capture_version: REVIEW_CAPTURE_VERSION,
                ui_version: document.querySelector("#version")?.textContent?.replace(/\s+/g, " ").trim() || null,
                update_branch_label: document.querySelector("#updateBranchLabel")?.textContent?.trim() || null,
                backend: appConfig?.backend ?? document.querySelector("#backend")?.value ?? null,
                update_branch: appConfig?.update_branch ?? null,
                app_config: appConfig,
            },
            environment: {
                user_agent: navigator.userAgent,
                platform: navigator.platform || null,
                language: navigator.language || null,
                hardware_concurrency: navigator.hardwareConcurrency ?? null,
                device_memory_gb_hint: navigator.deviceMemory ?? null,
                screen: { width: screen.width, height: screen.height, pixel_ratio: root.devicePixelRatio || 1 },
                viewport: { width: root.innerWidth, height: root.innerHeight },
            },
            ui_state: {
                primary_sampler: document.querySelector("#sampler_name")?.value ?? null,
                additional_samplers: selectedAdditionalSamplers(),
                seed: document.querySelector("#seed")?.value ?? null,
                random_seed_enabled: !!document.querySelector("#random_seed")?.checked,
                inference_steps: parseOptionalInt(document.querySelector("#num_inference_steps")?.value),
                output_after_step: parseOptionalInt(document.querySelector("#output_after_step")?.value),
                intermediate_representation: document.querySelector("#intermediate_representation")?.value ?? null,
                model: document.querySelector("#stable_diffusion_model")?.value ?? null,
                cfg: document.querySelector("#guidance_scale")?.value ?? null,
                output_format: document.querySelector("#output_format")?.value ?? null,
            },
            tasks: rawTasks,
        }
        return await sanitizeMetadata(metadata)
    }

    function copyComputedStyles(source, clone) {
        const sourceElements = [source, ...source.querySelectorAll("*")]
        const cloneElements = [clone, ...clone.querySelectorAll("*")]
        const count = Math.min(sourceElements.length, cloneElements.length)

        for (let index = 0; index < count; index++) {
            const sourceElement = sourceElements[index]
            const cloneElement = cloneElements[index]
            const computed = root.getComputedStyle(sourceElement)
            COMPUTED_STYLE_PROPERTIES.forEach((property) => {
                const value = computed.getPropertyValue(property)
                if (value) {
                    cloneElement.style.setProperty(property, value)
                }
            })
        }
    }

    function blobToDataUrl(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader()
            reader.onload = () => resolve(reader.result)
            reader.onerror = reject
            reader.readAsDataURL(blob)
        })
    }

    async function resolveImageToDataUrl(src) {
        if (!src || src.startsWith("data:")) {
            return src
        }
        try {
            const response = await fetch(src)
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`)
            }
            return await blobToDataUrl(await response.blob())
        } catch (error) {
            console.warn("FlexDiffusion review capture could not inline image", src, error)
            return src
        }
    }

    async function cloneTaskForScreenshot(source) {
        const clone = source.cloneNode(true)
        copyComputedStyles(source, clone)

        clone.querySelectorAll(".stopTask, .useSettings, .imgPreviewItemClearBtn, .tasksBtns, .spinner, .simple-tooltip").forEach((element) => element.remove())
        clone.querySelectorAll(".collapsible-content").forEach((element) => {
            element.style.display = "block"
            element.style.maxHeight = "none"
            element.style.height = "auto"
            element.style.overflow = "visible"
        })
        clone.querySelectorAll(".header-content").forEach((element) => {
            element.classList.add("active")
        })

        const sourceImages = Array.from(source.querySelectorAll("img"))
        const cloneImages = Array.from(clone.querySelectorAll("img"))
        await Promise.all(
            cloneImages.map(async (img, index) => {
                const original = sourceImages[index]
                if (!original) {
                    return
                }
                img.src = await resolveImageToDataUrl(original.currentSrc || original.src)
                img.removeAttribute("crossorigin")
            })
        )

        return clone
    }

    function nextFrame() {
        return new Promise((resolve) => root.requestAnimationFrame(() => root.requestAnimationFrame(resolve)))
    }

    async function waitForImages(element) {
        const images = Array.from(element.querySelectorAll("img"))
        await Promise.all(
            images.map((img) => {
                if (img.complete) {
                    return img.decode?.().catch(() => undefined) || Promise.resolve()
                }
                return new Promise((resolve) => {
                    img.addEventListener("load", resolve, { once: true })
                    img.addEventListener("error", resolve, { once: true })
                })
            })
        )
    }

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image()
            img.onload = () => resolve(img)
            img.onerror = reject
            img.src = src
        })
    }

    function canvasToBlob(canvas) {
        return new Promise((resolve, reject) => {
            canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Canvas PNG encoding failed"))), "image/png")
        })
    }

    async function renderForeignObjectPng(taskEntries) {
        const rectWidths = taskEntries.map((entry) => Math.ceil(entry.getBoundingClientRect().width || entry.scrollWidth || 1200))
        const exportWidth = Math.max(640, ...rectWidths)
        const exportRoot = document.createElement("div")
        exportRoot.setAttribute("xmlns", "http://www.w3.org/1999/xhtml")
        exportRoot.style.position = "fixed"
        exportRoot.style.left = "-100000px"
        exportRoot.style.top = "0"
        exportRoot.style.width = `${exportWidth}px`
        exportRoot.style.boxSizing = "border-box"
        exportRoot.style.padding = "12px"
        exportRoot.style.background = root.getComputedStyle(document.body).backgroundColor || "#161616"
        exportRoot.style.color = root.getComputedStyle(document.body).color || "#dddddd"
        exportRoot.style.fontFamily = root.getComputedStyle(document.body).fontFamily || "sans-serif"
        exportRoot.style.display = "block"
        exportRoot.style.visibility = "hidden"
        exportRoot.style.zIndex = "-2147483648"

        try {
            for (const entry of taskEntries) {
                const clone = await cloneTaskForScreenshot(entry)
                clone.style.width = "100%"
                clone.style.maxWidth = "none"
                clone.style.boxSizing = "border-box"
                clone.style.marginBottom = "12px"
                exportRoot.appendChild(clone)
            }
            document.body.appendChild(exportRoot)
            await waitForImages(exportRoot)
            await nextFrame()

            const width = Math.ceil(exportRoot.scrollWidth)
            const height = Math.ceil(exportRoot.scrollHeight)
            if (width < 1 || height < 1) {
                throw new Error("Nothing measurable to capture")
            }

            // Keep exports below common browser canvas limits while preserving the
            // complete vertical review sheet.
            const maxDimension = 16384
            const maxPixels = 80_000_000
            const scale = Math.min(1, maxDimension / width, maxDimension / height, Math.sqrt(maxPixels / (width * height)))
            const serializer = new XMLSerializer()
            // The live staging node is kept far off-screen and hidden while its
            // layout is measured. Do not serialize those staging-only styles or
            // the foreignObject would faithfully render an invisible sheet.
            const stagingCss = exportRoot.style.cssText
            exportRoot.style.position = "relative"
            exportRoot.style.left = "0"
            exportRoot.style.top = "0"
            exportRoot.style.visibility = "visible"
            exportRoot.style.zIndex = "auto"
            const xhtml = serializer.serializeToString(exportRoot)
            exportRoot.style.cssText = stagingCss
            const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><foreignObject x="0" y="0" width="100%" height="100%">${xhtml}</foreignObject></svg>`
            const svgUrl = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }))
            try {
                const rendered = await loadImage(svgUrl)
                const canvas = document.createElement("canvas")
                canvas.width = Math.max(1, Math.floor(width * scale))
                canvas.height = Math.max(1, Math.floor(height * scale))
                const ctx = canvas.getContext("2d")
                ctx.drawImage(rendered, 0, 0, canvas.width, canvas.height)
                return await canvasToBlob(canvas)
            } finally {
                URL.revokeObjectURL(svgUrl)
            }
        } finally {
            exportRoot.remove()
        }
    }

    function wrapText(ctx, text, maxWidth) {
        const words = String(text || "").split(/\s+/)
        const lines = []
        let line = ""
        words.forEach((word) => {
            const candidate = line ? `${line} ${word}` : word
            if (line && ctx.measureText(candidate).width > maxWidth) {
                lines.push(line)
                line = word
            } else {
                line = candidate
            }
        })
        if (line) {
            lines.push(line)
        }
        return lines.length ? lines : [""]
    }

    async function fallbackReviewSheetPng(taskEntries, metadata) {
        const width = 1600
        const padding = 30
        const taskGap = 32
        const imageGap = 12
        const imageMaxWidth = 360
        const imageMaxHeight = 360
        const tasks = metadata.tasks || []

        const loadedTaskImages = []
        for (const entry of taskEntries) {
            const imgs = Array.from(entry.querySelectorAll(".imgItem img")).filter((img) => img.closest(".imgItem")?.style.display !== "none")
            const loaded = []
            for (const img of imgs) {
                try {
                    loaded.push(await loadImage(await resolveImageToDataUrl(img.currentSrc || img.src)))
                } catch (error) {
                    console.warn("FlexDiffusion review capture fallback skipped image", error)
                }
            }
            loadedTaskImages.push(loaded)
        }

        let estimatedHeight = 100
        tasks.forEach((task, index) => {
            const promptLines = Math.max(1, Math.ceil(String(task.request?.prompt || task.prompt || "").length / 95))
            const rows = Math.max(1, Math.ceil((loadedTaskImages[index]?.length || 0) / 4))
            estimatedHeight += 150 + promptLines * 22 + rows * (imageMaxHeight + imageGap) + taskGap
        })
        const height = Math.min(16384, Math.max(600, estimatedHeight))
        const canvas = document.createElement("canvas")
        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext("2d")
        ctx.fillStyle = "#171717"
        ctx.fillRect(0, 0, width, height)
        ctx.fillStyle = "#eeeeee"
        ctx.font = "bold 28px sans-serif"
        ctx.fillText("FlexDiffusion Review Capture", padding, 45)
        ctx.font = "16px sans-serif"
        ctx.fillStyle = "#aaaaaa"
        ctx.fillText(metadata.capture?.local_time || "", padding, 72)

        let y = 110
        for (let index = 0; index < tasks.length && y < height - 50; index++) {
            const task = tasks[index]
            const req = task.request || {}
            ctx.fillStyle = "#272727"
            ctx.fillRect(padding, y, width - padding * 2, 115)
            ctx.fillStyle = "#ffffff"
            ctx.font = "bold 20px sans-serif"
            ctx.fillText(`Task ${index + 1}: ${req.sampler_name || "unknown sampler"}`, padding + 15, y + 28)
            ctx.font = "15px sans-serif"
            ctx.fillStyle = "#d0d0d0"
            const summary = `Seed ${req.seed ?? ""} · ${req.width ?? ""}x${req.height ?? ""} · ${req.num_inference_steps ?? ""} steps · CFG ${req.guidance_scale ?? ""} · ${req.use_stable_diffusion_model || ""}`
            ctx.fillText(summary, padding + 15, y + 53)
            const promptLines = wrapText(ctx, req.prompt || task.prompt || "", width - padding * 2 - 30).slice(0, 2)
            promptLines.forEach((line, lineIndex) => ctx.fillText(line, padding + 15, y + 78 + lineIndex * 20))
            y += 130

            const images = loadedTaskImages[index] || []
            for (let imageIndex = 0; imageIndex < images.length; imageIndex++) {
                const img = images[imageIndex]
                const column = imageIndex % 4
                const row = Math.floor(imageIndex / 4)
                const x = padding + column * (imageMaxWidth + imageGap)
                const top = y + row * (imageMaxHeight + 44 + imageGap)
                const ratio = Math.min(imageMaxWidth / img.naturalWidth, imageMaxHeight / img.naturalHeight, 1)
                const drawWidth = Math.max(1, Math.round(img.naturalWidth * ratio))
                const drawHeight = Math.max(1, Math.round(img.naturalHeight * ratio))
                ctx.drawImage(img, x, top, drawWidth, drawHeight)
                const out = task.outputs?.[imageIndex] || {}
                const labelParts = []
                if (out.output_step !== null && out.output_step !== undefined) {
                    labelParts.push(`Step ${out.output_step}/${out.total_steps ?? req.num_inference_steps ?? "?"}`)
                }
                if (out.intermediate_representation) {
                    labelParts.push(out.intermediate_representation)
                }
                ctx.fillStyle = "#d8d8d8"
                ctx.font = "14px sans-serif"
                ctx.fillText(labelParts.join(" · ") || `Output ${imageIndex + 1}`, x, top + drawHeight + 20)
            }
            const rows = Math.max(1, Math.ceil(images.length / 4))
            y += rows * (imageMaxHeight + 44 + imageGap) + taskGap
        }
        return await canvasToBlob(canvas)
    }

    function saveBlob(blob, filename) {
        if (typeof saveAs === "function") {
            saveAs(blob, filename)
            return
        }
        const url = URL.createObjectURL(blob)
        const anchor = document.createElement("a")
        anchor.href = url
        anchor.download = filename
        document.body.appendChild(anchor)
        anchor.click()
        anchor.remove()
        root.setTimeout(() => URL.revokeObjectURL(url), 1000)
    }

    async function captureReview(button) {
        const tasks = visibleTaskEntries()
        if (tasks.length === 0) {
            alert("There are no visible FlexDiffusion result tasks to capture.")
            return
        }

        const timestamp = makeTimestamp(new Date())
        const pngName = `${timestamp}.png`
        const mdName = `${timestamp}.md`
        const originalHtml = button.innerHTML
        button.disabled = true
        button.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span> Capturing…</span>'

        try {
            const metadata = await collectMetadata(tasks, timestamp)
            let pngBlob
            try {
                pngBlob = await renderForeignObjectPng(tasks)
            } catch (error) {
                console.warn("FlexDiffusion DOM screenshot failed; using review-sheet fallback", error)
                pngBlob = await fallbackReviewSheetPng(tasks, metadata)
                metadata.capture.screenshot_mode = "review-sheet-fallback"
                metadata.capture.screenshot_error = `${error?.name || "Error"}: ${error?.message || error}`
            }
            if (!metadata.capture.screenshot_mode) {
                metadata.capture.screenshot_mode = "dom-screenshot"
            }

            const markdown = buildReviewMarkdown(metadata, pngName)
            saveBlob(pngBlob, pngName)
            // A slight separation avoids some browsers collapsing two synthetic
            // downloads into one event while keeping matching timestamp names.
            root.setTimeout(() => saveBlob(new Blob([markdown], { type: "text/markdown;charset=utf-8" }), mdName), 120)
        } catch (error) {
            console.error("FlexDiffusion review capture failed", error)
            alert(`Review capture failed: ${error?.message || error}`)
        } finally {
            button.disabled = false
            button.innerHTML = originalHtml
        }
    }

    function install() {
        const downloadButton = document.querySelector("#show-download-popup")
        const previewTools = document.querySelector("#preview-tools")
        if (!downloadButton || !previewTools) {
            root.setTimeout(install, 50)
            return
        }
        if (root.__flexDiffusionReviewCaptureInstalled) {
            return
        }
        root.__flexDiffusionReviewCaptureInstalled = true

        const button = document.createElement("button")
        button.id = "flex-review-capture"
        button.className = "tertiaryButton"
        button.title = "Save a PNG review screenshot and matching editable Markdown metadata sidecar"
        button.innerHTML = '<i class="fa-solid fa-camera"></i><span> Review capture</span>'
        downloadButton.insertAdjacentElement("afterend", button)
        button.addEventListener("click", () => captureReview(button))
    }

    install()
})()
