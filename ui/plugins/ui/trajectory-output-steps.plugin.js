;(function () {
    "use strict"

    const root = typeof window !== "undefined" ? window : globalThis
    const SPARSE_PREFIX = "sparse:"

    const REPRESENTATION_LABELS = {
        denoised: "Denoised estimate",
        solver_state: "Solver state",
        final: "Final",
    }

    function validateTotalSteps(totalSteps) {
        const total = Number(totalSteps)
        if (!Number.isInteger(total) || total < 1) {
            throw new Error("Inference Steps must be a positive integer")
        }
        return total
    }

    function parseCaptureExpression(expression, totalSteps) {
        const total = validateTotalSteps(totalSteps)
        const text = String(expression ?? "").trim()
        if (!text) {
            throw new Error("Enter at least one capture iteration")
        }

        const selected = new Set()
        text.split(",").forEach((rawToken) => {
            const token = rawToken.trim()
            if (!token) {
                throw new Error("Capture iteration list contains an empty item")
            }

            let start
            let end
            const single = token.match(/^\d+$/)
            const range = token.match(/^(\d+)\s*-\s*(\d+)$/)
            if (single) {
                start = end = Number(token)
            } else if (range) {
                start = Number(range[1])
                end = Number(range[2])
                if (start > end) {
                    throw new Error(`Capture range must be ascending: ${token}`)
                }
            } else {
                throw new Error(`Invalid capture iteration item: ${token}`)
            }

            if (start < 1 || end > total) {
                throw new Error(`Capture iteration ${token} is outside 1-${total}`)
            }
            for (let step = start; step <= end; step += 1) {
                selected.add(step)
            }
        })

        return Array.from(selected).sort((a, b) => a - b)
    }

    function formatCaptureSteps(steps) {
        const values = Array.from(new Set((steps || []).map(Number))).sort((a, b) => a - b)
        if (!values.length) {
            return ""
        }

        const groups = []
        let start = values[0]
        let previous = values[0]
        for (const step of values.slice(1)) {
            if (step === previous + 1) {
                previous = step
                continue
            }
            groups.push(start === previous ? `${start}` : `${start}-${previous}`)
            start = previous = step
        }
        groups.push(start === previous ? `${start}` : `${start}-${previous}`)
        return groups.join(", ")
    }

    function endpointSolverSteps(steps, totalSteps, enabled) {
        if (!enabled || !steps?.length) {
            return []
        }
        const total = validateTotalSteps(totalSteps)
        const selected = new Set()
        const first = steps[0]
        const last = steps[steps.length - 1]
        if (first < total) {
            selected.add(first)
        }
        // k-diffusion does not expose a callback boundary after the final
        // integration update. If the last requested step is the actual final
        // step, the ordinary sampler return is authoritative instead.
        if (last < total) {
            selected.add(last)
        }
        return Array.from(selected).sort((a, b) => a - b)
    }

    function encodeSparseTransport(steps, totalSteps, includeEndpointSolverStates) {
        const canonical = formatCaptureSteps(steps).replace(/\s+/g, "")
        // totalSteps is validated here so malformed UI state cannot be encoded.
        validateTotalSteps(totalSteps)
        return `${SPARSE_PREFIX}${canonical}|endpoints=${includeEndpointSolverStates ? 1 : 0}`
    }

    function decodeSparseTransport(value, totalSteps) {
        const match = String(value ?? "").trim().toLowerCase().match(/^sparse:(.+)\|endpoints=([01])$/)
        if (!match) {
            return null
        }
        return {
            steps: parseCaptureExpression(match[1], totalSteps),
            includeEndpointSolverStates: match[2] === "1",
        }
    }

    function outputCountSummary(steps, totalSteps, includeEndpointSolverStates) {
        const total = validateTotalSteps(totalSteps)
        const denoisedCount = (steps || []).filter((step) => step < total).length
        const solverCount = endpointSolverSteps(steps, total, includeEndpointSolverStates).length
        const totalOutputs = denoisedCount + solverCount + 1
        return {
            denoisedCount,
            solverCount,
            finalCount: 1,
            totalOutputs,
        }
    }

    root.__flexDiffusionSparseCaptureTest = {
        parseCaptureExpression,
        formatCaptureSteps,
        endpointSolverSteps,
        encodeSparseTransport,
        decodeSparseTransport,
        outputCountSummary,
    }

    if (typeof document === "undefined") {
        return
    }
    if (root.__flexDiffusionOutputAfterStepInstalled) {
        return
    }

    function install() {
        const inferenceStepsField = document.querySelector("#num_inference_steps")
        if (
            !inferenceStepsField ||
            typeof getCurrentUserRequest !== "function" ||
            typeof showImages !== "function"
        ) {
            root.setTimeout(install, 50)
            return
        }

        if (root.__flexDiffusionOutputAfterStepInstalled) {
            return
        }
        root.__flexDiffusionOutputAfterStepInstalled = true

        const inferenceRow = inferenceStepsField.closest("tr")
        if (!inferenceRow) {
            console.error("FlexDiffusion sparse capture: could not find Inference Steps row")
            return
        }

        let captureField = document.querySelector("#output_after_step")
        let captureRow = captureField?.closest("tr")
        if (!captureField) {
            captureRow = document.createElement("tr")
            captureRow.id = "output_after_step_row"
            captureRow.className = "pl-5"
            captureRow.innerHTML = `
                <td>
                    <label for="output_after_step">Capture iterations:</label>
                </td>
                <td>
                    <input
                        id="output_after_step"
                        name="output_after_step"
                        type="text"
                        style="width: 150pt"
                        value="${inferenceStepsField.value}"
                        placeholder="10, 16, 20-24"
                        spellcheck="false"
                        autocomplete="off"
                    >
                    <i class="fa-solid fa-circle-question help-btn">
                        <span class="simple-tooltip top-left">
                            Printer-style syntax: 10, 16, 20-24 captures denoised estimates at
                            10, 16, 20, 21, 22, 23 and 24. The final sampler image is always included.
                            MVP: classic txt2img fixed-step k-diffusion samplers.
                        </span>
                    </i>
                    <br><small id="capture_iteration_summary"></small>
                </td>
            `
            inferenceRow.insertAdjacentElement("afterend", captureRow)
            captureField = captureRow.querySelector("#output_after_step")
        } else {
            captureField.type = "text"
            captureField.removeAttribute("min")
            captureField.removeAttribute("step")
            captureField.removeAttribute("onkeypress")
            captureField.style.width = "150pt"
        }

        // Remove the old representation selector if a stale/hot-reloaded DOM
        // already contains it. Sparse capture is denoised-first by design.
        document.querySelector("#intermediate_representation_row")?.remove()

        const endpointRow = document.createElement("tr")
        endpointRow.id = "capture_endpoint_solver_states_row"
        endpointRow.className = "pl-5"
        endpointRow.innerHTML = `
            <td><label for="capture_endpoint_solver_states">Solver endpoints:</label></td>
            <td>
                <input id="capture_endpoint_solver_states" name="capture_endpoint_solver_states" type="checkbox">
                <label for="capture_endpoint_solver_states">Include first/last solver states</label>
                <i class="fa-solid fa-circle-question help-btn">
                    <span class="simple-tooltip top-left">
                        Adds the raw noisy solver state at the first and last requested callback boundaries.
                        If the last requested iteration is the true final step, the sampler's ordinary final image
                        is used instead because k-diffusion has no post-final callback boundary.
                    </span>
                </i>
            </td>
        `
        captureRow.insertAdjacentElement("afterend", endpointRow)

        const endpointField = endpointRow.querySelector("#capture_endpoint_solver_states")
        const summaryField = captureRow.querySelector("#capture_iteration_summary")
        const backendField = document.querySelector("#backend")
        const initImagePreviewContainer = document.querySelector("#init_image_preview_container")
        let linkedToInferenceSteps = true

        function totalSteps() {
            return Math.max(1, parseInt(inferenceStepsField.value) || 1)
        }

        function isClassicBackend() {
            return backendField ? backendField.value === "ed_classic" : true
        }

        function isImg2Img() {
            return !!initImagePreviewContainer?.classList.contains("has-image")
        }

        function mvpAvailable() {
            return isClassicBackend() && !isImg2Img()
        }

        function parseCurrentCaptureSteps(reportValidity) {
            try {
                const steps = parseCaptureExpression(captureField.value, totalSteps())
                captureField.setCustomValidity("")
                return steps
            } catch (error) {
                captureField.setCustomValidity(error.message || String(error))
                if (reportValidity) {
                    captureField.reportValidity()
                }
                return null
            }
        }

        function updateSummary() {
            const steps = parseCurrentCaptureSteps(false)
            if (!steps) {
                summaryField.textContent = captureField.validationMessage
                summaryField.style.color = "var(--error-color, #b33)"
                return
            }
            summaryField.style.color = ""
            const count = outputCountSummary(steps, totalSteps(), endpointField.checked)
            const pieces = []
            if (count.denoisedCount) {
                pieces.push(`${count.denoisedCount} denoised capture${count.denoisedCount === 1 ? "" : "s"}`)
            }
            if (count.solverCount) {
                pieces.push(`${count.solverCount} endpoint solver state${count.solverCount === 1 ? "" : "s"}`)
            }
            pieces.push("final")
            summaryField.textContent = `${pieces.join(" + ")} = ${count.totalOutputs} output${count.totalOutputs === 1 ? "" : "s"} / sampler`
        }

        function updateVisibility() {
            const available = mvpAvailable()
            captureRow.style.display = available ? "" : "none"
            captureField.disabled = !available

            const steps = available ? parseCurrentCaptureSteps(false) : null
            const hasIntermediate = !!steps?.some((step) => step < totalSteps())
            endpointRow.style.display = available && hasIntermediate ? "" : "none"
            endpointField.disabled = !available || !hasIntermediate
            updateSummary()
        }

        inferenceStepsField.addEventListener("input", function () {
            if (linkedToInferenceSteps) {
                captureField.value = String(totalSteps())
            }
            updateVisibility()
        })

        captureField.addEventListener("input", function () {
            linkedToInferenceSteps = captureField.value.trim() === String(totalSteps())
            updateVisibility()
        })

        captureField.addEventListener("change", function () {
            const steps = parseCurrentCaptureSteps(true)
            if (steps) {
                captureField.value = formatCaptureSteps(steps)
                linkedToInferenceSteps = steps.length === 1 && steps[0] === totalSteps()
            }
            updateVisibility()
        })

        captureField.addEventListener("blur", function () {
            captureField.dispatchEvent(new Event("change"))
        })
        endpointField.addEventListener("change", updateSummary)

        if (backendField) {
            backendField.addEventListener("change", updateVisibility)
        }

        if (initImagePreviewContainer) {
            new MutationObserver(updateVisibility).observe(initImagePreviewContainer, {
                attributes: true,
                attributeFilter: ["class"],
            })
        }

        new MutationObserver(updateVisibility).observe(document.body, {
            attributes: true,
            attributeFilter: ["class"],
        })

        // Reuse the existing task-config keys so older Easy Diffusion UI code
        // continues to render these values without needing a core UI schema edit.
        if (typeof taskConfigSetup !== "undefined" && taskConfigSetup.taskConfig) {
            taskConfigSetup.taskConfig.output_after_step = {
                label: "Capture iterations",
                visible: ({ reqBody }) =>
                    Array.isArray(reqBody?.capture_steps) ||
                    String(reqBody?.intermediate_representation || "").startsWith(SPARSE_PREFIX) ||
                    (reqBody?.output_after_step !== undefined && reqBody?.output_after_step !== null),
                value: ({ reqBody }) => {
                    if (Array.isArray(reqBody?.capture_steps)) {
                        return formatCaptureSteps(reqBody.capture_steps)
                    }
                    const decoded = decodeSparseTransport(reqBody?.intermediate_representation, reqBody?.num_inference_steps || totalSteps())
                    if (decoded) {
                        return formatCaptureSteps(decoded.steps)
                    }
                    return reqBody?.output_after_step
                },
            }
            taskConfigSetup.taskConfig.intermediate_representation = {
                label: "Endpoint solver states",
                visible: ({ reqBody }) =>
                    Array.isArray(reqBody?.capture_steps) ||
                    String(reqBody?.intermediate_representation || "").startsWith(SPARSE_PREFIX),
                value: ({ reqBody }) => {
                    if (typeof reqBody?.capture_endpoint_solver_states === "boolean") {
                        return reqBody.capture_endpoint_solver_states ? "First + last" : "Off"
                    }
                    const decoded = decodeSparseTransport(reqBody?.intermediate_representation, reqBody?.num_inference_steps || totalSteps())
                    return decoded?.includeEndpointSolverStates ? "First + last" : "Off"
                },
            }
        }

        const originalGetCurrentUserRequest = getCurrentUserRequest
        getCurrentUserRequest = function (...args) {
            const task = originalGetCurrentUserRequest.apply(this, args)
            if (mvpAvailable()) {
                const steps = parseCurrentCaptureSteps(true)
                if (!steps) {
                    throw new Error(captureField.validationMessage || "Invalid capture iterations")
                }
                const includeEndpoints = endpointField.checked
                task.reqBody.capture_steps = [...steps]
                task.reqBody.capture_endpoint_solver_states = includeEndpoints
                // Compatibility transport for the existing classic backend request
                // model. output_after_step opens the callback bridge at the first
                // selected boundary; the sparse marker tells the collector which
                // exact boundaries to retain.
                task.reqBody.output_after_step = Math.min(...steps)
                task.reqBody.intermediate_representation = encodeSparseTransport(
                    steps,
                    totalSteps(),
                    includeEndpoints
                )
            }
            return task
        }

        if (typeof restoreTaskToUI === "function") {
            const originalRestoreTaskToUI = restoreTaskToUI
            restoreTaskToUI = function (...args) {
                const result = originalRestoreTaskToUI.apply(this, args)
                const task = args[0]
                const reqBody = task?.reqBody || {}
                const total = totalSteps()

                let steps = null
                let includeEndpoints = false
                if (Array.isArray(reqBody.capture_steps) && reqBody.capture_steps.length) {
                    try {
                        steps = parseCaptureExpression(reqBody.capture_steps.join(","), total)
                        includeEndpoints = !!reqBody.capture_endpoint_solver_states
                    } catch (_) {
                        steps = null
                    }
                }
                if (!steps) {
                    const decoded = decodeSparseTransport(reqBody.intermediate_representation, total)
                    if (decoded) {
                        steps = decoded.steps
                        includeEndpoints = decoded.includeEndpointSolverStates
                    }
                }
                if (!steps && reqBody.output_after_step !== undefined && reqBody.output_after_step !== null) {
                    const start = Math.max(1, Math.min(parseInt(reqBody.output_after_step) || total, total))
                    steps = start < total
                        ? Array.from({ length: total - start }, (_, index) => start + index)
                        : [total]
                    includeEndpoints = ["both", "solver_state"].includes(reqBody.intermediate_representation)
                }
                if (!steps) {
                    steps = [total]
                }

                captureField.value = formatCaptureSteps(steps)
                endpointField.checked = includeEndpoints
                linkedToInferenceSteps = steps.length === 1 && steps[0] === total
                updateVisibility()
                return result
            }
        }

        // Easy Diffusion's core renderer reverses final response.output before
        // drawing it. Sparse capture responses are deliberately chronological,
        // so pre-reverse a shallow copy and let the core reverse restore order.
        const originalShowImages = showImages
        showImages = function (reqBody, res, outputContainer, livePreview) {
            const hasStepMetadata =
                !livePreview &&
                Array.isArray(res?.output) &&
                res.output.some(
                    (entry) =>
                        entry?.output_step !== undefined &&
                        entry?.output_step !== null &&
                        entry?.total_steps !== undefined &&
                        entry?.total_steps !== null
                )

            let displayResponse = res
            if (hasStepMetadata) {
                displayResponse = {
                    ...res,
                    output: [...res.output].reverse(),
                }
            }

            const result = originalShowImages.call(this, reqBody, displayResponse, outputContainer, livePreview)

            if (hasStepMetadata) {
                const imageItems = Array.from(outputContainer.querySelectorAll(".imgItem"))
                displayResponse.output.forEach((entry, index) => {
                    const item = imageItems[index]
                    const image = item?.querySelector("img")
                    if (!item || !image) {
                        return
                    }

                    const step = parseInt(entry.output_step)
                    const total = parseInt(entry.total_steps)
                    const representation = entry.intermediate_representation || (entry.is_intermediate ? "solver_state" : "final")
                    image.setAttribute("data-steps", step)
                    image.setAttribute("data-output-step", step)
                    image.setAttribute("data-total-steps", total)
                    image.setAttribute("data-intermediate-representation", representation)

                    const counter = image.getAttribute("data-imagecounter")
                    if (
                        counter &&
                        typeof imageRequest !== "undefined" &&
                        imageRequest[counter]
                    ) {
                        imageRequest[counter].output_step = step
                        imageRequest[counter].total_steps = total
                        imageRequest[counter].is_intermediate = !!entry.is_intermediate
                        imageRequest[counter].output_representation = representation
                        // Review Capture historically looked at this field first.
                        // Make the per-image value precise rather than inheriting
                        // the task-level transport marker / legacy "both" value.
                        imageRequest[counter].intermediate_representation = representation
                    }

                    let stepLabel = item.querySelector(".imgStepLabel")
                    if (!stepLabel) {
                        stepLabel = document.createElement("span")
                        stepLabel.className = "imgInfoLabel imgStepLabel"
                        const seedLabel = item.querySelector(".imgSeedLabel")
                        seedLabel?.insertAdjacentElement("afterend", stepLabel)
                    }
                    const repLabel = entry.is_intermediate ? REPRESENTATION_LABELS[representation] : null
                    stepLabel.innerText = repLabel
                        ? `Step ${step}/${total} · ${repLabel}`
                        : `Step ${step}/${total}`
                })
            }

            return result
        }

        if (typeof getDownloadFilename === "function") {
            const originalGetDownloadFilename = getDownloadFilename
            getDownloadFilename = function (img, suffix) {
                const name = originalGetDownloadFilename.call(this, img, suffix)
                const representation = img?.dataset?.intermediateRepresentation
                if (!representation || representation === "final") {
                    return name
                }
                const marker = representation === "denoised" ? "denoised" : "solver-state"
                const dot = name.lastIndexOf(".")
                if (dot === -1) {
                    return `${name}_${marker}`
                }
                return `${name.slice(0, dot)}_${marker}${name.slice(dot)}`
            }
        }

        captureField.value = String(totalSteps())
        endpointField.checked = false
        updateVisibility()
        root.setTimeout(updateVisibility, 250)
        root.setTimeout(updateVisibility, 1000)
    }

    install()
})()