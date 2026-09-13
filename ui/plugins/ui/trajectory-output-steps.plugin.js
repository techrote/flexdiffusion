;(function () {
    "use strict"

    if (window.__flexDiffusionOutputAfterStepInstalled) {
        return
    }

    const REPRESENTATION_LABELS = {
        denoised: "Denoised estimate",
        solver_state: "Solver state",
        final: "Final",
    }

    function install() {
        const inferenceStepsField = document.querySelector("#num_inference_steps")
        if (
            !inferenceStepsField ||
            typeof getCurrentUserRequest !== "function" ||
            typeof showImages !== "function"
        ) {
            window.setTimeout(install, 50)
            return
        }

        if (window.__flexDiffusionOutputAfterStepInstalled) {
            return
        }
        window.__flexDiffusionOutputAfterStepInstalled = true

        const inferenceRow = inferenceStepsField.closest("tr")
        if (!inferenceRow) {
            console.error("FlexDiffusion Output After Step: could not find Inference Steps row")
            return
        }

        let outputAfterStepField = document.querySelector("#output_after_step")
        let outputAfterStepRow = outputAfterStepField?.closest("tr")

        if (!outputAfterStepField) {
            outputAfterStepRow = document.createElement("tr")
            outputAfterStepRow.id = "output_after_step_row"
            outputAfterStepRow.className = "pl-5"
            outputAfterStepRow.innerHTML = `
                <td>
                    <label for="output_after_step">Output After Step:</label>
                </td>
                <td>
                    <input
                        id="output_after_step"
                        name="output_after_step"
                        type="number"
                        min="1"
                        step="1"
                        style="width: 42pt"
                        value="${inferenceStepsField.value}"
                        onkeypress="preventNonNumericalInput(event)"
                        inputmode="numeric"
                    >
                    <i class="fa-solid fa-circle-question help-btn">
                        <span class="simple-tooltip top-left">
                            Render every completed denoising step from this value through the final step.
                            The denoising trajectory runs once; selected intermediate latents are copied to
                            CPU and decoded after sampling. MVP: classic txt2img fixed-step k-diffusion samplers.
                        </span>
                    </i>
                </td>
            `
            inferenceRow.insertAdjacentElement("afterend", outputAfterStepRow)
            outputAfterStepField = outputAfterStepRow.querySelector("#output_after_step")
        }

        let representationField = document.querySelector("#intermediate_representation")
        let representationRow = representationField?.closest("tr")
        if (!representationField) {
            representationRow = document.createElement("tr")
            representationRow.id = "intermediate_representation_row"
            representationRow.className = "pl-5"
            representationRow.innerHTML = `
                <td>
                    <label for="intermediate_representation">Intermediate View:</label>
                </td>
                <td>
                    <select id="intermediate_representation" name="intermediate_representation">
                        <option value="both" selected>Both</option>
                        <option value="denoised">Denoised estimate</option>
                        <option value="solver_state">Solver state</option>
                    </select>
                    <i class="fa-solid fa-circle-question help-btn">
                        <span class="simple-tooltip top-left">
                            Denoised estimate is the model's current predicted clean image. Solver state is
                            the actual noisy latent being integrated. Both are sampled at the same completed-step boundary.
                        </span>
                    </i>
                </td>
            `
            outputAfterStepRow.insertAdjacentElement("afterend", representationRow)
            representationField = representationRow.querySelector("#intermediate_representation")
        }

        const backendField = document.querySelector("#backend")
        const initImagePreviewContainer = document.querySelector("#init_image_preview_container")
        let linkedToInferenceSteps = true

        function totalSteps() {
            return Math.max(1, parseInt(inferenceStepsField.value) || 1)
        }

        function clampedOutputStep() {
            const total = totalSteps()
            let value = parseInt(outputAfterStepField.value)
            if (!Number.isFinite(value)) {
                value = total
            }
            return Math.max(1, Math.min(value, total))
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

        function resetToFinal() {
            linkedToInferenceSteps = true
            outputAfterStepField.value = totalSteps()
        }

        function updateVisibility() {
            const available = mvpAvailable()
            outputAfterStepRow.style.display = available ? "" : "none"
            outputAfterStepField.disabled = !available

            const hasIntermediates = available && clampedOutputStep() < totalSteps()
            representationRow.style.display = hasIntermediates ? "" : "none"
            representationField.disabled = !hasIntermediates

            if (!available) {
                resetToFinal()
            }
        }

        inferenceStepsField.addEventListener("input", function () {
            const total = totalSteps()
            if (linkedToInferenceSteps) {
                outputAfterStepField.value = total
                updateVisibility()
                return
            }
            const current = clampedOutputStep()
            outputAfterStepField.value = current
            if (current === total) {
                linkedToInferenceSteps = true
            }
            updateVisibility()
        })

        outputAfterStepField.addEventListener("change", function () {
            const value = clampedOutputStep()
            outputAfterStepField.value = value
            linkedToInferenceSteps = value === totalSteps()
            updateVisibility()
        })

        outputAfterStepField.addEventListener("blur", function () {
            outputAfterStepField.dispatchEvent(new Event("change"))
        })

        if (backendField) {
            backendField.addEventListener("change", updateVisibility)
        }

        if (initImagePreviewContainer) {
            new MutationObserver(updateVisibility).observe(initImagePreviewContainer, {
                attributes: true,
                attributeFilter: ["class"],
            })
        }

        // getAppConfig() updates body classes after applying the backend config
        // without firing a change event on the backend select. Observe that
        // transition so the row reliably appears after an ed_classic restart.
        new MutationObserver(updateVisibility).observe(document.body, {
            attributes: true,
            attributeFilter: ["class"],
        })

        // Add active values to the normal task summary.
        if (typeof taskConfigSetup !== "undefined" && taskConfigSetup.taskConfig) {
            taskConfigSetup.taskConfig.output_after_step = {
                label: "Output After Step",
                visible: ({ reqBody }) =>
                    reqBody?.output_after_step !== undefined &&
                    reqBody?.output_after_step !== null &&
                    reqBody.output_after_step < reqBody.num_inference_steps,
            }
            taskConfigSetup.taskConfig.intermediate_representation = {
                label: "Intermediate View",
                visible: ({ reqBody }) =>
                    reqBody?.output_after_step !== undefined &&
                    reqBody?.output_after_step !== null &&
                    reqBody.output_after_step < reqBody.num_inference_steps,
                value: ({ reqBody }) => {
                    const value = reqBody?.intermediate_representation || "both"
                    if (value === "both") return "Both"
                    return REPRESENTATION_LABELS[value] || value
                },
            }
        }

        const originalGetCurrentUserRequest = getCurrentUserRequest
        getCurrentUserRequest = function (...args) {
            const task = originalGetCurrentUserRequest.apply(this, args)
            if (mvpAvailable()) {
                const step = clampedOutputStep()
                outputAfterStepField.value = step
                task.reqBody.output_after_step = step
                task.reqBody.intermediate_representation = representationField.value || "both"
            }
            return task
        }

        if (typeof restoreTaskToUI === "function") {
            const originalRestoreTaskToUI = restoreTaskToUI
            restoreTaskToUI = function (...args) {
                const result = originalRestoreTaskToUI.apply(this, args)
                const task = args[0]
                const requested = task?.reqBody?.output_after_step
                const representation = task?.reqBody?.intermediate_representation
                if (requested !== undefined && requested !== null && mvpAvailable()) {
                    outputAfterStepField.value = Math.max(1, Math.min(parseInt(requested) || totalSteps(), totalSteps()))
                    linkedToInferenceSteps = parseInt(outputAfterStepField.value) === totalSteps()
                    representationField.value = ["both", "denoised", "solver_state"].includes(representation)
                        ? representation
                        : "both"
                } else {
                    resetToFinal()
                    representationField.value = "both"
                }
                updateVisibility()
                return result
            }
        }

        // Easy Diffusion's core renderer reverses final response.output before
        // drawing it. Output After Step responses are deliberately chronological,
        // so pre-reverse a shallow copy and let the core reverse restore that
        // chronological order. Then annotate the ordinary image cards.
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
                // Core showImages() has reversed displayResponse.output in place,
                // so it is chronological again at this point.
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

        // Denoised and solver-state images from the same step would otherwise
        // have identical Easy Diffusion filenames. Add a representation suffix
        // to individual and ZIP downloads while leaving final-image names alone.
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

        resetToFinal()
        representationField.value = "both"
        updateVisibility()
        // Settings are populated asynchronously during startup on some builds.
        window.setTimeout(updateVisibility, 250)
        window.setTimeout(updateVisibility, 1000)
    }

    install()
})()
