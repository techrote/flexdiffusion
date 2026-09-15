;(function () {
    "use strict"

    const root = typeof window !== "undefined" ? window : globalThis

    function orderedUniqueSamplerNames(primary, extras) {
        const ordered = []
        const seen = new Set()
        ;[primary].concat(extras || []).forEach((name) => {
            if (typeof name !== "string" || name.trim() === "" || seen.has(name)) {
                return
            }
            seen.add(name)
            ordered.push(name)
        })
        return ordered
    }

    root.__flexDiffusionSamplerComparisonTest = {
        orderedUniqueSamplerNames,
    }

    if (typeof document === "undefined") {
        return
    }
    if (root.__flexDiffusionSamplerComparisonInstalled) {
        return
    }

    function install() {
        const samplerField = document.querySelector("#sampler_name")
        const samplerRow = samplerField?.closest("tr")
        const makeImageButton = document.querySelector("#makeImage")
        const seedField = document.querySelector("#seed")
        const randomSeedField = document.querySelector("#random_seed")

        if (
            !samplerField ||
            !samplerRow ||
            !makeImageButton ||
            !seedField ||
            !randomSeedField ||
            typeof makeImage !== "function" ||
            typeof getCurrentUserRequest !== "function"
        ) {
            root.setTimeout(install, 50)
            return
        }

        if (root.__flexDiffusionSamplerComparisonInstalled) {
            return
        }
        root.__flexDiffusionSamplerComparisonInstalled = true

        // FlexDiffusion comparisons are intended to be reproducible by default.
        // Keep the seed field active and remove Easy Diffusion's automatic-random
        // checkbox from normal use. Explicit actions such as "Make Similar" may
        // still choose their own seed deliberately.
        const randomSeedLabel = document.querySelector('label[for="random_seed"]')
        function forceFixedSeedMode() {
            randomSeedField.checked = false
            randomSeedField.disabled = true
            randomSeedField.style.display = "none"
            if (randomSeedLabel) {
                randomSeedLabel.style.display = "none"
            }
            seedField.disabled = false
            if (!Number.isFinite(parseInt(seedField.value))) {
                seedField.value = "0"
            }
        }
        forceFixedSeedMode()
        randomSeedField.addEventListener("change", forceFixedSeedMode)
        randomSeedField.addEventListener("input", forceFixedSeedMode)

        // Restore/import operations can mutate the hidden checkbox. Enforce the
        // fixed-seed rule immediately before every normal request is built.
        const originalGetCurrentUserRequest = getCurrentUserRequest
        getCurrentUserRequest = function (...args) {
            forceFixedSeedMode()
            const task = originalGetCurrentUserRequest.apply(this, args)
            if (task?.reqBody) {
                task.reqBody.used_random_seed = false
            }
            return task
        }

        const compareRow = document.createElement("tr")
        compareRow.id = "sampler_compare_row"
        compareRow.className = "pl-5"
        compareRow.innerHTML = `
            <td><label>Also run:</label></td>
            <td>
                <details id="sampler_compare_details">
                    <summary id="sampler_compare_summary" style="cursor:pointer">Add/compare samplers…</summary>
                    <div id="sampler_compare_options" style="max-height: 150px; overflow-y: auto; margin-top: 4px"></div>
                </details>
                <small>Checked samplers run as separate tasks with the same seed.</small>
            </td>
        `
        samplerRow.insertAdjacentElement("afterend", compareRow)

        const compareDetails = compareRow.querySelector("#sampler_compare_details")
        const compareSummary = compareRow.querySelector("#sampler_compare_summary")
        const compareOptions = compareRow.querySelector("#sampler_compare_options")
        const selectedExtras = new Set()

        function optionIsAvailable(option) {
            if (!option?.value) {
                return false
            }
            if (option.classList.contains("gated-feature") && option.style.display === "none") {
                return false
            }
            return true
        }

        function updateSummary() {
            const count = selectedExtras.size
            compareSummary.textContent = count > 0 ? `Also run ${count} sampler${count === 1 ? "" : "s"}…` : "Add/compare samplers…"
        }

        function rebuildOptions() {
            const currentPrimary = samplerField.value
            compareOptions.innerHTML = ""

            Array.from(samplerField.options).forEach((option) => {
                if (!optionIsAvailable(option) || option.value === currentPrimary) {
                    return
                }

                const label = document.createElement("label")
                label.style.display = "block"
                label.style.whiteSpace = "nowrap"

                const checkbox = document.createElement("input")
                checkbox.type = "checkbox"
                checkbox.value = option.value
                checkbox.checked = selectedExtras.has(option.value)
                checkbox.style.marginRight = "4px"
                checkbox.addEventListener("change", function () {
                    if (this.checked) {
                        selectedExtras.add(this.value)
                    } else {
                        selectedExtras.delete(this.value)
                    }
                    updateSummary()
                })

                label.appendChild(checkbox)
                label.appendChild(document.createTextNode(option.textContent))
                compareOptions.appendChild(label)
            })

            updateSummary()
        }

        function updateVisibility() {
            const samplerVisible = root.getComputedStyle(samplerRow).display !== "none"
            compareRow.style.display = samplerVisible ? "" : "none"
            if (!samplerVisible) {
                compareDetails.open = false
            }
        }

        function selectedSamplerNames() {
            return orderedUniqueSamplerNames(samplerField.value, Array.from(selectedExtras))
        }

        samplerField.addEventListener("change", function () {
            rebuildOptions()
            updateVisibility()
        })
        document.addEventListener("refreshModels", function () {
            root.setTimeout(rebuildOptions, 0)
            root.setTimeout(updateVisibility, 0)
        })
        new MutationObserver(function () {
            rebuildOptions()
            updateVisibility()
        }).observe(samplerField, {
            subtree: true,
            attributes: true,
            attributeFilter: ["style", "class"],
        })
        new MutationObserver(updateVisibility).observe(samplerRow, {
            attributes: true,
            attributeFilter: ["style", "class"],
        })

        const originalMakeImage = makeImage
        function makeComparedImages() {
            forceFixedSeedMode()
            const samplers = selectedSamplerNames()
            if (samplers.length <= 1) {
                return originalMakeImage()
            }

            const originalSampler = samplerField.value
            try {
                samplers.forEach((samplerName) => {
                    samplerField.value = samplerName
                    originalMakeImage()
                })
            } finally {
                samplerField.value = originalSampler
                samplerField.dispatchEvent(new Event("change"))
            }
        }

        // Ctrl+Enter and any future callers resolve the global function at call
        // time, so replace it with the comparison-aware wrapper.
        makeImage = makeComparedImages

        // The existing Make Image click listener already holds a reference to the
        // old function. Intercept only when multiple samplers are selected, then
        // enqueue each sampler as its own ordinary task (matching manual serial
        // runs). With no extras selected, stock click behaviour is untouched.
        makeImageButton.addEventListener(
            "click",
            function (event) {
                if (selectedSamplerNames().length <= 1) {
                    return
                }
                event.preventDefault()
                event.stopImmediatePropagation()
                makeComparedImages()
            },
            true
        )

        rebuildOptions()
        updateVisibility()
        root.setTimeout(rebuildOptions, 250)
        root.setTimeout(updateVisibility, 250)
        root.setTimeout(rebuildOptions, 1000)
        root.setTimeout(updateVisibility, 1000)
    }

    install()
})()
