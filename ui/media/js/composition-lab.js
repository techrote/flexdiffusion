/* Optional RGB image-composition workstation for sampler checkpoint experiments.
 * This intentionally operates on decoded images, not directly on diffusion latents.
 */
;(function (root) {
    "use strict"
    if (typeof document === "undefined" || typeof createTab !== "function") return

    createTab({
        id: "blend-lab",
        label: "Blend Lab",
        icon: "fa-layer-group",
        content: `
            <div class="blend-lab-header">
                <h2>Masked image compositing</h2>
                <p>Combine two generated images or saved iteration frames. Use a grayscale or alpha map to control the upper image's contribution at each pixel.</p>
                <p class="blend-lab-note">RGB post-processing: these GIMP-inspired modes are not latent-space sampler updates, and linear-light GEGL colour management is not replicated.</p>
            </div>
            <div class="blend-lab-grid">
                <section class="blend-lab-panel">
                    <div class="blend-lab-field">
                        <label for="blend-lab-bottom">Image A · lower layer / base</label>
                        <input id="blend-lab-bottom" type="file" accept="image/*">
                    </div>
                    <div class="blend-lab-field">
                        <label for="blend-lab-top">Image B · upper layer</label>
                        <input id="blend-lab-top" type="file" accept="image/*">
                    </div>
                    <div class="blend-lab-field">
                        <label for="blend-lab-coverage">Coverage map (optional)</label>
                        <input id="blend-lab-coverage" type="file" accept="image/*">
                        <button id="blend-lab-clear-map" class="secondaryButton" type="button">Clear map</button>
                        <small>White uses the upper layer; black preserves the lower layer. Gray gives intermediate opacity. A missing map is all white.</small>
                    </div>
                    <div class="blend-lab-field">
                        <label for="blend-lab-mode">Blend mode</label>
                        <select id="blend-lab-mode"></select>
                    </div>
                    <div class="blend-lab-field">
                        <label for="blend-lab-opacity">Upper-layer opacity: <output id="blend-lab-opacity-label">100%</output></label>
                        <input id="blend-lab-opacity" type="range" min="0" max="100" step="1" value="100">
                    </div>
                    <div class="blend-lab-field">
                        <label for="blend-lab-mask-channel">Map source</label>
                        <select id="blend-lab-mask-channel">
                            <option value="luminance">Luminance × alpha</option>
                            <option value="alpha">Alpha channel only</option>
                        </select>
                        <label class="blend-lab-inline" for="blend-lab-invert">
                            <input type="checkbox" id="blend-lab-invert"> Invert coverage (black/white)
                        </label>
                    </div>
                    <div class="blend-lab-actions">
                        <button id="blend-lab-save" class="secondaryButton" type="button" disabled>Save composited PNG</button>
                        <button id="blend-lab-use-init" class="primaryButton" type="button" disabled>Use as img2img source</button>
                    </div>
                    <p id="blend-lab-status" role="status" aria-live="polite">Choose images A and B to compose.</p>
                </section>
                <section class="blend-lab-output">
                    <canvas id="blend-lab-canvas" aria-label="Composited image preview"></canvas>
                    <small>Output uses image A's native dimensions. Image B and the coverage map are resampled to match A. PNG export retains transparency.</small>
                </section>
            </div>
        `,
    })
    // Keep research tabs adjacent to Generate, ahead of settings/help.
    const tabSettings = document.getElementById("tab-settings")
    const labTab = document.getElementById("tab-blend-lab")
    if (tabSettings && labTab) tabSettings.before(labTab)

    const q = (name) => document.getElementById("blend-lab-" + name)
    const mode = q("mode")
    const modeGroups = [
        ["Normal", ["normal"]],
        ["Lighten", ["lighten", "screen", "dodge", "addition"]],
        ["Darken", ["darken", "multiply", "burn", "linear-burn"]],
        ["Contrast", ["overlay", "soft-light", "hard-light"]],
        ["Difference", ["difference", "exclusion", "subtract", "grain-extract", "grain-merge", "divide"]],
    ]
    const definitions = root.FlexDiffusionCompositor
    if (!definitions) return
    const labels = new Map(definitions.MODES)
    modeGroups.forEach(([label, members]) => {
        const group = document.createElement("optgroup")
        group.label = label
        members.forEach((key) => {
            const option = document.createElement("option")
            option.value = key
            option.textContent = labels.get(key)
            group.appendChild(option)
        })
        mode.appendChild(group)
    })

    const state = { bottom: null, top: null, coverage: null, pixels: null }
    const versions = { bottom: 0, top: 0, coverage: 0 }
    const canvas = q("canvas")
    const status = q("status")
    const saveBtn = q("save")
    const useBtn = q("use-init")
    const pixelsMax = 8388608 // bounded ~8 MP max; avoid UI-thread memory explosions

    function invalidate(message) {
        state.pixels = null
        saveBtn.disabled = true
        useBtn.disabled = true
        if (message) status.textContent = message
    }

    async function decodeFile(file) {
        if (!file || !file.type.startsWith("image/")) {
            throw new Error("Please choose an image file.")
        }
        const url = URL.createObjectURL(file)
        try {
            const image = new Image()
            image.src = url
            await image.decode()
            if (!image.naturalWidth || !image.naturalHeight) {
                throw new Error("The image has no readable dimensions.")
            }
            return image
        } finally {
            URL.revokeObjectURL(url)
        }
    }

    function raster(image, width, height) {
        const surface = document.createElement("canvas")
        surface.width = width
        surface.height = height
        const context = surface.getContext("2d", { willReadFrequently: true })
        context.imageSmoothingEnabled = true
        context.imageSmoothingQuality = "high"
        context.drawImage(image, 0, 0, width, height)
        return context.getImageData(0, 0, width, height).data
    }

    function recompose() {
        const base = state.bottom
        const upper = state.top
        if (!base || !upper) {
            canvas.width = canvas.height = 0
            invalidate("Choose images A and B to compose.")
            return
        }
        const width = base.naturalWidth
        const height = base.naturalHeight
        if (!width || !height || width * height > pixelsMax) {
            invalidate("Image A exceeds the 8-megapixel safety limit. Downscale it before composing.")
            return
        }
        try {
            const a = raster(base, width, height)
            const b = raster(upper, width, height)
            const mask = state.coverage ? raster(state.coverage, width, height) : null
            const pixels = definitions.composeRgba(a, b, mask, {
                mode: mode.value,
                opacity: Number(q("opacity").value) / 100,
                maskChannel: q("mask-channel").value,
                invertMask: q("invert").checked,
            })
            canvas.width = width
            canvas.height = height
            const context = canvas.getContext("2d")
            const output = context.createImageData(width, height)
            output.data.set(pixels)
            context.putImageData(output, 0, 0)
            state.pixels = pixels
            saveBtn.disabled = false
            useBtn.disabled = false
            const notes = []
            if (upper.naturalWidth !== width || upper.naturalHeight !== height) notes.push("B resized")
            if (state.coverage && (state.coverage.naturalWidth !== width || state.coverage.naturalHeight !== height)) notes.push("map resized")
            status.textContent = `${width} × ${height} · ${labels.get(mode.value)} · ${q("opacity").value}% opacity` +
                (notes.length ? " · " + notes.join(", ") : "")
        } catch (error) {
            invalidate("Compositing failed: " + error.message)
        }
    }

    for (const key of ["bottom", "top", "coverage"]) {
        const input = q(key)
        input.addEventListener("change", async () => {
            const version = ++versions[key]
            const file = input.files[0]
            if (!file) {
                state[key] = null
                recompose()
                return
            }
            invalidate("Decoding " + file.name + "…")
            try {
                const image = await decodeFile(file)
                if (versions[key] !== version) return // stale async selection
                state[key] = image
                recompose()
            } catch (error) {
                if (versions[key] !== version) return
                state[key] = null
                invalidate("Unable to load " + file.name + ": " + error.message)
            }
        })
    }

    q("clear-map").addEventListener("click", () => {
        ++versions.coverage
        state.coverage = null
        q("coverage").value = ""
        recompose()
    })

    for (const key of ["mode", "opacity", "mask-channel", "invert"]) {
        q(key).addEventListener("change", recompose)
    }
    q("opacity").addEventListener("input", () => {
        q("opacity-label").textContent = q("opacity").value + "%"
    })

    saveBtn.addEventListener("click", () => {
        if (!state.pixels) return
        canvas.toBlob((blob) => {
            if (!blob) {
                status.textContent = "Could not encode the PNG."
                return
            }
            const url = URL.createObjectURL(blob)
            const link = document.createElement("a")
            link.href = url
            link.download = "flexdiffusion-blend.png"
            link.click()
            root.setTimeout(() => URL.revokeObjectURL(url), 60000)
        }, "image/png")
    })

    useBtn.addEventListener("click", () => {
        if (!state.pixels) return
        const init = document.getElementById("init_image_preview")
        if (!init) {
            status.textContent = "The generator's image input is not available."
            return
        }
        // Keep existing generator controls and batching; replace only the input
        // image, not a previous inpainting selection from another source.
        const maskToggle = document.getElementById("enable_mask")
        if (maskToggle) {
            maskToggle.checked = false
            maskToggle.dispatchEvent(new Event("change"))
        }
        if (typeof imageInpainter !== "undefined") {
            imageInpainter.layers.drawing.ctx.clearRect(0, 0, imageInpainter.width, imageInpainter.height)
            imageInpainter.history.clear()
        }
        init.src = canvas.toDataURL("image/png")
        if (typeof setImageWidthHeight === "function" &&
            typeof MAX_INIT_IMAGE_DIMENSION !== "undefined" &&
            canvas.width <= MAX_INIT_IMAGE_DIMENSION && canvas.height <= MAX_INIT_IMAGE_DIMENSION) {
            setImageWidthHeight(canvas.width, canvas.height)
        }
        document.getElementById("tab-main").click()
        status.textContent = "Composition loaded as the initial image. Adjust generation settings as needed."
    })
})(typeof window !== "undefined" ? window : globalThis)
