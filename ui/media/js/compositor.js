/* RGB compositing primitives for the Blend Lab.
 * Inputs and output are straight-alpha RGBA, sRGB byte channels. Layer B is above
 * layer A; a grayscale coverage map scales B's opacity before compositing.
 *
 * Operations follow familiar GIMP / SVG-style formulae but are deliberately not
 * represented as bit-for-bit equivalents of GEGL's linear-light, colour-managed
 * algorithms. No diffusion latents are processed by this module.
 */
;(function (root) {
    "use strict"

    const MODES = [
        ["normal", "Normal"],
        ["lighten", "Lighten only"],
        ["screen", "Screen"],
        ["dodge", "Dodge"],
        ["addition", "Addition"],
        ["darken", "Darken only"],
        ["multiply", "Multiply"],
        ["burn", "Burn"],
        ["linear-burn", "Linear burn"],
        ["overlay", "Overlay"],
        ["soft-light", "Soft light"],
        ["hard-light", "Hard light"],
        ["difference", "Difference"],
        ["exclusion", "Exclusion"],
        ["subtract", "Subtract"],
        ["grain-extract", "Grain extract"],
        ["grain-merge", "Grain merge"],
        ["divide", "Divide"],
    ]

    const clip = (v) => Math.min(1, Math.max(0, v))

    function blendChannel(a, b, mode) {
        switch (mode) {
            case "normal": return b
            case "lighten": return Math.max(a, b)
            case "screen": return 1 - (1 - a) * (1 - b)
            case "dodge": return b >= 1 ? 1 : clip(a / (1 - b))
            case "addition": return clip(a + b)
            case "darken": return Math.min(a, b)
            case "multiply": return a * b
            case "burn": return b <= 0 ? 0 : 1 - clip((1 - a) / b)
            case "linear-burn": return clip(a + b - 1)
            case "overlay": return a <= 0.5 ? 2 * a * b : 1 - 2 * (1 - a) * (1 - b)
            case "soft-light": {
                const d = a <= 0.25 ? ((16 * a - 12) * a + 4) * a : Math.sqrt(a)
                return b <= 0.5 ? a - (1 - 2 * b) * a * (1 - a) : a + (2 * b - 1) * (d - a)
            }
            case "hard-light": return b <= 0.5 ? 2 * a * b : 1 - 2 * (1 - a) * (1 - b)
            case "difference": return Math.abs(a - b)
            case "exclusion": return a + b - 2 * a * b
            case "subtract": return clip(a - b)
            case "grain-extract": return clip(a - b + 0.5)
            case "grain-merge": return clip(a + b - 0.5)
            case "divide": return b <= 0 ? (a <= 0 ? 0 : 1) : clip(a / b)
            default: throw new RangeError("Unsupported composition mode: " + mode)
        }
    }

    function composeRgba(bottom, top, coverage = null, options = {}) {
        if (!bottom || !top || bottom.length !== top.length || bottom.length % 4 !== 0) {
            throw new RangeError("Both images must have matching RGBA dimensions")
        }
        if (coverage !== null && coverage.length !== bottom.length) {
            throw new RangeError("Coverage mask must match image dimensions")
        }
        const mode = options.mode || "normal"
        if (!MODES.some(([id]) => id === mode)) throw new RangeError("Unsupported mode: " + mode)
        const opacity = options.opacity ?? 1
        if (!Number.isFinite(opacity) || opacity < 0 || opacity > 1) {
            throw new RangeError("Opacity must be within 0 and 1")
        }
        const maskChannel = options.maskChannel || "luminance"
        if (!["luminance", "alpha"].includes(maskChannel)) throw new RangeError("Unknown mask channel")
        const invertMask = !!options.invertMask
        const result = new Uint8ClampedArray(bottom.length)

        for (let i = 0; i < bottom.length; i += 4) {
            const backAlpha = bottom[i + 3] / 255
            const topAlpha = top[i + 3] / 255
            let map = 1
            if (coverage) {
                const mAlpha = coverage[i + 3] / 255
                map = maskChannel === "alpha" ? mAlpha
                    : ((0.2126 * coverage[i] + 0.7152 * coverage[i + 1] + 0.0722 * coverage[i + 2]) / 255) * mAlpha
                if (invertMask) map = 1 - map
            }
            const opacityTop = topAlpha * opacity * map
            const outAlpha = opacityTop + backAlpha * (1 - opacityTop)
            for (let c = 0; c < 3; c++) {
                const a = bottom[i + c] / 255
                const b = top[i + c] / 255
                const blended = blendChannel(a, b, mode)
                // W3C-style nonpremultiplied source-over with a separable blend.
                const premultiplied = (1 - opacityTop) * backAlpha * a +
                    opacityTop * ((1 - backAlpha) * b + backAlpha * blended)
                result[i + c] = Math.round(255 * (outAlpha ? premultiplied / outAlpha : 0))
            }
            result[i + 3] = Math.round(outAlpha * 255)
        }
        return result
    }

    const api = { MODES, blendChannel, composeRgba }
    root.FlexDiffusionCompositor = api
    if (typeof module !== "undefined" && module.exports) module.exports = api
})(typeof window !== "undefined" ? window : globalThis)
