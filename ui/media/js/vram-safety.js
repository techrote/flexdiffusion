"use strict"

// Conservative browser-side batching/preview guard. The backend independently
// rechecks the memory profile on the actual device when a task starts.
var VramSafety = (function () {
    const HEAVY_MODEL_HINTS = ["sdxl", "sd_xl", "flux", "pony", "sd3", "stable-diffusion-3", "z-image", "lumina", "qwen-image", "anima"]

    function assess({ width, height, numOutputs, modelName, totalVramGb }) {
        const pixels = Number(width) * Number(height)
        const outputs = Number(numOutputs)
        const total = Number(totalVramGb)
        if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(pixels) || pixels <= 0 || !Number.isFinite(outputs) || outputs <= 0) {
            return { risk: "unknown", disableLivePreview: false, serialOutputs: false }
        }

        const model = String(modelName || "").toLowerCase()
        const largeModel = HEAVY_MODEL_HINTS.some((hint) => model.includes(hint))
        let risk = "normal"
        if (
            (total < 4.5 && pixels >= 768 * 768) ||
            (total < 6.5 && pixels >= 1024 * 1024) ||
            (total < 8.5 && largeModel && pixels >= 768 * 768) ||
            (total < 6.5 && outputs > 1 && pixels >= 512 * 512)
        ) {
            risk = "high"
        } else if (
            (total < 4.5 && pixels >= 512 * 512) ||
            (total < 8.5 && pixels >= 1024 * 1024) ||
            (total < 8.5 && largeModel) ||
            (total < 8.5 && outputs > 1 && pixels >= 768 * 768)
        ) {
            risk = "elevated"
        }

        return {
            risk,
            disableLivePreview: risk === "high",
            serialOutputs: risk === "high" && outputs > 1,
        }
    }

    function getConservativeGpuTotalGb(devices) {
        // An unpinned task can land on any active GPU: use the smallest known
        // capacity rather than incorrectly claiming the biggest GPU is used.
        const active = devices && devices.active
        if (!active) return null
        const totals = Object.values(active)
            .map((d) => Number(d && d.mem_total))
            .filter((n) => Number.isFinite(n) && n > 0)
        // Unknown capacities should not be used to justify an aggressive mode.
        if (Object.keys(active).some((key) => key !== "cpu" && !(Number(active[key] && active[key].mem_total) > 0))) return null
        return totals.length > 0 ? Math.min(...totals) : null
    }

    return { assess, getConservativeGpuTotalGb }
})()

if (typeof module !== "undefined" && module.exports) {
    module.exports = VramSafety
}
