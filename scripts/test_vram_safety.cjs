"use strict"

const assert = require("node:assert/strict")
const test = require("node:test")
const safety = require("../ui/media/js/vram-safety.js")

test("4 GB 1024px job switches previews off", () => {
    const advice = safety.assess({ width: 1024, height: 1024, numOutputs: 1, totalVramGb: 4, modelName: "SD 1.4" })
    assert.equal(advice.risk, "high")
    assert.equal(advice.disableLivePreview, true)
    assert.equal(advice.serialOutputs, false)
})

test("single-image batching for risky multi-image job", () => {
    const advice = safety.assess({ width: 1024, height: 1024, numOutputs: 4, totalVramGb: 4, modelName: "SD 1.4" })
    assert.equal(advice.serialOutputs, true)
})

test("unknown VRAM does not invent a budget", () => {
    const advice = safety.assess({ width: 1024, height: 1024, numOutputs: 4, totalVramGb: null })
    assert.equal(advice.risk, "unknown")
    assert.equal(advice.serialOutputs, false)
})

test("large GPU does not needlessly serialise", () => {
    const advice = safety.assess({ width: 1024, height: 1024, numOutputs: 4, totalVramGb: 24, modelName: "SD 1.4" })
    assert.equal(advice.risk, "normal")
    assert.equal(advice.serialOutputs, false)
})

test("model family contributes to decision", () => {
    const advice = safety.assess({ width: 768, height: 768, numOutputs: 1, totalVramGb: 8, modelName: "SDXL" })
    assert.equal(advice.risk, "high")
})

test("unknown active GPU capacity cannot justify high settings", () => {
    const value = safety.getConservativeGpuTotalGb({
        active: { "cuda:0": { mem_total: 12 }, "cuda:1": { name: "other" } },
    })
    assert.equal(value, null)
})

test("uses smallest active GPU capacity", () => {
    const value = safety.getConservativeGpuTotalGb({
        active: { "cuda:0": { mem_total: 12 }, "cuda:1": { mem_total: 4 } },
    })
    assert.equal(value, 4)
})
