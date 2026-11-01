"use strict"
const assert = require("node:assert/strict")
const { MODES, blendChannel, composeRgba } = require("../ui/media/js/compositor.js")

const pixel = (r, g, b, a = 255) => Uint8ClampedArray.from([r, g, b, a])
function composited(a, b, mask = null, mode = "normal", opacity = 1, invertMask = false, maskChannel = "luminance") {
    return [...composeRgba(a, b, mask, { mode, opacity, invertMask, maskChannel })]
}

assert.equal(MODES.length, 18)
assert.deepEqual(composited(pixel(40, 90, 140), pixel(200, 120, 30)), [200, 120, 30, 255])
assert.deepEqual(composited(pixel(40, 90, 140), pixel(200, 120, 30), pixel(0, 0, 0)), [40, 90, 140, 255])
assert.deepEqual(composited(pixel(40, 90, 140), pixel(200, 120, 30), pixel(255, 255, 255)), [200, 120, 30, 255])
assert.deepEqual(composited(pixel(0, 0, 0), pixel(255, 255, 255), pixel(128, 128, 128)), [128, 128, 128, 255])
assert.deepEqual(composited(pixel(0, 0, 0), pixel(255, 255, 255), pixel(128, 128, 128), "normal", 0.5), [64, 64, 64, 255])
assert.deepEqual(composited(pixel(10, 20, 30), pixel(200, 200, 200), pixel(0, 0, 0), "normal", 1, true), [200, 200, 200, 255])
assert.deepEqual(composited(pixel(10, 20, 30), pixel(200, 200, 200), pixel(255, 255, 255), "normal", 1, true), [10, 20, 30, 255])
assert.deepEqual(composited(pixel(10, 20, 30), pixel(200, 200, 200), pixel(0, 0, 0, 255), "normal", 1, false, "alpha"), [200, 200, 200, 255])
assert.deepEqual(composited(pixel(10, 20, 30), pixel(200, 200, 200), pixel(255, 255, 255, 0), "normal", 1, false, "alpha"), [10, 20, 30, 255])

// A transparent lower layer must reveal B rather than multiply B by black.
assert.deepEqual(composited(pixel(0, 0, 0, 0), pixel(180, 60, 30), null, "multiply"), [180, 60, 30, 255])
assert.deepEqual(composited(pixel(0, 0, 0, 0), pixel(200, 100, 50, 128)), [200, 100, 50, 128])
assert.deepEqual(composited(pixel(0, 0, 0, 0), pixel(255, 255, 255, 0)), [0, 0, 0, 0])

// Known endpoint / neutral-value identities in normalized sRGB.
for(const [mode,a,b,expected] of [
    ["multiply", 0.3, 1, 0.3],
    ["multiply", 0.3, 0, 0],
    ["screen", 0.3, 0, 0.3],
    ["screen", 0.3, 1, 1],
    ["difference", 0.3, 0.3, 0],
    ["difference", 0.3, 1, 0.7],
    ["exclusion", 0.5, 0.5, 0.5],
    ["overlay", 0.2, 0.5, 0.2],
    ["hard-light", 0.2, 0.5, 0.2],
    ["soft-light", 0.2, 0.5, 0.2],
    ["dodge", 0.4, 0, 0.4],
    ["burn", 0.4, 1, 0.4],
    ["addition", 0.7, 0.7, 1],
    ["subtract", 0.2, 0.7, 0],
    ["grain-extract", 0.5, 0.5, 0.5],
    ["grain-merge", 0.5, 0.5, 0.5],
    ["linear-burn", 0.3, 0.6, 0],
    ["divide", 0.5, 0, 1],
    ["divide", 0, 0, 0],
]) assert.ok(Math.abs(blendChannel(a,b,mode)-expected)<1e-9, mode)

// All operations stay finite over representative extreme combinations.
for(const [name] of MODES) {
    for (const a of [0, 0.125, 0.5, 0.875, 1]) {
        for (const b of [0, 0.125, 0.5, 0.875, 1]) {
            const v=blendChannel(a,b,name)
            assert.ok(Number.isFinite(v)&&v>=0&&v<=1, name+":"+a+"/"+b+"="+v)
        }
    }
}
const inputA=Uint8ClampedArray.from([123, 45, 67, 128, 0, 0, 0, 0])
const inputB=Uint8ClampedArray.from([234, 12, 100, 193, 12, 23, 34, 255])
const snapshotA=[...inputA], snapshotB=[...inputB]
const complex=composeRgba(inputA,inputB,null,{mode:"soft-light",opacity:.37})
assert.equal(complex.length,inputA.length)
assert.deepEqual([...inputA],snapshotA)
assert.deepEqual([...inputB],snapshotB)
for(const invalid of [
    () => composeRgba([1,2,3],pixel(2,3,4)),
    () => composeRgba(pixel(2,3,4),pixel(2,3,4),[1,2]),
    () => composeRgba(pixel(2,3,4),pixel(2,3,4),null,{opacity:2}),
    () => composeRgba(pixel(2,3,4),pixel(2,3,4),null,{mode:"hsv-hue"}),
]) assert.throws(invalid)

console.log("Blend Lab: 18 modes, grayscale/alpha masks, opacity, inversion, transparency and validation passed")
