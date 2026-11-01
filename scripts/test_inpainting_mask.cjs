const test = require("node:test")
const assert = require("node:assert/strict")
const {
    alphaToInpaintingMaskPixels,
    invertInpaintingMaskPixels,
    opaqueInpaintingMaskPixels,
} = require("../ui/media/js/inpainting-mask.js")

const rgba = (...values) => Uint8ClampedArray.from(values)

test("transparent PNG alpha becomes white, opaque becomes black, and partial alpha stays gray", () => {
    const source = rgba(255, 20, 50, 255, 80, 70, 60, 128, 0, 40, 200, 0)
    const savedSource = source.slice()
    const { pixels, hasTransparency } = alphaToInpaintingMaskPixels(source)
    assert.equal(hasTransparency, true)
    assert.deepEqual([...pixels], [0, 0, 0, 0, 127, 127, 127, 76, 255, 255, 255, 153])
    assert.deepEqual(source, savedSource, "conversion must not modify the source image")
})

test("opaque images are detected without affecting existing masks", () => {
    const { pixels, hasTransparency } = alphaToInpaintingMaskPixels(rgba(10, 20, 30, 255))
    assert.equal(hasTransparency, false)
    assert.deepEqual([...pixels], [0, 0, 0, 0])
})

test("inversion swaps white and black while retaining partial edge intensity", () => {
    const { pixels } = alphaToInpaintingMaskPixels(rgba(0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 255))
    const reversed = invertInpaintingMaskPixels(pixels)
    assert.deepEqual([...reversed], [0, 0, 0, 0, 128, 128, 128, 77, 255, 255, 255, 153])
    assert.deepEqual(invertInpaintingMaskPixels(reversed), pixels)
})

test("inversion also handles blank pixels and existing hand-painted masks", () => {
    const inverted = invertInpaintingMaskPixels(rgba(0, 0, 0, 0, 255, 255, 255, 153, 0, 0, 0, 255))
    assert.deepEqual([...inverted], [255, 255, 255, 153, 0, 0, 0, 0, 255, 255, 255, 153])
})

test("backend PNG export is opaque grayscale and treats unpainted pixels as black", () => {
    const converted = opaqueInpaintingMaskPixels(rgba(255, 255, 255, 153, 25, 25, 25, 15, 99, 99, 99, 0))
    assert.deepEqual([...converted], [255, 255, 255, 255, 25, 25, 25, 255, 0, 0, 0, 255])
})

test("malformed RGBA buffers are rejected", () => {
    for (const fn of [alphaToInpaintingMaskPixels, invertInpaintingMaskPixels, opaqueInpaintingMaskPixels]) {
        assert.throws(() => fn(null), TypeError)
        assert.throws(() => fn([1, 2, 3]), TypeError)
    }
})
