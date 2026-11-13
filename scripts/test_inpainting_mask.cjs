const test = require("node:test")
const assert = require("node:assert/strict")
const {
    alphaToInpaintingMaskPixels,
    importInpaintingMaskPixels,
    invertInpaintingMaskPixels,
    opaqueInpaintingMaskPixels,
} = require("../ui/media/js/inpainting-mask.js")

const rgba = (...values) => Uint8ClampedArray.from(values)

test("transparent PNG alpha becomes white, opaque becomes black, and partial alpha stays gray", () => {
    const source = rgba(255, 20, 50, 255, 80, 70, 60, 128, 0, 40, 200, 0)
    const savedSource = source.slice()
    const { pixels, hasTransparency } = alphaToInpaintingMaskPixels(source)
    assert.equal(hasTransparency, true)
    assert.deepEqual([...pixels], [0, 0, 0, 0, 255, 255, 255, 127, 255, 255, 255, 255])
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
    assert.deepEqual([...reversed], [0, 0, 0, 0, 255, 255, 255, 128, 255, 255, 255, 255])
    assert.deepEqual(invertInpaintingMaskPixels(reversed), pixels)
})

test("inversion also handles blank pixels and existing hand-painted masks", () => {
    const inverted = invertInpaintingMaskPixels(rgba(0, 0, 0, 0, 255, 255, 255, 153, 0, 0, 0, 255))
    assert.deepEqual([...inverted], [255, 255, 255, 255, 255, 255, 255, 102, 255, 255, 255, 255])
})

test("backend PNG export is opaque grayscale and treats unpainted pixels as black", () => {
    const converted = opaqueInpaintingMaskPixels(rgba(255, 255, 255, 153, 25, 25, 25, 15, 99, 99, 99, 0))
    assert.deepEqual([...converted], [153, 153, 153, 255, 1, 1, 1, 255, 0, 0, 0, 255])
})

test("malformed RGBA buffers are rejected", () => {
    for (const fn of [alphaToInpaintingMaskPixels, invertInpaintingMaskPixels, opaqueInpaintingMaskPixels]) {
        assert.throws(() => fn(null), TypeError)
        assert.throws(() => fn([1, 2, 3]), TypeError)
    }
})

test("imported opaque PNG masks appear translucent and roundtrip without changing coverage", () => {
    const exported = rgba(0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255)
    const displayed = importInpaintingMaskPixels(exported)
    assert.deepEqual([...displayed], [0, 0, 0, 0, 255, 255, 255, 128, 255, 255, 255, 255])
    assert.deepEqual(opaqueInpaintingMaskPixels(displayed), exported)
})

test("import ignores hidden RGB under transparent alpha and scales partially transparent masks", () => {
    const result = importInpaintingMaskPixels(rgba(
        255, 255, 255, 0,
        255, 255, 255, 128,
        128, 128, 128, 255,
    ))
    assert.deepEqual([...result], [0, 0, 0, 0, 255, 255, 255, 128, 255, 255, 255, 128])
})

test("export really is opaque grayscale even for arbitrary painted RGB colors", () => {
    const exported = opaqueInpaintingMaskPixels(rgba(255, 0, 0, 153, 0, 255, 0, 153, 0, 0, 255, 153))
    assert.deepEqual([...exported], [46, 46, 46, 255, 90, 90, 90, 255, 17, 17, 17, 255])
})

test("partial-alpha extraction survives exported PNG and image re-import", () => {
    const alphaInput = rgba(234, 22, 2, 255, 89, 44, 0, 128, 255, 255, 255, 0)
    const expected = rgba(0, 0, 0, 255, 127, 127, 127, 255, 255, 255, 255, 255)
    const visible = alphaToInpaintingMaskPixels(alphaInput)
    assert.equal(visible.hasTransparency, true)
    const exported = opaqueInpaintingMaskPixels(visible.pixels)
    assert.deepEqual(exported, expected)
    assert.deepEqual(opaqueInpaintingMaskPixels(importInpaintingMaskPixels(exported)), expected)
})

test("import validates buffer shape like the other conversion functions", () => {
    assert.throws(() => importInpaintingMaskPixels(null), TypeError)
    assert.throws(() => importInpaintingMaskPixels([1, 2, 3]), TypeError)
})
