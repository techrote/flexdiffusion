// An inpainting mask uses white for pixels to regenerate and black for pixels
// to preserve. Store coverage directly in canvas alpha, with white RGB, so
// browsers cannot quantize low-opacity grayscale RGB during PNG roundtrips.
// CSS provides the 0.6 visual overlay opacity without corrupting mask values.

function checkRgbaPixels(pixels) {
    if (!pixels || typeof pixels.length !== "number" || pixels.length % 4 !== 0) {
        throw new TypeError("Expected an RGBA pixel buffer")
    }
}

function maskCoverageAt(pixels, i) {
    if (pixels[i + 3] === 0) return 0
    // White paint is mask coverage; imported grayscale and legacy colored
    // pixels retain their luminance, modulated by actual alpha.
    const luminance = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2]
    return Math.round((luminance * pixels[i + 3]) / 255)
}

function setPreviewCoverage(pixels, i, value) {
    // White overlaid in CSS is much easier to interpret than a dark gray fill.
    const channel = value === 0 ? 0 : 255
    pixels[i] = pixels[i + 1] = pixels[i + 2] = channel
    pixels[i + 3] = value
}

function alphaToInpaintingMaskPixels(pixels) {
    checkRgbaPixels(pixels)
    const result = new Uint8ClampedArray(pixels.length)
    let hasTransparency = false
    for (let i = 0; i < pixels.length; i += 4) {
        const alpha = pixels[i + 3]
        const coverage = 255 - alpha
        if (alpha < 255) hasTransparency = true
        setPreviewCoverage(result, i, coverage)
    }
    return { pixels: result, hasTransparency }
}

// A mask-file or persisted opaque PNG is a coverage image, not an opaque
// drawing overlay. Convert its grayscale/alpha to the editor representation.
function importInpaintingMaskPixels(pixels) {
    checkRgbaPixels(pixels)
    const result = new Uint8ClampedArray(pixels.length)
    for (let i = 0; i < pixels.length; i += 4) {
        setPreviewCoverage(result, i, maskCoverageAt(pixels, i))
    }
    return result
}

function invertInpaintingMaskPixels(pixels) {
    checkRgbaPixels(pixels)
    const result = new Uint8ClampedArray(pixels.length)
    for (let i = 0; i < pixels.length; i += 4) {
        setPreviewCoverage(result, i, 255 - maskCoverageAt(pixels, i))
    }
    return result
}

function opaqueInpaintingMaskPixels(pixels) {
    checkRgbaPixels(pixels)
    const result = new Uint8ClampedArray(pixels.length)
    for (let i = 0; i < pixels.length; i += 4) {
        const value = maskCoverageAt(pixels, i)
        result[i] = result[i + 1] = result[i + 2] = value
        result[i + 3] = 255
    }
    return result
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        alphaToInpaintingMaskPixels,
        importInpaintingMaskPixels,
        invertInpaintingMaskPixels,
        opaqueInpaintingMaskPixels,
    }
}
