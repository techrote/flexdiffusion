// An inpainting mask uses white for pixels to regenerate and black for pixels to preserve.
// The editor uses translucent pixels for its overlay; export is opaque grayscale.
const INPAINTING_MASK_PREVIEW_OPACITY = 0.6

function checkRgbaPixels(pixels) {
    if (!pixels || typeof pixels.length !== "number" || pixels.length % 4 !== 0) {
        throw new TypeError("Expected an RGBA pixel buffer")
    }
}

function alphaToInpaintingMaskPixels(pixels) {
    checkRgbaPixels(pixels)
    const result = new Uint8ClampedArray(pixels.length)
    let hasTransparency = false
    for (let i = 0; i < pixels.length; i += 4) {
        const alpha = pixels[i + 3]
        const maskValue = 255 - alpha
        if (alpha < 255) hasTransparency = true
        result[i] = result[i + 1] = result[i + 2] = maskValue
        result[i + 3] = Math.round(maskValue * INPAINTING_MASK_PREVIEW_OPACITY)
    }
    return { pixels: result, hasTransparency }
}

function invertInpaintingMaskPixels(pixels) {
    checkRgbaPixels(pixels)
    const result = new Uint8ClampedArray(pixels.length)
    for (let i = 0; i < pixels.length; i += 4) {
        // An unpainted (transparent) editor pixel is a black / preserved mask pixel.
        const value =
            pixels[i + 3] === 0
                ? 0
                : Math.round(0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2])
        const inverted = 255 - value
        result[i] = result[i + 1] = result[i + 2] = inverted
        result[i + 3] = Math.round(inverted * INPAINTING_MASK_PREVIEW_OPACITY)
    }
    return result
}

function opaqueInpaintingMaskPixels(pixels) {
    checkRgbaPixels(pixels)
    const result = new Uint8ClampedArray(pixels.length)
    for (let i = 0; i < pixels.length; i += 4) {
        const visible = pixels[i + 3] !== 0
        result[i] = visible ? pixels[i] : 0
        result[i + 1] = visible ? pixels[i + 1] : 0
        result[i + 2] = visible ? pixels[i + 2] : 0
        result[i + 3] = 255
    }
    return result
}

// Browser: classic script globals. Node: lightweight tests without a DOM dependency.
if (typeof module !== "undefined" && module.exports) {
    module.exports = { alphaToInpaintingMaskPixels, invertInpaintingMaskPixels, opaqueInpaintingMaskPixels }
}
