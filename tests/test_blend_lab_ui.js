"use strict"

// Dependency-free integration harness for the actual Blend Lab UI script.
// Exercises DOM event wiring, asynchronous image decoding, raster handoff,
// img2img mask reset and PNG download without a generation server.
const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")
const compositor = require("../ui/media/js/compositor.js")

const nodes = new Map()
const objectUrls = new Map()
const downloads = []
let serial = 0
let clearedMask = 0
let clearedHistory = 0
let maskChanges = 0
let dimensions = null

class Element {
    constructor(id = "") {
        this.id = id
        this.children = []
        this.events = new Map()
        this.files = []
        this.value = ""
        this.checked = false
        this.disabled = false
        this.textContent = ""
        this.style = {}
    }
    addEventListener(type, fn) {
        if (!this.events.has(type)) this.events.set(type, [])
        this.events.get(type).push(fn)
    }
    emit(type) {
        return Promise.all((this.events.get(type) || []).map(fn => fn({ type, target: this })))
    }
    dispatchEvent(event) {
        if (this.id === "enable_mask" && event.type === "change") ++maskChanges
        return this.emit(event.type)
    }
    appendChild(child) { this.children.push(child); return child }
    before() {}
    click() {
        this.clicked = (this.clicked || 0) + 1
        if (this.href) downloads.push({ href: this.href, download: this.download })
        this.emit("click")
    }
}
class Canvas extends Element {
    constructor() { super(); this.width = 0; this.height = 0; this.pixels = null; this.image = null }
    getContext() {
        const canvas = this
        return {
            imageSmoothingEnabled: true,
            imageSmoothingQuality: "high",
            drawImage(image) { canvas.image = image },
            getImageData() {
                if (!canvas.image?.pixels) throw new Error("oversized image was passed to raster")
                const output = new Uint8ClampedArray(canvas.width * canvas.height * 4)
                const image = canvas.image
                for (let y = 0; y < canvas.height; y++) {
                    for (let x = 0; x < canvas.width; x++) {
                        const srcX = Math.min(image.naturalWidth - 1, Math.floor(x * image.naturalWidth / canvas.width))
                        const srcY = Math.min(image.naturalHeight - 1, Math.floor(y * image.naturalHeight / canvas.height))
                        const src = (srcY * image.naturalWidth + srcX) * 4
                        output.set(image.pixels.subarray(src, src + 4), (y * canvas.width + x) * 4)
                    }
                }
                return { data: output }
            },
            createImageData(width, height) { return { data: new Uint8ClampedArray(width * height * 4) } },
            putImageData(imageData) { canvas.pixels = imageData.data.slice() },
        }
    }
    toBlob(callback, type) {
        assert.equal(type, "image/png")
        callback({ kind: "png", pixels: this.pixels.slice(), type })
    }
    toDataURL(type) {
        assert.equal(type, "image/png")
        return "data:image/png;base64," + Buffer.from(this.pixels).toString("base64")
    }
}
class ImageMock {
    set src(url) {
        const file = objectUrls.get(url)
        if (!file) throw new Error("unregistered blob url")
        this.naturalWidth = file.width
        this.naturalHeight = file.height
        this.pixels = file.pixels
        this.file = file
    }
    decode() {
        if (this.file.reject) return Promise.reject(new Error("decode failure"))
        if (this.file.hold) return new Promise(resolve => { this.file.release = resolve })
        return Promise.resolve()
    }
}
function get(id) {
    if (!nodes.has(id)) {
        const node = id === "blend-lab-canvas" ? new Canvas() : new Element(id)
        node.id = id
        nodes.set(id, node)
    }
    return nodes.get(id)
}
const document = {
    getElementById(id) { return nodes.get(id) || null },
    createElement(tag) { return tag === "canvas" ? new Canvas() : new Element() },
}
const URL = {
    createObjectURL(file) {
        const url = "blob:fixture/" + (++serial)
        objectUrls.set(url, file)
        return url
    },
    revokeObjectURL(url) { objectUrls.delete(url) },
}
const root = { FlexDiffusionCompositor: compositor, setTimeout() {} }
const mode = { normal: "normal", opacity: "100", "mask-channel": "luminance" }
for (const id of ["tab-settings", "tab-main", "init_image_preview", "enable_mask"]) get(id)
for (const [id, val] of Object.entries(mode)) get("blend-lab-" + id).value = val
get("enable_mask").checked = true
const imageInpainter = {
    width: 1,
    height: 1,
    layers: { drawing: { ctx: { clearRect() { ++clearedMask } } } },
    history: { clear() { ++clearedHistory } },
}
function createTab(request) {
    assert.equal(request.id, "blend-lab")
    for (const [, id] of request.content.matchAll(/id="(blend-lab-[a-z-]+)"/g)) get(id)
    get("blend-lab-save").disabled = true
    get("blend-lab-use-init").disabled = true
    get("tab-" + request.id)
}
vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "../ui/media/js/composition-lab.js"), "utf8"),
    { window: root, document, createTab, Image: ImageMock, URL, Event: class Event { constructor(type) { this.type = type } }, imageInpainter, setImageWidthHeight(w, h) { dimensions = [w, h] }, MAX_INIT_IMAGE_DIMENSION: 768 },
    { filename: "composition-lab.js" }
)
const q = (id) => get("blend-lab-" + id)
const file = (name, rgba, width = 1, height = 1) => ({
    name, type: "image/png", width, height,
    pixels: rgba ? Uint8ClampedArray.from(rgba) : null,
})
async function select(id, f) {
    q(id).files = f ? [f] : []
    await q(id).emit("change")
}
async function run() {
    assert.equal(q("mode").children.length, 5, "mode option groups")
    assert.ok(q("save").disabled && q("use-init").disabled, "no early export")
    await select("bottom", file("A.png", [10, 20, 30, 255]))
    await select("top", file("B.png", [200, 150, 100, 255]))
    assert.deepEqual([...q("canvas").pixels], [200, 150, 100, 255])
    assert.equal(q("canvas").width, 1)
    assert.equal(q("save").disabled, false)
    await select("coverage", file("black.png", [0, 0, 0, 255]))
    assert.deepEqual([...q("canvas").pixels], [10, 20, 30, 255], "black coverage excludes upper image")
    q("invert").checked = true
    await q("invert").emit("change")
    assert.deepEqual([...q("canvas").pixels], [200, 150, 100, 255], "mask inversion")
    q("invert").checked = false
    q("mask-channel").value = "alpha"
    await q("mask-channel").emit("change")
    assert.deepEqual([...q("canvas").pixels], [200, 150, 100, 255], "alpha only ignores black RGB")
    await select("coverage", file("transparent.png", [255, 255, 255, 0]))
    assert.deepEqual([...q("canvas").pixels], [10, 20, 30, 255], "transparent alpha map excludes upper")
    await q("clear-map").emit("click")
    assert.deepEqual([...q("canvas").pixels], [200, 150, 100, 255], "clear map restores unmasked B")
    await q("save").emit("click")
    assert.equal(downloads.length, 1, "PNG export click")
    assert.equal(downloads[0].download, "flexdiffusion-blend.png")
    assert.equal(get("enable_mask").checked, true)
    await q("use-init").emit("click")
    assert.equal(get("enable_mask").checked, false, "old inpainting mask disabled")
    assert.equal(maskChanges, 1, "mask change event triggered")
    assert.equal(clearedMask, 1, "old mask strokes removed")
    assert.equal(clearedHistory, 1, "old mask history removed")
    assert.deepEqual(dimensions, [1, 1], "valid handoff dimensions applied")
    assert.match(get("init_image_preview").src, /^data:image\/png;base64,/)
    assert.equal(get("tab-main").clicked, 1, "return to Generate tab")

    // No decoded input, including B or map, may exceed the cap even when A is tiny.
    await select("top", file("huge.png", null, 5000, 5000))
    assert.equal(q("save").disabled, true, "invalid loading disables stale export")
    assert.match(q("status").textContent, /8.megapixel safety limit/i, "huge upper image rejected before raster")
    await select("top", file("valid.png", [200, 150, 100, 255]))
    assert.equal(q("save").disabled, false, "successful replacement recovers")
    await select("coverage", file("huge-map.png", null, 5000, 5000))
    assert.match(q("status").textContent, /8.megapixel safety limit/i, "huge map rejected")
    assert.equal(q("save").disabled, true)
    await q("clear-map").emit("click")
    assert.equal(q("save").disabled, false)

    // A cancelled or superseded async decode must not replace a later choice.
    const stale = file("stale.png", [50, 50, 50, 255])
    stale.hold = true
    q("top").files = [stale]
    const pending = q("top").emit("change")
    await select("top", file("fresh.png", [44, 55, 66, 255]))
    stale.release()
    await pending
    assert.deepEqual([...q("canvas").pixels], [44, 55, 66, 255], "last selection wins")
    assert.equal(objectUrls.size, 1, "only the delayed PNG download URL remains")
    assert.equal([...objectUrls.values()][0].kind, "png", "all image decode URLs revoked")
    console.log("Blend Lab UI: image decode, all tab controls, masks, export, img2img, size guards, stale async decode passed")
}
run().catch(error => { console.error(error); process.exitCode = 1 })
