"use strict"

// Real Chromium DOM/canvas smoke against the production workspace and Blend Lab
// scripts, isolated from the model/backend service. Uses the Chrome installed on
// ubuntu-latest; no npm dependencies or dev-server are required.
const assert = require("node:assert/strict")
const fs = require("node:fs")
const os = require("node:os")
const path = require("node:path")
const { spawnSync } = require("node:child_process")
const { pathToFileURL } = require("node:url")

const repo = path.resolve(__dirname, "..")
const files = [
    "ui/media/js/compositor.js",
    "ui/media/js/generation-workspaces.js",
    "ui/media/js/composition-lab.js",
]
const resources = files.map(file => '<script src="' + pathToFileURL(path.join(repo, file)).href + '"><\/script>').join("\n")
const css = pathToFileURL(path.join(repo, "ui/media/css/generation-workspaces.css")).href
const client = String.raw`
window.browserTest = { maskClears: 0, historyClears: 0, maskChange: 0, decodeCalls: 0, decoded: 0, decodeErrors: [] }
const originalDecode = Image.prototype.decode
Image.prototype.decode = function(...args) {
    ++window.browserTest.decodeCalls
    return originalDecode.apply(this, args).then(
        value => { ++window.browserTest.decoded; return value },
        error => { window.browserTest.decodeErrors.push(error.message); throw error },
    )
}
const imageInpainter = {
    width: 2, height: 1,
    layers: { drawing: { ctx: { clearRect() { ++window.browserTest.maskClears } } } },
    history: { clear() { ++window.browserTest.historyClears } },
}
const MAX_INIT_IMAGE_DIMENSION = 768
function setImageWidthHeight(w, h) { window.browserTest.dimensions = [w, h] }
function createTab(request) {
    const tab = document.createElement("button")
    tab.id = "tab-" + request.id
    tab.textContent = request.label
    document.body.appendChild(tab)
    const content = document.createElement("div")
    content.id = "tab-content-" + request.id
    content.innerHTML = '<div class="tab-content-inner">' + request.content + '</div>'
    document.body.appendChild(content)
    tab.addEventListener("click", () => {
        document.dispatchEvent(new CustomEvent("tabClick", { detail: { tab } }))
    })
}
document.getElementById("tab-main").addEventListener("click", function() {
    document.dispatchEvent(new CustomEvent("tabClick", { detail: { tab: this } }))
})
document.getElementById("enable_mask").addEventListener("change", () => ++window.browserTest.maskChange)
HTMLAnchorElement.prototype.click = function() {
    window.browserTest.download = { name: this.download, href: this.href }
}
`
const smoke = String.raw`
(async () => {
    const assert = (condition, text) => { if (!condition) throw new Error(text) }
    const q = name => document.getElementById("blend-lab-" + name)
    const waitFor = async (test, name) => {
        for (let i = 0; i < 800; i++) {
            if (test()) return
            await new Promise(resolve => setTimeout(resolve, 100))
        }
        throw new Error("Timed out: " + name + " · " + q("status").textContent + " · " + JSON.stringify(window.browserTest))
    }
    const sample = rgba => {
        const canvas = document.createElement("canvas")
        canvas.width = 2
        canvas.height = 1
        const ctx = canvas.getContext("2d")
        const img = ctx.createImageData(2, 1)
        img.data.set(rgba)
        ctx.putImageData(img, 0, 0)
        return new Promise(resolve => canvas.toBlob(resolve, "image/png"))
    }
    const load = async (id, blob) => {
        const input = q(id)
        const transfer = new DataTransfer()
        transfer.items.add(new File([blob], id + ".png", { type: "image/png" }))
        input.files = transfer.files
        input.dispatchEvent(new Event("change", { bubbles: true }))
    }
    const pixel = i => [...q("canvas").getContext("2d").getImageData(i, 0, 1, 1).data]
    const pA = document.getElementById("editor"), pB = document.getElementById("preview")
    const queue = document.getElementById("task-queue-marker")
    assert(window.__flexDiffusionGenerationWorkspace === "vanilla", "initial Generate mode")
    assert(getComputedStyle(document.getElementById("bend_grid_row")).display === "none", "experiment grid leaked in vanilla CSS")
    document.getElementById("tab-sampler-lab").click()
    assert(window.__flexDiffusionGenerationWorkspace === "sampler", "sampler mode not selected")
    assert(pA.parentElement.id === "generation-sampler-form" && pB.parentElement.id === "generation-sampler-form", "editor and preview not moved")
    document.getElementById("tab-main").click()
    assert(window.__flexDiffusionGenerationWorkspace === "vanilla", "vanilla mode not restored")
    assert(pA.parentElement.id === "tab-content-main" && pB.parentElement.id === "tab-content-main", "original editor identity lost")
    assert(document.getElementById("task-queue-marker") === queue, "queue element replaced")

    const a = await sample([20,30,40,255,10,15,30,255])
    const b = await sample([200,100,50,255,170,80,30,255])
    const mask = await sample([0,0,0,255,255,255,255,255])
    await load("bottom", a)
    await load("top", b)
    await waitFor(() => q("canvas").width === 2 && !q("save").disabled, "A+B raster")
    assert(pixel(0)[0] === 200 && pixel(1)[0] === 170, "normal image blend")
    await load("coverage", mask)
    await waitFor(() => pixel(0)[0] === 20 && pixel(1)[0] === 170, "coverage map")
    q("invert").checked = true
    q("invert").dispatchEvent(new Event("change"))
    assert(pixel(0)[0] === 200 && pixel(1)[0] === 10, "coverage invert")

    q("invert").checked = false
    q("invert").dispatchEvent(new Event("change"))
    q("save").click()
    await waitFor(() => !!window.browserTest.download, "PNG save")
    assert(window.browserTest.download.name === "flexdiffusion-blend.png", "wrong PNG filename")
    const initial = document.getElementById("init_image_preview")
    const toggled = document.getElementById("enable_mask")
    assert(toggled.checked, "mask should begin enabled")
    q("use-init").click()
    assert(!toggled.checked && window.browserTest.maskChange === 1, "mask toggle not cleared")
    assert(window.browserTest.maskClears === 1 && window.browserTest.historyClears === 1, "old mask raster/history not cleared")
    assert(initial.src.startsWith("data:image/png;base64,"), "PNG img2img handoff failed")
    assert(window.browserTest.dimensions.join("x") === "2x1", "handoff dimensions incorrect")
    assert(window.__flexDiffusionGenerationWorkspace === "vanilla", "handoff did not return to Generate")
    document.getElementById("result").textContent = "PASS-BLEND-BROWSER"
})().catch(err => {
    document.getElementById("result").textContent = "FAIL-BLEND-BROWSER: " + err.message
})
`
const html = `<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="${css}"></head>
<body>
<div id="tab-content-main">
  <div id="editor"><div id="bend_grid_row">grid</div><div id="circuit_bend_row">circuit</div></div>
  <div id="preview"><div id="task-queue-marker">queue</div></div>
</div>
<button id="tab-main">Generate</button><button id="tab-settings">Settings</button>
<input id="enable_mask" type="checkbox" checked><img id="init_image_preview">
<output id="result">PENDING</output>
<script>${client}<\/script>
${resources}
<script>${smoke}<\/script>
</body></html>`
const candidates = [process.env.CHROME_BIN, "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"].filter(Boolean)
const browser = candidates.find(name => {
    const x = spawnSync("which", [name], { encoding: "utf8" })
    return x.status === 0
})
assert.ok(browser, "Chromium/Chrome is required to run the browser smoke; do not silently skip")
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "flex-blend-browser-"))
try {
    const fixture = path.join(tmp, "fixture.html")
    fs.writeFileSync(fixture, html)
    const command = [
        "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
        "--disable-extensions", "--disable-background-networking", "--no-first-run",
        "--allow-file-access-from-files", "--virtual-time-budget=95000", "--dump-dom",
        pathToFileURL(fixture).href,
    ]
    const run = spawnSync(browser, command, { encoding: "utf8", timeout: 45000, maxBuffer: 4 * 1024 * 1024 })
    const match = run.stdout?.match(/<output id="result">([^<]*)<\/output>/)
    if (run.error || run.status !== 0 || !match || !match[1].startsWith("PASS-BLEND-BROWSER")) {
        console.error("Chrome result:", match?.[1] || "(no result marker)")
        console.error("Chrome exit:", run.status, run.error?.message || "")
        console.error("stderr (tail):", (run.stderr || "").slice(-1200))
        process.exitCode = 1
    } else {
        console.log("Real Chromium: workspace DOM identity, vanilla CSS isolation, decoded PNGs, mask inversion, export and img2img handoff PASS")
    }
} finally {
    fs.rmSync(tmp, { recursive: true, force: true })
}
