"use strict"
// Drive the *actual* image-editor scripts in real Chrome via the built-in CDP
// WebSocket. Unlike --dump-dom with a virtual-time-budget, this waits for
// asynchronous FileReader/Image callbacks to complete rather than sampling
// the page before its PASS/FAIL result has settled. No npm dependencies.
const { spawn, spawnSync } = require("node:child_process")
const { existsSync, mkdtempSync, readFileSync, rmSync } = require("node:fs")
const os = require("node:os")
const path = require("node:path")
const { pathToFileURL } = require("node:url")

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const assertBefore = Date.now() + 65000
const options = [process.env.CHROME_BIN, "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]
let executable
for (const binary of options.filter(Boolean)) {
    const probe = spawnSync(binary, ["--version"], {encoding: "utf8", timeout: 5000})
    if (!probe.error && probe.status === 0) {
        executable = binary
        break
    }
}
if (!executable) {
    console.error("FAIL: no Chrome/Chromium executable found")
    process.exit(1)
}
const profile = mkdtempSync(path.join(os.tmpdir(), "flexdiffusion-mask-cdp-"))
const uri = pathToFileURL(path.resolve(__dirname, "../tests/inpainting-browser.html")).href
let child, socket, stderr = ""
const exceptions = []
async function waitForTarget() {
    const activePort = path.join(profile, "DevToolsActivePort")
    let port
    while (Date.now() < assertBefore) {
        if (child.exitCode !== null) throw new Error("Chrome exited before DevTools was ready")
        if (existsSync(activePort)) {
            port = Number(readFileSync(activePort, "utf8").split(/\r?\n/)[0])
            if (port) break
        }
        await delay(100)
    }
    if (!port) throw new Error("Chrome DevTools did not start")
    const url = "http://127.0.0.1:" + port + "/json/list"
    while (Date.now() < assertBefore) {
        const targets = await fetch(url).then((r) => r.json()).catch(() => [])
        const target = targets.find((t) => t.type === "page" && t.url.includes("inpainting-browser.html"))
        if (target && target.webSocketDebuggerUrl) return target.webSocketDebuggerUrl
        await delay(100)
    }
    throw new Error("Browser did not open the test page")
}
async function openCdp(url) {
    socket = new WebSocket(url)
    await new Promise((resolve, reject) => {
        socket.addEventListener("open", resolve, {once: true})
        socket.addEventListener("error", reject, {once: true})
    })
    const pending = new Map()
    let nextId = 0
    socket.addEventListener("message", (event) => {
        const data = JSON.parse(event.data)
        if (data.method === "Runtime.exceptionThrown") {
            exceptions.push(data.params?.exceptionDetails?.text || "uncaught script exception")
        }
        if (data.id && pending.has(data.id)) {
            const resolve = pending.get(data.id)
            pending.delete(data.id)
            resolve(data)
        }
    })
    const call = (method, params = {}) => new Promise((resolve) => {
        const id = ++nextId
        pending.set(id, resolve)
        socket.send(JSON.stringify({id, method, params}))
    })
    await call("Runtime.enable")
    return call
}
async function main() {
    const args = [
        "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
        "--disable-background-networking", "--no-first-run", "--no-default-browser-check",
        "--allow-file-access-from-files", "--remote-debugging-port=0",
        "--user-data-dir=" + profile, uri,
    ]
    child = spawn(executable, args, {stdio: ["ignore", "ignore", "pipe"]})
    child.stderr.on("data", (chunk) => { stderr = (stderr + chunk.toString()).slice(-1800) })
    const call = await openCdp(await waitForTarget())
    while (Date.now() < assertBefore) {
        if (child.exitCode !== null) throw new Error("Chrome closed before test completion")
        const response = await call("Runtime.evaluate", {
            expression: "document.getElementById('result')?.textContent || 'RUNNING'",
            returnByValue: true,
        })
        const result = response.result?.result?.value || "RUNNING"
        if (result.startsWith("PASS:")) {
            console.log("Chrome/Chromium:", executable)
            console.log(result)
            return
        }
        if (result.startsWith("FAIL:")) throw new Error(result)
        await delay(120)
    }
    throw new Error("Browser fixture did not finish before deadline" + (exceptions.length ? ": " + exceptions.join("; ") : ""))
}
main()
    .catch((error) => {
        console.error("FAIL: Chromium inpainting editor regression:", error.message)
        console.error("Stderr:", stderr)
        process.exitCode = 1
    })
    .finally(async () => {
        if (socket) socket.close()
        if (child && child.exitCode === null) child.kill()
        await delay(250)
        rmSync(profile, {recursive: true, force: true})
    })
