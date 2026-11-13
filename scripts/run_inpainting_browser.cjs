"use strict"
// Launch a real browser against the actual image-editor scripts; no npm install.
const { spawnSync } = require("node:child_process")
const { mkdtempSync, rmSync } = require("node:fs")
const os = require("node:os")
const path = require("node:path")
const { pathToFileURL } = require("node:url")

const options = [process.env.CHROME_BIN, "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]
let executable
for (const binary of options.filter(Boolean)) {
    const probe = spawnSync(binary, ["--version"], { encoding: "utf8", timeout: 5000 })
    if (!probe.error && probe.status === 0) {
        executable = binary
        break
    }
}
if (!executable) {
    console.error("FAIL: no real Chrome/Chromium executable was found")
    process.exit(1)
}
const profile = mkdtempSync(path.join(os.tmpdir(), "flexdiffusion-mask-chrome-"))
try {
    const uri = pathToFileURL(path.resolve(__dirname, "../tests/inpainting-browser.html")).href
    const args = [
        "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
        "--no-first-run", "--allow-file-access-from-files",
        "--disable-background-networking", "--virtual-time-budget=4500",
        "--user-data-dir=" + profile, "--dump-dom", uri,
    ]
    const run = spawnSync(executable, args, {encoding: "utf8", timeout: 90000, maxBuffer: 10 * 1024 * 1024})
    const result = /<pre id="result">([^<]*)<\/pre>/.exec(run.stdout || "")
    if (run.error || run.status !== 0 || !result || !result[1].startsWith("PASS:")) {
        console.error("FAIL: Chromium inpainting editor regression", run.error || "", "exit", run.status)
        console.error("Result:", result?.[1] || "(no result)")
        console.error("Stderr:", (run.stderr || "").slice(-3500))
        process.exitCode = 1
    } else {
        console.log("Chromium:", executable)
        console.log(result[1])
    }
} finally {
    rmSync(profile, {recursive: true, force: true})
}
