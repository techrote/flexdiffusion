"use strict"
const assert = require("node:assert/strict")
const fs = require("node:fs")
const vm = require("node:vm")
const path = require("node:path")

const nodes = {}
const moves = []
function element(id) {
    const target = {
        id,
        children: [],
        append(...items) {
            this.children = items
            moves.push([this.id, items.map(item => item.id)])
        },
        before(other) { moves.push(["before " + this.id, [other.id]]) },
    }
    nodes[id] = target
    return target
}

for (const id of [
    "tab-content-main", "editor", "preview", "tab-settings", "tab-main",
    "generation-sampler-form", "generation-sampler-help",
]) element(id)
const listeners = new Map()
const window = {}
const document = {
    getElementById(id) { return nodes[id] || null },
    addEventListener(event, callback) { listeners.set(event, callback) },
}
const created = []
function createTab(request) {
    created.push(request)
    element("tab-" + request.id)
}
const context = { window, document, createTab }
vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "../ui/media/js/generation-workspaces.js"), "utf8"),
    context,
    { filename: "generation-workspaces.js" }
)

assert.equal(created.length, 1)
assert.equal(created[0].id, "sampler-lab")
assert.equal(window.__flexDiffusionGenerationWorkspace, "vanilla")
assert.ok(listeners.has("tabClick"))
const click = (id) => listeners.get("tabClick")({ detail: { tab: { id } } })
click("tab-sampler-lab")
assert.equal(window.__flexDiffusionGenerationWorkspace, "sampler")
assert.deepEqual(moves.at(-1), ["generation-sampler-form", ["editor", "preview"]])
click("tab-settings")
assert.equal(window.__flexDiffusionGenerationWorkspace, "sampler") // no form reparent to settings
click("tab-main")
assert.equal(window.__flexDiffusionGenerationWorkspace, "vanilla")
assert.deepEqual(moves.at(-1), ["tab-content-main", ["editor", "preview"]])

// The sampler plugin must explicitly gate request injection in ordinary Generate.
const plugin = fs.readFileSync(path.join(__dirname, "../ui/plugins/ui/circuit-bend.plugin.js"), "utf8")
assert.match(plugin, /root\.__flexDiffusionGenerationWorkspace !== "vanilla"\) task\.reqBody\.circuit_bend = read\(\)/)
console.log("Generate workspaces: shared editor reparenting, tab routing, vanilla circuit-bend isolation passed")
