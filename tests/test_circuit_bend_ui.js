const assert = require("assert")
require("../ui/plugins/ui/circuit-bend.plugin.js")
require("../ui/plugins/ui/trajectory-output-steps.plugin.js")
require("../ui/plugins/ui/circuit-bend-grid.plugin.js")
const bend = globalThis.__flexDiffusionCircuitBendTest
const sparse = globalThis.__flexDiffusionSparseCaptureTest
assert.strictEqual(bend.validate({ enabled: true, pattern: "held", seed: 4 }, 20).pattern, "held")
for (const invalid of [{ noise: NaN }, { churn: Infinity }, { seed: -1 }, { every: 1.5 }, { start: 5, end: 4 }, { end: 21 }]) {
    assert.throws(() => bend.validate(invalid, 20))
}
assert.deepStrictEqual(sparse.outputCountSummary([10, 30], 30, true, 31), {
    denoisedCount: 2, solverCount: 2, finalCount: 1, totalOutputs: 5,
})
assert.strictEqual(sparse.outputCountSummary([30], 30, true, 31).totalOutputs, 1)
console.log("Circuit bend controls and native schedule counts passed")

const envelopeOptions = bend.validate({ window_unit: "percent", start: 25, end: 75, fade_in: 25, fade_out: 25 }, 20)
assert.deepStrictEqual(bend.resolveWindow(envelopeOptions, 31), [8, 24])
assert.deepStrictEqual([4, 5, 6, 8, 10, 12, 14, 15, 16].map((i) => Number(bend.envelope(envelopeOptions, i, 20).toFixed(6))), [0, 0, .4, 1, 1, 1, .4, 0, 0])
for (const invalid of [{ memory: 2 }, { start: 1.2 }, { window_unit: "percent", end: 101 }, { history_strength: 2 }]) assert.throws(() => bend.validate(invalid, 20))
const grid = globalThis.__flexDiffusionBendGridTest
const recipe = { schema: grid.SCHEMA, request: { prompt: "dinosaur", seed: 2, width: 512, height: 512,
    sampler_name: "euler", num_inference_steps: 20, num_outputs: 4,
    circuit_bend: bend.validate({ enabled: true, memory: .5 }, 20) },
    x: { key: "memory", values: [0, .5, 1] }, y: { key: "noise", values: [.75, 1.25] },
    seeds: [2, 3, 4, 5], baseline: true }
const unchanged = JSON.stringify(recipe)
const tasks = grid.buildGrid(recipe, "test")
assert.strictEqual(tasks.length, 28)
assert.strictEqual(new Set(tasks.map((task) => task.reqBody.bend_grid.cell_id)).size, 28)
assert(tasks.every((task) => task.reqBody.num_outputs === 1 && task.numOutputsTotal === 1 && task.batchCount === 1 && task.seed === task.reqBody.seed))
assert.strictEqual(tasks[0].reqBody.circuit_bend, null)
assert.strictEqual(tasks[1].reqBody.circuit_bend.memory, 0)
assert.strictEqual(tasks[6].reqBody.circuit_bend.memory, 1)
assert.strictEqual(tasks[7].seed, 3)
assert.strictEqual(JSON.stringify(recipe), unchanged)
assert.deepStrictEqual(grid.buildGrid(JSON.parse(unchanged), "test"), tasks)
for (const replacement of [{ x: { key: "memory", values: [0, 2] } }, { y: recipe.x }, { seeds: [2, 2] }, { seeds: [-1] }, { request: { ...recipe.request, sampler_name: "ddim" } }]) {
    assert.throws(() => grid.buildGrid({ ...recipe, ...replacement }, "test"))
}
const plms = { ...recipe, request: { ...recipe.request, sampler_name: "plms", circuit_bend: bend.validate({ enabled: true, mode: "plms_history" }, 20) }, x: { key: "history_strength", values: [0, .5, 1] }, y: { key: "fade_out", values: [0, 50] } }
assert.strictEqual(grid.buildGrid(plms, "p").length, 28)
assert.throws(() => grid.buildGrid({ ...plms, x: recipe.x }, "p"))
const final = { data: "final", intermediate_representation: "final", is_intermediate: false }
assert.strictEqual(grid.finalOutput([{ data: "checkpoint", is_intermediate: true }, final]), final)
assert.strictEqual(grid.finalOutput([final, { data: "checkpoint", is_intermediate: true }]), final)
console.log("Noise envelopes and reproducible grid contracts passed")

// Exercise the application's actual text metadata importer, without installing
// drag/drop handlers or requiring the rest of the application DOM.
const fs = require("fs")
const path = require("path")
for (const file of ["euler-noise-memory-grid.json", "plms-history-grid.json", "plms-seed-mixing-grid.json", "ddim-noise-grid.json", "plms-ddim-sampler-grid.json"]) {
    const example = JSON.parse(fs.readFileSync(path.join(__dirname, "../benchmarks", file), "utf8"))
    assert.strictEqual(grid.buildGrid(example, "example").length, 28)
    assert.strictEqual(example.request.width, 512)
    assert.deepStrictEqual(example.seeds, [2, 3, 4, 5])
}
const vm = require("vm")
const source = fs.readFileSync(path.join(__dirname, "../ui/media/js/dnd.js"), "utf8")
const savedBend = { enabled: true, pattern: "held", churn: 20, noise: 1.5, start: 5, end: 15, every: 2, seed: 123 }
const scope = { console: { log() {} }, window: { __flexDiffusionReadCircuitBend: () => savedBend } }
vm.createContext(scope)
vm.runInContext(source.slice(0, source.indexOf('document.addEventListener("drop"')), scope)
const task = scope.parseTaskFromText(`Prompt: dinosaur\nSampler: heun\nCircuit Bend: ${JSON.stringify(savedBend)}\n`)
assert.strictEqual(JSON.stringify(task.reqBody.circuit_bend), JSON.stringify(savedBend))
assert.strictEqual(vm.runInContext("TASK_MAPPING.circuit_bend.readUI()", scope), savedBend)
const gridTask = scope.parseTaskFromText(`Prompt: dinosaur\nBend Grid: {"run_id":"r","cell_id":"c"}\n`)
assert.strictEqual(gridTask.reqBody.bend_grid.cell_id, "c")
console.log("Circuit bend text metadata and copy-settings roundtrip passed")

for (const invalid of [{ eta: 1.1 }, { donor_seed: 1.5 }, { blend_period: 0 }, { blend_start: -.1 },
    { blend_curve: "random" }, { schedule: "invalid" }, { history_source: "unknown" }, { blend_target: "unknown" }]) {
    assert.throws(() => bend.validate(invalid, 16))
}
for (const [curve, expected] of [["progressive", [0, .25, .5, .75, 1]],
    ["oscillate", [0, .5, 1, .5, 0]], ["interlace", [0, 0, 1, 1, 0]]]) {
    const options = bend.validate({ enabled: true, mode: "plms_history", history_source: "seed",
        blend_curve: curve, blend_period: curve === "interlace" ? 2 : 4 }, 5)
    assert.deepStrictEqual([1, 2, 3, 4, 5].map(i => +bend.blendWeight(options, i, 5).toFixed(6)), expected)
}
const ddimRecipe = { ...recipe, request: { ...recipe.request, sampler_name: "ddim", num_inference_steps: 16,
    circuit_bend: bend.validate({ enabled: true, mode: "ddim", schedule: "exact", eta: .7, seed: 123 }, 16) },
    x: { key: "eta", values: [0, .5, 1] }, y: { key: "memory", values: [0, 1] } }
const ddimTasks = grid.buildGrid(ddimRecipe, "ddim")
assert.strictEqual(ddimTasks.length, 28)
assert.strictEqual(ddimTasks[0].reqBody.circuit_bend.schedule, "exact")
assert.strictEqual(ddimTasks[0].reqBody.circuit_bend.eta, 0)
const donorRecipe = { ...plms, request: { ...plms.request, num_inference_steps: 16,
    circuit_bend: bend.validate({ enabled: true, mode: "plms_history", schedule: "exact", history_source: "seed",
        donor_seed: 5, blend_target: "trajectory", blend_curve: "oscillate" }, 16) },
    x: { key: "blend_period", values: [2, 4, 8] }, y: { key: "blend_end", values: [.5, 1] } }
const donorTasks = grid.buildGrid(donorRecipe, "donor")
assert.strictEqual(donorTasks[0].reqBody.circuit_bend.history_strength, 0)
assert.strictEqual(donorTasks[1].reqBody.circuit_bend.donor_seed, 5)
assert.strictEqual(donorTasks[1].reqBody.circuit_bend.blend_target, "trajectory")
assert.deepStrictEqual(grid.buildGrid(JSON.parse(JSON.stringify(donorRecipe)), "donor"), donorTasks)
console.log("PLMS seed mixing, DDIM controls and exact-schedule grid references passed")
for(const invalid of [{mix_method:'unknown'},{mix_block:0},{mix_block:1.5},{mix_exchange:1.1},{mix_period:0},{mix_history:'unknown'},{mix_first:'heun'}]) {
    assert.throws(()=>bend.validate({mode:'sampler_mix',...invalid},16));
}
const hybridRecipe = {...ddimRecipe, request:{...ddimRecipe.request,sampler_name:'plms',
    circuit_bend:bend.validate({enabled:true,mode:'sampler_mix',schedule:'exact',eta:0,mix_method:'alternate'},16)},
    x:{key:'mix_block',values:[1,2,4]},y:{key:'eta',values:[0,.5]}};
const hybridTasks=grid.buildGrid(hybridRecipe,'hybrid');
assert.equal(hybridTasks.length,28);
assert.equal(hybridTasks[0].reqBody.circuit_bend.mode,'plms_history');
assert.equal(hybridTasks[0].reqBody.circuit_bend.history_strength,0);
assert.equal(hybridTasks[1].reqBody.circuit_bend.mode,'sampler_mix');
assert.throws(()=>grid.buildGrid({...hybridRecipe,x:{key:'mix_exchange',values:[0,.5]}},'bad-axis'));
assert.deepStrictEqual(grid.buildGrid(JSON.parse(JSON.stringify(hybridRecipe)),'hybrid'),hybridTasks);
console.log('Sampler mixing validation, grid axes and reference recipes passed');
