"use strict"
// Exercise the real EditorHistory implementation without an external DOM package.
const test = require("node:test")
const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")

const src = fs.readFileSync(path.join(__dirname, "../ui/media/js/image-editor.js"), "utf8")
const start = src.indexOf("class EditorHistory {")
const end = src.indexOf("\nclass ImageEditor {", start)
assert.ok(start >= 0 && end > start)
const EditorHistory = vm.runInNewContext(src.slice(start, end) + "\nEditorHistory", {
    IMAGE_EDITOR_ACTIONS: [],
    IMAGE_EDITOR_TOOLS: [],
})

function createEditor(initial = [33, 33, 33, 153]) {
    let pixels = Uint8ClampedArray.from(initial)
    const ctx = {
        getImageData: () => ({ data: pixels.slice() }),
        putImageData: (data) => { pixels = data.data.slice() },
        clearRect: () => { pixels = new Uint8ClampedArray(4) },
    }
    const editor = {
        width: 1,
        height: 1,
        layers: { drawing: { ctx } },
        setBrush: () => {},
        cancelPendingMaskLoad: () => {},
    }
    editor.history = new EditorHistory(editor)
    editor.history.clear()
    return {
        editor,
        read: () => [...pixels],
        write: (values) => { pixels = Uint8ClampedArray.from(values) },
    }
}

test("undo restores a previously saved/imported nonblank baseline, redo restores raster", () => {
    const { editor, read, write } = createEditor([17, 17, 17, 77])
    write([255, 255, 255, 153])
    editor.history.pushRaster()
    editor.history.undo()
    assert.deepEqual(read(), [17, 17, 17, 77])
    editor.history.redo()
    assert.deepEqual(read(), [255, 255, 255, 153])
})

test("new edit after undo truncates redo and preserves baseline", () => {
    const { editor, read, write } = createEditor([17, 17, 17, 77])
    write([100, 100, 100, 100])
    editor.history.pushRaster()
    write([200, 200, 200, 150])
    editor.history.pushRaster()
    editor.history.undo()
    write([35, 35, 35, 40])
    editor.history.pushRaster()
    editor.history.redo()
    assert.deepEqual(read(), [35, 35, 35, 40])
    editor.history.undo()
    editor.history.undo()
    assert.deepEqual(read(), [17, 17, 17, 77])
})

test("history reset after undo resets rewind index and captures new baseline", () => {
    const { editor, read, write } = createEditor([1, 1, 1, 1])
    write([255, 255, 255, 153])
    editor.history.pushRaster()
    editor.history.undo()
    write([10, 10, 10, 50])
    editor.history.clear()
    assert.equal(editor.history.rewind_index, 0)
    write([20, 20, 20, 100])
    editor.history.pushRaster()
    editor.history.undo()
    assert.deepEqual(read(), [10, 10, 10, 50])
})

test("periodic history snapshots retain deterministic bitmap replay", () => {
    const { editor, read, write } = createEditor([4, 4, 4, 4])
    for (let i = 1; i <= 25; i++) {
        write([i, i, i, i])
        editor.history.pushRaster()
    }
    for (let i = 24; i >= 0; i--) {
        editor.history.undo()
        assert.deepEqual(read(), i ? [i, i, i, i] : [4, 4, 4, 4])
    }
    for (let i = 1; i <= 25; i++) {
        editor.history.redo()
        assert.deepEqual(read(), [i, i, i, i])
    }
})
