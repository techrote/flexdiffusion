/* Distinct generation workspaces without cloning the Easy Diffusion form.
 * The shared editor/gallery retain their DOM identities, listeners and queue.
 * Sampler Lab is opt-in; vanilla Generate never submits a circuit-bend payload.
 */
;(function (root) {
    "use strict"
    if (typeof document === "undefined" || typeof createTab !== "function") return

    const standard = document.getElementById("tab-content-main")
    const editor = document.getElementById("editor")
    const preview = document.getElementById("preview")
    if (!standard || !editor || !preview) return
    root.__flexDiffusionGenerationWorkspace = "vanilla"

    createTab({
        id: "sampler-lab",
        label: "Sampler Lab",
        icon: "fa-shuffle",
        content: `
            <div class="generation-workspace-heading">
                <h2>Sampler / Trajectory Lab</h2>
                <p>Experimental sampler alternation, coupled trajectories, checkpoint capture and parameter grids.</p>
                <p>The generator, image queue, prompt, seed, model, batching and image settings are shared with the ordinary Generate tab. Experimental circuit-bend requests are only submitted from this workspace.</p>
                <small id="generation-sampler-help">Expand <b>Image Settings</b> and <b>Circuit bend → Bend controls</b> to configure PLMS/DDIM experiments. Native sampler switching requires the classic backend.</small>
            </div>
            <div id="generation-sampler-form" class="generation-workspace-form"></div>
        `,
    })
    const labTab = document.getElementById("tab-sampler-lab")
    const tabSettings = document.getElementById("tab-settings")
    const labForm = document.getElementById("generation-sampler-form")
    if (!labTab || !labForm) return
    if (tabSettings) tabSettings.before(labTab)

    function activate(workspace) {
        root.__flexDiffusionGenerationWorkspace = workspace
        if (workspace === "sampler") {
            labForm.append(editor, preview)
            const help = document.getElementById("generation-sampler-help")
            if (help && !document.getElementById("circuit_bend_row")) {
                help.textContent = "Sampler research plugin is unavailable. Check the installed UI plugins before attempting sampler experiments."
            }
        } else {
            standard.append(editor, preview)
        }
    }

    document.addEventListener("tabClick", (event) => {
        const id = event.detail?.tab?.id
        if (id === "tab-main") activate("vanilla")
        if (id === "tab-sampler-lab") activate("sampler")
    })
    root.__flexDiffusionGenerationWorkspaces = { get active() { return root.__flexDiffusionGenerationWorkspace } }
})(typeof window !== "undefined" ? window : globalThis)
