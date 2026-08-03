/**
 * dnd.ts — the renderer's HTML5 drag-and-drop MIME types.
 *
 * WINDOW_DRAG_MIME     — dragging a signal window (by its titlebar grip);
 *                        payload = the source windowId. Dropping it on a
 *                        navigator's titlebar adds the signal as a NAMED
 *                        navigator (backend `add_navigator_from_window`).
 * NAVIGATOR_DRAG_MIME  — dragging a navigator chip out of its window;
 *                        payload = JSON {windowId, name}. Dropping it on the
 *                        MDI area extracts the navigator into its own signal
 *                        tree (backend `extract_navigator`). Mirrors
 *                        spyde/actions/base.py NAVIGATOR_DRAG_MIME.
 * SIGNAL_REF_DRAG_MIME — dragging a SubWindow's console-ref grip; payload =
 *                        JSON {windowId}. Dropping it on the ConsoleBar input
 *                        resolves windowId → variable name (via the latest
 *                        `console_vars` "signal" entries) and inserts the name
 *                        at the caret. Distinct from WINDOW_DRAG_MIME (which
 *                        targets a navigator titlebar, not the console).
 * CONSOLE_VAR_DRAG_MIME — dragging a console result chip (`out`/`assign`
 *                        console_vars entry) out of the ConsoleBar; payload =
 *                        JSON {name}. Dropping it on the MDI area sends
 *                        `console_create_window` to open it as a new signal
 *                        window.
 * WORKFLOW_NODE_DRAG_MIME — dragging a node from the Workflow tree (Plot Control
 *                        dock); payload = JSON {windowId, signalId, name}.
 *                        Dropping it on the ConsoleBar binds that tree node into
 *                        the console namespace and inserts its variable name.
 * FIGURE_DRAG_MIME    — dragging a window-header pill as a FIGURE reference;
 *                        payload = JSON {windowId, figId?, title?, view?}.
 *                        Stamped by window-header pills alongside the other
 *                        window MIMEs. Dropping it on the Report sidebar embeds
 *                        that figure into the report (backend `report_add_figure`).
 */
export const WINDOW_DRAG_MIME = 'application/x-spyde-window'
export const NAVIGATOR_DRAG_MIME = 'application/x-spyde-navigator'
export const SIGNAL_REF_DRAG_MIME = 'application/x-spyde-signal-ref'
export const CONSOLE_VAR_DRAG_MIME = 'application/x-spyde-console-var'
export const WORKFLOW_NODE_DRAG_MIME = 'application/x-spyde-workflow-node'
export const FIGURE_DRAG_MIME = 'application/x-spyde-figure'

// ── In-process fallback for the dragged window payload ───────────────────────
//
// `dataTransfer.types` is readable throughout a drag, but `getData()` is only
// permitted on DROP — and the drop is where it can come back EMPTY. A drag that
// leaves the renderer for the OS drag pasteboard (a real trackpad drag in the
// packaged app, unlike a synthesized one in a test) can arrive back with the
// custom MIME listed in `types` but with no readable payload behind it. The
// compose handlers then resolve a null source window and silently `return`,
// which presents as: the drop zones light up correctly, you release, and
// NOTHING HAPPENS.
//
// Both drag source and drop target are in this one renderer process, so the
// payload never actually needs to survive a round trip through the OS. The Pill
// stashes it here at dragstart; the drop reads it only when `getData()` yields
// nothing. Cleared on dragend/drop (SpyDEContext) so a stale payload can never
// be applied to an unrelated later drop.
export interface WindowDragPayload {
  windowId: number
  figId?: string
  view?: string
}

let _dragStash: WindowDragPayload | null = null

/** Called by the drag SOURCE at dragstart. */
export function stashWindowDrag(payload: WindowDragPayload | null): void {
  _dragStash = payload
}

/** Read by a drop target when `dataTransfer.getData()` came back empty. */
export function peekWindowDrag(): WindowDragPayload | null {
  return _dragStash
}
