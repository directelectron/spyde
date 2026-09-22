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
 * MEMBER_DRAG_MIME    — dragging a multi-angle member from one slot of the
 *                        loader's ring to another; payload = the member index.
 *                        Dropping it re-angles that member (backend
 *                        `maped_set_member`), as opposed to a FILE coming in
 *                        from the desktop onto the same slot.
 *
 * It also holds what a drop CARRIES: `pathsFromDrop` for files arriving from the
 * desktop, and the in-process payload stash below.
 */
export const WINDOW_DRAG_MIME = 'application/x-spyde-window'
export const NAVIGATOR_DRAG_MIME = 'application/x-spyde-navigator'
export const SIGNAL_REF_DRAG_MIME = 'application/x-spyde-signal-ref'
export const CONSOLE_VAR_DRAG_MIME = 'application/x-spyde-console-var'
export const WORKFLOW_NODE_DRAG_MIME = 'application/x-spyde-workflow-node'
export const FIGURE_DRAG_MIME = 'application/x-spyde-figure'
export const MEMBER_DRAG_MIME = 'application/x-maped-member'

/**
 * The OS paths behind a file drop, and a note when there are none.
 *
 * A sandboxed renderer has no `File.path` (Electron 44 removed it), so each
 * File is resolved through the preload's `webUtils.getPathForFile`. That is the
 * one failure a drop handler can have SILENTLY — every zone lights up, the drop
 * is accepted, and nothing happens — so a drop that resolves to nothing says
 * so, and a caller that shows `note` cannot repeat that.
 */
export function pathsFromDrop(e: { dataTransfer: DataTransfer }): {
  paths: string[]
  note: string
} {
  const files = Array.from(e.dataTransfer.files)
  if (files.length === 0) return { paths: [], note: 'That drop carried no files.' }
  const paths = files
    .map((f) => window.electron.pathForFile?.(f))
    .filter((p): p is string => !!p)
  if (paths.length === 0) {
    return {
      paths: [],
      note: `Could not read a file path from ${files.length} dropped item(s) `
        + '— use Add datasets… instead.',
    }
  }
  return {
    paths,
    note: paths.length < files.length
      ? `${files.length - paths.length} of ${files.length} dropped items had no readable path.`
      : '',
  }
}

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

// The same fallback for the multi-angle loader's slot-to-slot member drag,
// whose payload is just an index. Separate from the window stash so neither can
// be mistaken for the other on a drop; cleared by the same dragend/drop.
let _memberStash: number | null = null

/** Called by the drag SOURCE at dragstart. */
export function stashMemberDrag(index: number | null): void {
  _memberStash = index
}

/** Read by a slot when `dataTransfer.getData()` came back empty. */
export function peekMemberDrag(): number | null {
  return _memberStash
}
