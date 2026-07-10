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
 */
export const WINDOW_DRAG_MIME = 'application/x-spyde-window'
export const NAVIGATOR_DRAG_MIME = 'application/x-spyde-navigator'
export const SIGNAL_REF_DRAG_MIME = 'application/x-spyde-signal-ref'
export const CONSOLE_VAR_DRAG_MIME = 'application/x-spyde-console-var'
