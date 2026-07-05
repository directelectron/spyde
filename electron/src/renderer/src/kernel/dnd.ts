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
 */
export const WINDOW_DRAG_MIME = 'application/x-spyde-window'
export const NAVIGATOR_DRAG_MIME = 'application/x-spyde-navigator'
