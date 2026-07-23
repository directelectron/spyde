/**
 * movieEditorClaim.ts — which MDI window's iframe the full-screen Movie editor
 * currently owns (laundry #6 / #13).
 *
 * THE BUG: MovieEditor surfaces the tree's REAL signal figure by mounting a
 * SECOND `<iframe>` for the SAME `fig_id` (via SeamlessFigureFrame) while the
 * source signal's ordinary MDI window (WindowContent, rendered by MDIArea) is
 * ALSO still mounted — Movie editor is a `position:fixed` overlay on TOP of the
 * MDI area, it doesn't replace it. Both iframes call
 * `iframeRefs.current.set(figId, el)` on the SAME key, so exactly one of them
 * "wins" the ref that `state_update` / `state_update_binary` pushes go to
 * (`SpyDEContext`'s `iframeRefs.current.get(figId)`):
 *   - If the MDI window's iframe wins, the editor's own iframe never receives
 *     the live pushes — the annotation-widget `awi_state` pushes (and the
 *     tile-mode `enable_tile`/`update_tile_source` detail-tile pushes) land on
 *     the HIDDEN window instead of the visible editor, so the editor shows a
 *     stale/blurry frame while the invisible MDI iframe repaints live.
 *   - If the editor's iframe wins, the MDI window's iframe (still mounted,
 *     just behind the editor's z-index:9300 overlay) silently keeps its LAST
 *     replayed state — including any annotation widgets the editor added —
 *     so closing the editor reveals a window that still shows the movie's
 *     crop/text/image overlays baked onto the live plot (laundry #6).
 *
 * THE FIX: while the editor holds a cell's live signal window, MDIArea excludes
 * that window from the windows it renders (a real unmount, not just
 * `display:none` — SubWindow's `hidden` prop only hides visually and would
 * leave the iframe registered). That leaves exactly ONE consumer of the shared
 * fig_id's pushes: the editor's SeamlessFigureFrame. Restoring is automatic —
 * once the claim clears (editor close / cell switch / component unmount), the
 * window reappears in MDIArea's next render with a FRESH iframe mount (a normal
 * `onLoad` → `replayState` gives it the current widget/tile state, so nothing
 * needs to be undone server-side for this half of the story — the backend-side
 * teardown of the annotation widgets themselves is separate, see movie.py).
 *
 * A plain module-scope store (not React state / Context) so MDIArea doesn't
 * need a SpyDEContext.tsx reducer round-trip for a value that only MovieGate
 * writes and MDIArea/WindowContent read reactively; mirrors the existing
 * `activeFigure.ts` idiom but adds subscribe() since this one must trigger a
 * re-render (activeFigure.ts is read only at drag-time).
 */

type Listener = () => void

let claimedWindowId: number | null = null
const listeners = new Set<Listener>()

/** Publish (or clear, with null) the MDI window id the Movie editor currently
 *  surfaces live (so MDIArea can exclude it while the editor holds it). */
export function setMovieEditorClaim(windowId: number | null): void {
  if (claimedWindowId === windowId) return
  claimedWindowId = windowId
  for (const l of listeners) l()
}

/** The window id currently claimed by an open Movie editor, or null. */
export function getMovieEditorClaim(): number | null {
  return claimedWindowId
}

/** React `useSyncExternalStore`-compatible subscribe. */
export function subscribeMovieEditorClaim(listener: Listener): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}
