/**
 * activeFigure.ts — which figure each window is CURRENTLY showing.
 *
 * WindowContent owns the view-toggle state (2D ⇄ 3D ⇄ density, view chips) as
 * LOCAL React state, so nothing outside the window's content re-renders when
 * the user flips views — in particular MDIArea, which builds the header pill's
 * drag payload at ITS render time, cannot know the toggle changed. This tiny
 * module-scope registry bridges that gap without a context ripple:
 * WindowContent publishes its shown figure on every change, and the Pill reads
 * it AT DRAGSTART TIME (the only moment the payload matters), so dragging a
 * window while its 3-D IPF view is up stamps `view:'3d'` in FIGURE_DRAG_MIME —
 * the Report Builder's cue to snapshot the scene instead of the 2-D map.
 *
 * Mutable module state (not React state) is deliberate: reads happen inside a
 * native dragstart handler, writes inside an effect — no re-render needed on
 * either side, and drag-time freshness is guaranteed.
 */

export interface ActiveFigureInfo {
  figId: string
  title?: string
  /** the figure's `view` tag ("3d" for the IPF explorer, "density", undefined
   *  for the primary 2-D figure) — what report_add_figure branches on. */
  view?: string
}

const registry = new Map<number, ActiveFigureInfo>()

/** Publish (or clear, with null) a window's currently-shown figure. */
export function setActiveFigure(windowId: number, info: ActiveFigureInfo | null): void {
  if (info == null) registry.delete(windowId)
  else registry.set(windowId, info)
}

/** The window's currently-shown figure, or null when unknown (fall back to the
 *  render-time payload heuristic). */
export function getActiveFigure(windowId: number): ActiveFigureInfo | null {
  return registry.get(windowId) ?? null
}
