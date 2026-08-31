import React, { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { SubWindow, MIN_W, MIN_H } from './SubWindow'
import type { Rect } from './SubWindow'
import { WindowContent } from './WindowContent'
import {
  useSpyDE, type SpyDEWindow, type NavigatorOptions,
} from '../kernel/SpyDEContext'
import { NAVIGATOR_DRAG_MIME, CONSOLE_VAR_DRAG_MIME } from '../kernel/dnd'
import { Pill, type PillSegment, type WindowPillPayload } from './Pill'
import { getMovieEditorClaim, subscribeMovieEditorClaim } from '../kernel/movieEditorClaim'

// A compact breadcrumb for a window: a small S-/N- kind badge + the editable
// dataset name, e.g. `S-my_scan` / `N-my_scan`. No trailing Root/nav segment —
// kept short so the pill doesn't crowd the titlebar.
function buildBreadcrumb(win: SpyDEWindow): PillSegment[] {
  const name = win.title || (win.isNavigator ? 'Navigator' : 'Signal')
  return [
    { text: win.isNavigator ? 'N-' : 'S-', tone: 'muted', prefix: true },
    { text: name, tone: 'accent', testid: 'breadcrumb-name' },
  ]
}

function windowPayloadFor(
  win: SpyDEWindow, navigatorOptions: Map<number, NavigatorOptions>,
): WindowPillPayload {
  const nav = win.isNavigator ? navigatorOptions.get(win.windowId) : undefined
  // Best-effort "currently shown" figure for the Report-sidebar drag payload.
  // This mirrors WindowContent's default fallback (the first non-3d/density/
  // tiled/stacked figure), not its full chip-selection state — good enough for
  // a drag hint; the backend can resolve the window's true active figure if
  // this is stale.
  const shownFig = win.figures.find(f =>
    f.view !== '3d' && f.view !== 'density' && f.view !== 'density3d'
    && f.view !== 'ipf2d'
    && f.viewLabel !== '__tiled__' && f.viewLabel !== '__stacked__') ?? win.figures[0]
  return {
    windowId: win.windowId,
    isNavigator: win.isNavigator,
    navName: nav?.current || nav?.names?.[0] || 'base',
    figId: shownFig?.figId,
    title: shownFig?.title,
    view: shownFig?.view,
  }
}

// Tiling: near-square grid (rows x cols) sized to fit `n` windows, cols first
// so wide areas get more columns than rows.
function tileGrid(n: number, areaW: number, areaH: number): { cols: number; rows: number } {
  if (n <= 0) return { cols: 1, rows: 1 }
  const targetCols = Math.max(1, Math.round(Math.sqrt(n * (areaW / Math.max(areaH, 1)))))
  const cols = Math.max(1, Math.min(n, targetCols))
  const rows = Math.max(1, Math.ceil(n / cols))
  return { cols, rows }
}

/** Highest z-index an MDI window may take. Everything that must float above
 *  the windows lives at CHROME_Z and up (WizardShell.tsx). */
export const MDI_Z_CEILING = 899

// The initial size a window opens at — square-ish by default, or matched to the
// backend-reported image aspect so the figure fills it (no letterbox). Kept
// deliberately compact so more windows fit on screen before overlapping.
function windowSize(aspect?: number): { w: number; h: number } {
  const TITLE = 32
  if (aspect && aspect > 0) {
    const innerH = Math.round(Math.min(300, Math.max(130, 460 / Math.max(aspect, 0.0001))))
    const innerW = Math.round(Math.min(620, Math.max(190, innerH * aspect)))
    return { w: innerW, h: innerH + TITLE }
  }
  return { w: 340, h: 320 }
}

function overlaps(a: Rect, b: Rect, gap = 10): boolean {
  return a.x < b.x + b.w + gap && a.x + a.w + gap > b.x &&
         a.y < b.y + b.h + gap && a.y + a.h + gap > b.y
}

// First-fit packing: scan the area in reading order for the first slot where a
// w×h window doesn't collide with anything already placed. Falls back to a tight
// cascade only when the area is genuinely full. This is what stops result
// windows (IPF / strain / refine / vectors) from burying each other.
function findFreeSlot(w: number, h: number, taken: Rect[], areaW: number, areaH: number,
                      n: number): { x: number; y: number } {
  const M = 14, step = 26
  const maxX = Math.max(M, areaW - w - M)
  const maxY = Math.max(M, areaH - h - M)
  for (let y = M; y <= maxY; y += step) {
    for (let x = M; x <= maxX; x += step) {
      const r = { x, y, w, h }
      if (!taken.some(t => overlaps(r, t))) return { x, y }
    }
  }
  return { x: M + (n % 6) * 30, y: M + (n % 6) * 30 }
}

// Vertical space reserved below each window row when tiling so the floating
// toolbar (which hangs below the window) stays visible: bar height + gap.
const TOOLBAR_RESERVE = 44

// Margin a recovered window is pulled back to inside the area edge.
const RECOVER_MARGIN = 8
// An area smaller than this in either axis is not a real layout — it is what
// gets measured while the window is hidden, minimised, or mid-display-change.
// Clamping to such a measurement would stack every window at the origin, so it
// is treated as "no information" and skipped. This is the guard that makes the
// recovery safe to run on a resume, which is exactly when a bogus 0 shows up.
const MIN_CREDIBLE_AREA = 120

// Remap a rect from an area of `from` to one of `to`, keeping its position and
// size RELATIVE to the area. This is what preserves the arrangement: a two-up
// layout stays two-up, just smaller, instead of collapsing into a pile at the
// left edge (clamping alone buries every window under the last one, which still
// reads as "my plot is gone").
//
// The scale is capped at 1 so this only ever shrinks — a grown area must never
// inflate a window the user sized by hand.
function remapRect(r: Rect, from: { w: number; h: number }, to: { w: number; h: number }): Rect {
  const sx = Math.min(1, to.w / Math.max(from.w, 1))
  const sy = Math.min(1, to.h / Math.max(from.h, 1))
  return {
    x: Math.round(r.x * sx), y: Math.round(r.y * sy),
    w: Math.round(r.w * sx), h: Math.round(r.h * sy),
  }
}

// Pull a window back inside an area of areaW×areaH, keeping its size when it
// fits and shrinking it (never below the manual-resize floor) when it doesn't.
// Returns null when the window is already fully inside — the common case, so
// recovery is a no-op that allocates nothing.
function recoverRect(r: Rect, areaW: number, areaH: number): Rect | null {
  const M = RECOVER_MARGIN
  const w = Math.max(MIN_W, Math.min(r.w, areaW - 2 * M))
  const h = Math.max(MIN_H, Math.min(r.h, areaH - 2 * M))
  // Math.max(M, …) keeps the top-left corner reachable even when the window is
  // larger than the area (area narrower than MIN_W): better to overflow the
  // bottom-right, which is only clipped, than the top-left, which would put the
  // titlebar — the sole drag handle — out of reach.
  const x = Math.min(Math.max(r.x, M), Math.max(M, areaW - w - M))
  const y = Math.min(Math.max(r.y, M), Math.max(M, areaH - h - M))
  if (x === r.x && y === r.y && w === r.w && h === r.h) return null
  return { x, y, w, h }
}

export function MDIArea() {
  const { state, iframeRefs, sendAction, setActiveWindow, replayState, tileWindowsRef } = useSpyDE()
  // The MDI window (if any) the full-screen Movie editor currently surfaces
  // live (laundry #6/#13 — see movieEditorClaim.ts). That window is EXCLUDED
  // below (a real unmount, not display:none) so its iframe doesn't fight the
  // editor's own iframe for the shared fig_id's live pushes — otherwise the
  // editor's annotation overlays / tile detail can render onto the hidden MDI
  // window instead of the visible editor, and closing the editor reveals a
  // window still showing the movie's overlays.
  const movieClaimedWindowId = useSyncExternalStore(subscribeMovieEditorClaim, getMovieEditorClaim)
  const [focusOrder, setFocusOrder] = useState<string[]>([])
  // Renderer-side minimize: minimized windows stay mounted (their figure
  // iframes keep streaming) but are display:none and listed in the top bar.
  const [minimized, setMinimized] = useState<Set<string>>(new Set())

  // Initial placement assigned to each window once (kept stable so re-renders
  // never fight a window the user has dragged).
  const placedRef = useRef<Map<string, { x: number; y: number }>>(new Map())
  // Window ids placed for the first time THIS render pass, drained by the
  // focus-on-open effect below.
  const newWindowIdsRef = useRef<string[]>([])
  // Live rect (position+size) of every window, updated continuously while
  // dragging/resizing (not just at rest) — snapping and the free-slot search
  // need the CURRENT layout, including windows mid-drag.
  const liveRectsRef = useRef<Map<string, Rect>>(new Map())
  const handleLiveRect = useCallback((id: string, rect: Rect) => {
    liveRectsRef.current.set(id, rect)
  }, [])
  // Forced layout (from Tile): bumping the generation makes every SubWindow
  // adopt its forced rect even if the user had manually resized/moved it.
  const [forced, setForced] = useState<{ gen: number; rects: Map<string, Rect> }>(
    { gen: 0, rects: new Map() },
  )
  const areaRef = useRef<HTMLDivElement>(null)
  const [areaSize, setAreaSize] = useState({ w: 1280, h: 820 })
  useEffect(() => {
    const el = areaRef.current
    if (!el) return
    const measure = () => setAreaSize({ w: el.clientWidth, h: el.clientHeight })
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const handleFocus = useCallback((id: string) => {
    setFocusOrder(prev => [...prev.filter(x => x !== id), id])
    setActiveWindow(parseInt(id, 10))
  }, [setActiveWindow])

  // Windows with an open caret. Their caret renders inside the window, whose
  // root forms a stacking context, so the caret cannot paint above a sibling
  // window however high its own z-index goes (measured: a caret at z-index 1002
  // lost to a window at 11). Keeping the owning window on top is the only thing
  // that keeps the caret reachable — otherwise a result window opened by the
  // very action the caret is running lands over it and its figure iframe eats
  // every click. See caret_above_windows.spec.ts.
  const [caretWindows, setCaretWindows] = useState<ReadonlySet<string>>(new Set())
  const setCaretOpen = React.useCallback((id: string, open: boolean) => {
    setCaretWindows((prev) => {
      if (prev.has(id) === open) return prev
      const next = new Set(prev)
      if (open) next.add(id); else next.delete(id)
      return next
    })
  }, [])

  const getZ = (id: string) => {
    if (caretWindows.has(id)) return MDI_Z_CEILING
    // Unfocused windows sit at a low base; focused windows are always above,
    // most-recently-focused highest. (A single focused window must beat
    // unfocused ones — `10 + 0` == base was the bug.)
    //
    // CLAMPED, because this grows with the number of windows and the floating
    // chrome above it does not. At five windows a focused one reached 14 —
    // the wizard panel's own z-index — and started painting over the caret the
    // user was mid-way through using; its figure iframe then swallowed the
    // clicks. Windows own 1..MDI_Z_CEILING and nothing else may enter that
    // range; see CHROME_Z in WizardShell.tsx for the layer above.
    const i = focusOrder.indexOf(id)
    return i === -1 ? 1 : Math.min(10 + i, MDI_Z_CEILING)
  }

  // Clicking the figure raises its window. The out-of-process iframe swallows the
  // mousedown so it never reaches the window root (and the blur/activeElement
  // trick was unreliable). Instead the figure HTML posts a `spyde_focus` message
  // on pointerdown (injected in Plot._ensure_figure), which we use to raise the
  // owning window — works regardless of focus quirks.
  React.useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (e.data?.type !== 'spyde_focus' || !e.data.figId) return
      const fig = state.figures.get(e.data.figId)
      if (fig) handleFocus(String(fig.windowId))
    }
    window.addEventListener('message', onMsg)
    return () => window.removeEventListener('message', onMsg)
  }, [state.figures, handleFocus])

  const handleClose = useCallback((id: string) => {
    const windowId = parseInt(id, 10)
    placedRef.current.delete(id)   // a reopened window gets a fresh free slot
    setMinimized(prev => {
      if (!prev.has(id)) return prev
      const next = new Set(prev); next.delete(id); return next
    })
    sendAction('close_window', {}, windowId)
  }, [sendAction])

  const handleMinimize = useCallback((id: string) => {
    setMinimized(prev => new Set(prev).add(id))
    setFocusOrder(prev => prev.filter(x => x !== id))
  }, [])

  // Rename the dataset (double-click the breadcrumb Name). Sends set_title to the
  // backend, which updates the shared root title on every window of the tree.
  const handleRename = useCallback((windowId: number, name: string) => {
    sendAction('set_title', { title: name }, windowId)
  }, [sendAction])

  const handleRestore = useCallback((id: string) => {
    setMinimized(prev => {
      const next = new Set(prev); next.delete(id); return next
    })
    handleFocus(id)
  }, [handleFocus])

  // Figure sizing is owned by WindowContent's ResizeObserver (it tracks the grid
  // box live across window-resize / tiling / view-bar height), so this is a
  // no-op kept only to satisfy SubWindow's required onResize prop.
  const handleResize = useCallback((_id: string, _w: number, _h: number) => {}, [])

  // Tile: arrange every VISIBLE window into a near-square grid filling the
  // area. An explicit user action (the "Tile" button), so it overrides
  // manually-placed/sized windows — unlike the automatic free-slot placement
  // on open, which never fights a window the user has touched.
  const tileWindows = useCallback(() => {
    const ids = Array.from(state.windows.values())
      .filter(w => w.visible && !minimized.has(String(w.windowId)))
      .map(w => String(w.windowId))
    if (ids.length === 0) return
    const areaW = areaRef.current?.clientWidth || areaSize.w
    const areaH = areaRef.current?.clientHeight || areaSize.h
    const { cols, rows } = tileGrid(ids.length, areaW, areaH)
    const M = 8
    const cellW = Math.floor((areaW - M * (cols + 1)) / cols)
    // Each row also reserves room for the floating toolbar hanging below its
    // windows, so tiled windows never bury the row above's toolbar.
    const cellH = Math.floor((areaH - (M + TOOLBAR_RESERVE) * rows - M) / rows)
    const rects = new Map<string, Rect>()
    ids.forEach((id, i) => {
      const col = i % cols
      const row = Math.floor(i / cols)
      const rect = {
        x: M + col * (cellW + M),
        y: M + row * (cellH + M + TOOLBAR_RESERVE),
        w: Math.max(220, cellW),
        h: Math.max(150, cellH),
      }
      rects.set(id, rect)
      placedRef.current.set(id, { x: rect.x, y: rect.y })
    })
    setForced(prev => ({ gen: prev.gen + 1, rects }))
  }, [state.windows, areaSize, minimized])

  useEffect(() => {
    tileWindowsRef.current = tileWindows
    return () => { tileWindowsRef.current = null }
  }, [tileWindowsRef, tileWindows])

  // ── Recover windows stranded outside the area ───────────────────────────────
  // Windows live at ABSOLUTE pixel coordinates: `placedRef` caches the initial
  // slot for the life of the window, and each SubWindow then owns its own `pos`
  // state. NOTHING re-derives either when the AREA changes size, and the area is
  // `overflow: hidden` — so anything that shrinks the area leaves windows sitting
  // outside the new bounds. `clampToVisible` doesn't save them: it runs only
  // while a titlebar is being dragged.
  //
  // A stranded window is not closed and not minimized. It is still in state,
  // still streaming, and NOT listed in the top bar (which enumerates minimized
  // windows only) — so short of Tile there is no way to reach it again. That is
  // the "all my plots vanished after I closed and reopened the laptop" symptom:
  // a Mac lid-close routinely changes the display configuration and resizes the
  // app window. Measured on the real failure, the backend, the Dask cluster and
  // the renderer PROCESS all survive the sleep — the windows were never lost,
  // only put somewhere you can't see. Unplugging an external display or dragging
  // the app window smaller strands them exactly the same way.
  //
  // So when an area change leaves ANY window out of bounds, reflow the whole
  // layout down into the new area — proportionally, so the arrangement the user
  // built survives as a smaller version of itself — and clamp what still
  // doesn't fit. Nothing happens while every window is inside, which is the
  // common case (growing the area, or a nudge that strands nothing), so an
  // ordinary resize never disturbs a layout.
  const recoveredAreaRef = useRef<{ w: number; h: number } | null>(null)
  useEffect(() => {
    const { w, h } = areaSize
    if (w < MIN_CREDIBLE_AREA || h < MIN_CREDIBLE_AREA) return
    const last = recoveredAreaRef.current
    if (last && last.w === w && last.h === h) return   // measure() re-fires with equal dims
    recoveredAreaRef.current = { w, h }

    // Minimized windows are included deliberately: they are display:none, so
    // recovering them now is what makes restoring one land in view.
    const current = new Map<string, Rect>()
    for (const win of state.windows.values()) {
      if (!win.visible) continue
      const id = String(win.windowId)
      const placed = placedRef.current.get(id)
      if (!placed) continue   // not laid out yet — placement below will fit it
      current.set(id, liveRectsRef.current.get(id) ?? { ...placed, ...windowSize(win.aspect) })
    }
    if (![...current.values()].some(r => recoverRect(r, w, h))) return

    const rects = new Map<string, Rect>()
    for (const [id, r] of current) {
      // Without a previous area (the very first credible measurement) there is
      // no ratio to remap by, so fall back to clamping alone.
      const scaled = last ? remapRect(r, last, { w, h }) : r
      const next = recoverRect(scaled, w, h) ?? scaled
      if (next.x === r.x && next.y === r.y && next.w === r.w && next.h === r.h) continue
      rects.set(id, next)
      placedRef.current.set(id, { x: next.x, y: next.y })
    }
    if (rects.size > 0) setForced(prev => ({ gen: prev.gen + 1, rects }))
  }, [areaSize, state.windows])

  const handleAction = useCallback((action: string, windowId: number, params: Record<string, unknown> = {}) => {
    // Toolbar buttons map to the generic toolbar_action dispatcher in Python,
    // which resolves the YAML-configured action function by name.
    sendAction('toolbar_action', { name: action, params }, windowId)
  }, [sendAction])

  // Drops onto the MDI background:
  //  • a navigator chip → extract that navigator into its own signal tree,
  //  • a console result chip → open that console variable as a new window,
  //  • files (incl. .zspy folders) → open them like File→Open.
  const onAreaDragOver = useCallback((e: React.DragEvent) => {
    const types = e.dataTransfer.types
    if (types.includes(NAVIGATOR_DRAG_MIME) || types.includes(CONSOLE_VAR_DRAG_MIME)
        || types.includes('Files')) {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    }
  }, [])
  const onAreaDrop = useCallback((e: React.DragEvent) => {
    const nav = e.dataTransfer.getData(NAVIGATOR_DRAG_MIME)
    if (nav) {
      e.preventDefault()
      try {
        const { windowId, name } = JSON.parse(nav) as { windowId: number; name: string }
        if (name != null && windowId != null) sendAction('extract_navigator', { name }, windowId)
      } catch { /* malformed payload — ignore */ }
      return
    }
    const consoleVar = e.dataTransfer.getData(CONSOLE_VAR_DRAG_MIME)
    if (consoleVar) {
      e.preventDefault()
      try {
        const { name } = JSON.parse(consoleVar) as { name: string }
        if (name != null) sendAction('console_create_window', { name })
      } catch { /* malformed payload — ignore */ }
      return
    }
    if (e.dataTransfer.files.length > 0) {
      e.preventDefault()
      // Sandboxed renderers have no File.path — the preload resolves each
      // File to its OS path via webUtils.getPathForFile.
      const paths = Array.from(e.dataTransfer.files)
        .map(f => window.electron.pathForFile?.(f))
        .filter((p): p is string => !!p)
      for (const path of paths) sendAction('open_file', { path })
    }
  }, [sendAction])

  const visibleWindows = Array.from(state.windows.values())
    .filter(w => w.visible && w.windowId !== movieClaimedWindowId)

  // Assign each window a non-overlapping initial position. Already-placed windows
  // keep their slot; NEW windows are packed into the first free gap — so result
  // windows don't bury each other. Uses each window's CURRENT live rect (which
  // reflects manual resizes) when known, so a new window's free-slot search
  // doesn't land on top of a window the user has enlarged.
  const placements = new Map<string, { x: number; y: number }>()
  const taken: Rect[] = []
  for (const win of visibleWindows) {
    const id = String(win.windowId)
    const placed = placedRef.current.get(id)
    if (!placed) continue
    const live = liveRectsRef.current.get(id)
    const { w, h } = live ? { w: live.w, h: live.h } : windowSize(win.aspect)
    taken.push({ x: placed.x, y: placed.y, w, h })
    placements.set(id, placed)
  }
  // Read the LIVE area size (the `areaSize` state can still be the default when
  // the first windows arrive); `areaSize` just forces a re-render on resize.
  const areaW = areaRef.current?.clientWidth || areaSize.w
  const areaH = areaRef.current?.clientHeight || areaSize.h
  for (const win of visibleWindows) {
    const id = String(win.windowId)
    if (placedRef.current.has(id)) continue
    const { w, h } = windowSize(win.aspect)
    const slot = findFreeSlot(w, h, taken, areaW, areaH, taken.length)
    placedRef.current.set(id, slot)
    placements.set(id, slot)
    taken.push({ ...slot, w, h })
    newWindowIdsRef.current.push(id)
  }

  // A freshly-opened window (e.g. a backend-initiated result window like the
  // Strain map, opened without the user clicking it) must land on TOP —
  // otherwise it stays at the unfocused base z-index and an existing window's
  // iframe can cover its controls, making them unclickable even though the new
  // window is visually "there". Matches normal desktop window-open behaviour.
  useEffect(() => {
    if (newWindowIdsRef.current.length === 0) return
    const ids = newWindowIdsRef.current
    newWindowIdsRef.current = []
    setFocusOrder(prev => [...prev.filter(x => !ids.includes(x)), ...ids])
  })

  const minimizedWins = visibleWindows.filter(w => minimized.has(String(w.windowId)))

  return (
    <div style={styles.outer}>
      {/* Top bar listing minimized windows — the SAME breadcrumb pill as the
          header. Click restores; it's also draggable (drop into MDI/console). */}
      {minimizedWins.length > 0 && (
        <div data-testid="minimized-bar" style={styles.minBar}>
          {minimizedWins.map(w => (
            <Pill
              key={w.windowId}
              testid={`min-chip-${w.windowId}`}
              size="sm"
              segments={buildBreadcrumb(w)}
              window={windowPayloadFor(w, state.navigatorOptions)}
              title={`${w.title} — click to restore, drag to add/bind`}
              onClick={() => handleRestore(String(w.windowId))}
            />
          ))}
        </div>
      )}
      <div
        ref={areaRef}
        data-testid="mdi-area"
        style={styles.area}
        onDragOver={onAreaDragOver}
        onDrop={onAreaDrop}
      >
        {visibleWindows.map((win) => {
          const id = String(win.windowId)
          const pos = placements.get(id) ?? { x: 40, y: 40 }
          const { w: initW, h: initH } = windowSize(win.aspect)
          const otherRects = visibleWindows
            .filter(w => String(w.windowId) !== id && !minimized.has(String(w.windowId)))
            .map(w => liveRectsRef.current.get(String(w.windowId)))
            .filter((r): r is Rect => r != null)
          return (
            <SubWindow
              key={id}
              id={id}
              windowId={win.windowId}
              title={win.title}
              breadcrumb={buildBreadcrumb(win)}
              windowPayload={windowPayloadFor(win, state.navigatorOptions)}
              onRename={handleRename}
              initialX={pos.x}
              initialY={pos.y}
              initialW={initW}
              initialH={initH}
              toolbarActions={win.toolbarActions}
              onClose={handleClose}
              onFocus={handleFocus}
              onMinimize={handleMinimize}
              onResize={handleResize}
              onAction={handleAction}
              zIndex={getZ(id)}
              onCaretOpenChange={(open: boolean) => setCaretOpen(id, open)}
              hidden={minimized.has(id)}
              acceptSignalDrop={win.isNavigator}
              onSignalDrop={(srcId) =>
                sendAction('add_navigator_from_window', { source_window_id: srcId }, win.windowId)}
              areaSize={{ w: areaW, h: areaH }}
              otherRects={otherRects}
              onLiveRect={handleLiveRect}
              forced={forced.rects.has(id) ? { gen: forced.gen, rect: forced.rects.get(id)! } : undefined}
            >
              <WindowContent
                win={win}
                iframeRefs={iframeRefs}
                replayState={replayState}
                sendAction={sendAction}
              />
            </SubWindow>
          )
        })}

        {visibleWindows.length === 0 && (
          <div style={styles.empty}>
            {state.ready ? 'Open a file to begin' : state.status}
          </div>
        )}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  outer: {
    flex: 1,
    display: 'flex', flexDirection: 'column',
    minWidth: 0, minHeight: 0,
  },
  minBar: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '4px 8px', flexShrink: 0,
    background: '#181825', borderBottom: '1px solid #313244',
  },
  area: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
    backgroundColor: '#11111b',
  },
  empty: {
    position: 'absolute', inset: 0,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: '#45475a', fontSize: 14, pointerEvents: 'none',
  },
}
