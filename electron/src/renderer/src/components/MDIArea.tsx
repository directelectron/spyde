import React, { useCallback, useEffect, useRef, useState } from 'react'
import { SubWindow } from './SubWindow'
import type { Rect } from './SubWindow'
import { WindowContent } from './WindowContent'
import { useSpyDE } from '../kernel/SpyDEContext'

// Tiling: near-square grid (rows x cols) sized to fit `n` windows, cols first
// so wide areas get more columns than rows.
function tileGrid(n: number, areaW: number, areaH: number): { cols: number; rows: number } {
  if (n <= 0) return { cols: 1, rows: 1 }
  const targetCols = Math.max(1, Math.round(Math.sqrt(n * (areaW / Math.max(areaH, 1)))))
  const cols = Math.max(1, Math.min(n, targetCols))
  const rows = Math.max(1, Math.ceil(n / cols))
  return { cols, rows }
}

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

export function MDIArea() {
  const { state, iframeRefs, sendAction, setActiveWindow, replayState, tileWindowsRef } = useSpyDE()
  const [focusOrder, setFocusOrder] = useState<string[]>([])

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

  const getZ = (id: string) => {
    // Unfocused windows sit at a low base; focused windows are always above,
    // most-recently-focused highest. (A single focused window must beat
    // unfocused ones — `10 + 0` == base was the bug.)
    const i = focusOrder.indexOf(id)
    return i === -1 ? 1 : 10 + i
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
    sendAction('close_window', {}, windowId)
  }, [sendAction])

  // Figure sizing is owned by WindowContent's ResizeObserver (it tracks the grid
  // box live across window-resize / tiling / view-bar height), so this is a
  // no-op kept only to satisfy SubWindow's required onResize prop.
  const handleResize = useCallback((_id: string, _w: number, _h: number) => {}, [])

  // Tile: arrange every VISIBLE window into a near-square grid filling the
  // area. An explicit user action (the "Tile" button), so it overrides
  // manually-placed/sized windows — unlike the automatic free-slot placement
  // on open, which never fights a window the user has touched.
  const tileWindows = useCallback(() => {
    const ids = Array.from(state.windows.values()).filter(w => w.visible).map(w => String(w.windowId))
    if (ids.length === 0) return
    const areaW = areaRef.current?.clientWidth || areaSize.w
    const areaH = areaRef.current?.clientHeight || areaSize.h
    const { cols, rows } = tileGrid(ids.length, areaW, areaH)
    const M = 8
    const cellW = Math.floor((areaW - M * (cols + 1)) / cols)
    const cellH = Math.floor((areaH - M * (rows + 1)) / rows)
    const rects = new Map<string, Rect>()
    ids.forEach((id, i) => {
      const col = i % cols
      const row = Math.floor(i / cols)
      const rect = {
        x: M + col * (cellW + M),
        y: M + row * (cellH + M),
        w: Math.max(220, cellW),
        h: Math.max(150, cellH),
      }
      rects.set(id, rect)
      placedRef.current.set(id, { x: rect.x, y: rect.y })
    })
    setForced(prev => ({ gen: prev.gen + 1, rects }))
  }, [state.windows, areaSize])

  useEffect(() => {
    tileWindowsRef.current = tileWindows
    return () => { tileWindowsRef.current = null }
  }, [tileWindowsRef, tileWindows])

  const handleAction = useCallback((action: string, windowId: number, params: Record<string, unknown> = {}) => {
    // Toolbar buttons map to the generic toolbar_action dispatcher in Python,
    // which resolves the YAML-configured action function by name.
    sendAction('toolbar_action', { name: action, params }, windowId)
  }, [sendAction])

  const visibleWindows = Array.from(state.windows.values()).filter(w => w.visible)

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

  return (
    <div ref={areaRef} data-testid="mdi-area" style={styles.area}>
      {visibleWindows.map((win) => {
        const id = String(win.windowId)
        const pos = placements.get(id) ?? { x: 40, y: 40 }
        const { w: initW, h: initH } = windowSize(win.aspect)
        const otherRects = visibleWindows
          .filter(w => String(w.windowId) !== id)
          .map(w => liveRectsRef.current.get(String(w.windowId)))
          .filter((r): r is Rect => r != null)
        return (
          <SubWindow
            key={id}
            id={id}
            windowId={win.windowId}
            title={win.title}
            initialX={pos.x}
            initialY={pos.y}
            initialW={initW}
            initialH={initH}
            toolbarActions={win.toolbarActions}
            onClose={handleClose}
            onFocus={handleFocus}
            onResize={handleResize}
            onAction={handleAction}
            zIndex={getZ(id)}
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
  )
}

const styles: Record<string, React.CSSProperties> = {
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
