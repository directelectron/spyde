import React, { useCallback, useEffect, useRef, useState } from 'react'
import { SubWindow } from './SubWindow'
import type { Rect } from './SubWindow'
import { WindowContent } from './WindowContent'
import { useSpyDE } from '../kernel/SpyDEContext'

// The initial size a window opens at — square-ish by default, or matched to the
// backend-reported image aspect so the figure fills it (no letterbox).
function windowSize(aspect?: number): { w: number; h: number } {
  const TITLE = 32
  if (aspect && aspect > 0) {
    const innerH = Math.round(Math.min(360, Math.max(150, 560 / Math.max(aspect, 0.0001))))
    const innerW = Math.round(Math.min(760, Math.max(220, innerH * aspect)))
    return { w: innerW, h: innerH + TITLE }
  }
  return { w: 400, h: 392 }
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
  const { state, iframeRefs, sendAction, setActiveWindow, replayState } = useSpyDE()
  const [focusOrder, setFocusOrder] = useState<string[]>([])

  // Initial placement assigned to each window once (kept stable so re-renders
  // never fight a window the user has dragged).
  const placedRef = useRef<Map<string, { x: number; y: number }>>(new Map())
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

  const handleAction = useCallback((action: string, windowId: number, params: Record<string, unknown> = {}) => {
    // Toolbar buttons map to the generic toolbar_action dispatcher in Python,
    // which resolves the YAML-configured action function by name.
    sendAction('toolbar_action', { name: action, params }, windowId)
  }, [sendAction])

  const visibleWindows = Array.from(state.windows.values()).filter(w => w.visible)

  // Assign each window a non-overlapping initial position. Already-placed windows
  // keep their slot; NEW windows are packed into the first free gap — so result
  // windows don't bury each other.
  const placements = new Map<string, { x: number; y: number }>()
  const taken: Rect[] = []
  for (const win of visibleWindows) {
    const id = String(win.windowId)
    const placed = placedRef.current.get(id)
    if (!placed) continue
    const { w, h } = windowSize(win.aspect)
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
  }

  return (
    <div ref={areaRef} data-testid="mdi-area" style={styles.area}>
      {visibleWindows.map((win) => {
        const id = String(win.windowId)
        const pos = placements.get(id) ?? { x: 40, y: 40 }
        const { w: initW, h: initH } = windowSize(win.aspect)
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
