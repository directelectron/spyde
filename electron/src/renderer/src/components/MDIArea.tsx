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

// Two rects share an edge when one's edge ≈ the other's (tol px) AND they
// overlap on the perpendicular axis — the trigger for forming a stick group.
function edgeAligned(a: Rect, b: Rect, tol = 5): boolean {
  const vOverlap = a.y < b.y + b.h && a.y + a.h > b.y
  const hOverlap = a.x < b.x + b.w && a.x + a.w > b.x
  const near = (p: number, q: number) => Math.abs(p - q) <= tol
  const vert = vOverlap && (near(a.x, b.x + b.w) || near(a.x + a.w, b.x))
  const horiz = hOverlap && (near(a.y, b.y + b.h) || near(a.y + a.h, b.y))
  return vert || horiz
}

// How a partner P relates to a window A (from their PRE-resize rects): are they
// joined side-by-side (→ link height) or stacked (→ link width), and on which side.
function orientationOf(a: Rect, p: Rect) {
  const vOver = Math.min(a.y + a.h, p.y + p.h) - Math.max(a.y, p.y)
  const hOver = Math.min(a.x + a.w, p.x + p.w) - Math.max(a.x, p.x)
  return { sideBySide: vOver >= hOver, pRight: p.x >= a.x, pBelow: p.y >= a.y }
}

// The rect a partner P should adopt after A resized: link the shared dimension
// (height when side-by-side / width when stacked) and follow A's moving edge so
// they stay touching — never overlapping, never gapping. ``ar`` is A's NEW rect,
// ``pr`` keeps P's free dimension.
function linkedRect(ar: Rect, pr: Rect, o: ReturnType<typeof orientationOf>): Rect {
  if (o.sideBySide) {
    return { x: o.pRight ? ar.x + ar.w : ar.x - pr.w, y: ar.y, w: pr.w, h: ar.h }
  }
  return { x: ar.x, y: o.pBelow ? ar.y + ar.h : ar.y - pr.h, w: ar.w, h: pr.h }
}

export function MDIArea() {
  const { state, iframeRefs, sendAction, setActiveWindow, replayState } = useSpyDE()
  const [focusOrder, setFocusOrder] = useState<string[]>([])

  // ── Stick / edge-snap grouping ─────────────────────────────────────────────
  const peersRef = useRef<Map<string, Rect>>(new Map())   // live geometry of all windows
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
  const [groupOf, setGroupOf] = useState<Map<string, number>>(new Map())  // id → groupId
  const [nudges, setNudges] = useState<Map<string, { dx: number; dy: number; nonce: number }>>(new Map())
  const [rects, setRects] = useState<Map<string, Rect & { nonce: number }>>(new Map())
  const dwellRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const nonceRef = useRef(0)
  const groupSeq = useRef(1)

  const reportGeom = useCallback((id: string, rect: Rect) => {
    peersRef.current.set(id, rect)
  }, [])

  const membersOf = useCallback((id: string): string[] => {
    const g = groupOf.get(id)
    if (g == null) return [id]
    return [...groupOf.entries()].filter(([, v]) => v === g).map(([k]) => k)
  }, [groupOf])

  // A stuck window dragged → move every OTHER member of its group by the same delta.
  const onStuckMove = useCallback((id: string, dx: number, dy: number) => {
    const members = membersOf(id).filter(m => m !== id)
    if (!members.length) return
    setNudges(prev => {
      const next = new Map(prev)
      for (const m of members) next.set(m, { dx, dy, nonce: ++nonceRef.current })
      return next
    })
  }, [membersOf])

  // A stuck window resized → propagate the linked dimension along the group's
  // edge-adjacency graph (BFS from the resized window), so partners match the
  // shared dim AND slide to follow the moving edge (no overlap, no gap).
  const onStuckResize = useCallback((id: string, w: number, h: number) => {
    const before = new Map(peersRef.current)      // pre-resize geometry (for orientation/adjacency)
    const start = before.get(id)
    if (start == null) return
    const members = new Set(membersOf(id))
    if (members.size < 2) return
    const adj = (x: string) => [...members].filter(
      m => m !== x && before.get(m) != null && edgeAligned(before.get(x)!, before.get(m)!))

    peersRef.current.set(id, { x: start.x, y: start.y, w, h })   // top-left fixed
    const visited = new Set([id])
    const queue = [id]
    const updates = new Map<string, Rect>()
    while (queue.length) {
      const a = queue.shift()!
      const ar = peersRef.current.get(a)!
      for (const p of adj(a)) {
        if (visited.has(p)) continue
        const nr = linkedRect(ar, peersRef.current.get(p)!, orientationOf(before.get(a)!, before.get(p)!))
        peersRef.current.set(p, nr)
        updates.set(p, nr)
        visited.add(p)
        queue.push(p)
      }
    }
    if (updates.size) {
      setRects(prev => {
        const next = new Map(prev)
        for (const [pid, r] of updates) next.set(pid, { ...r, nonce: ++nonceRef.current })
        return next
      })
    }
  }, [membersOf])

  // A vigorous shake breaks the whole stick group apart.
  const onShake = useCallback((id: string) => {
    setGroupOf(prev => {
      const g = prev.get(id)
      if (g == null) return prev
      const next = new Map(prev)
      for (const [k, v] of prev) if (v === g) next.delete(k)
      return next
    })
  }, [])

  // On drag/resize end, if this window now shares an edge with another and stays
  // put for ~1.1s, form (or extend) a stick group.
  const onGestureEnd = useCallback((id: string) => {
    if (dwellRef.current) clearTimeout(dwellRef.current)
    dwellRef.current = setTimeout(() => {
      const me = peersRef.current.get(id)
      if (!me) return
      const partners = [...peersRef.current.entries()]
        .filter(([pid, r]) => pid !== id && edgeAligned(me, r)).map(([pid]) => pid)
      if (!partners.length) return
      setGroupOf(prev => {
        const next = new Map(prev)
        let gid = next.get(id) ?? partners.map(p => next.get(p)).find(g => g != null) ?? groupSeq.current++
        next.set(id, gid)
        for (const p of partners) {
          const pg = next.get(p)
          if (pg != null && pg !== gid) {            // merge p's whole group into gid
            for (const [k, v] of next) if (v === pg) next.set(k, gid)
          } else next.set(p, gid)
        }
        return next
      })
    }, 1100)
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
    peersRef.current.delete(id)
    placedRef.current.delete(id)   // a reopened window gets a fresh free slot
    setGroupOf(prev => {
      if (!prev.has(id)) return prev
      const next = new Map(prev); next.delete(id); return next
    })
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
  // keep their slot (using live geometry if the user dragged them); NEW windows
  // are packed into the first free gap — so result windows don't bury each other.
  const placements = new Map<string, { x: number; y: number }>()
  const taken: Rect[] = []
  for (const win of visibleWindows) {
    const id = String(win.windowId)
    const placed = placedRef.current.get(id)
    if (!placed) continue
    const { w, h } = windowSize(win.aspect)
    taken.push(peersRef.current.get(id) ?? { x: placed.x, y: placed.y, w, h })
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
            peers={peersRef}
            reportGeom={reportGeom}
            onStuckMove={onStuckMove}
            onGestureEnd={onGestureEnd}
            groupNudge={nudges.get(id)}
            rectOverride={rects.get(id)}
            onStuckResize={onStuckResize}
            onShake={onShake}
            stuck={groupOf.has(id)}
          >
            <WindowContent
              win={win}
              iframeRefs={iframeRefs}
              replayState={replayState}
              sendAction={sendAction}
              ipfKey={state.ipfKey.get(win.windowId)}
              strainRings={state.strainRings.get(win.windowId)}
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
