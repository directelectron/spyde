import React, { useCallback, useRef, useState } from 'react'
import { SubWindow } from './SubWindow'
import type { Rect } from './SubWindow'
import { useSpyDE } from '../kernel/SpyDEContext'

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

export function MDIArea() {
  const { state, iframeRefs, sendAction, setActiveWindow, replayState } = useSpyDE()
  const [focusOrder, setFocusOrder] = useState<string[]>([])
  // Per-window IPF view ('2d' | '3d') for windows that carry a 3-D explorer figure.
  const [ipfView, setIpfView] = useState<Record<number, '2d' | '3d'>>({})

  // ── Stick / edge-snap grouping ─────────────────────────────────────────────
  const peersRef = useRef<Map<string, Rect>>(new Map())   // live geometry of all windows
  const [groupOf, setGroupOf] = useState<Map<string, number>>(new Map())  // id → groupId
  const [nudges, setNudges] = useState<Map<string, { dx: number; dy: number; nonce: number }>>(new Map())
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
    setGroupOf(prev => {
      if (!prev.has(id)) return prev
      const next = new Map(prev); next.delete(id); return next
    })
    sendAction('close_window', {}, windowId)
  }, [sendAction])

  const handleResize = useCallback((id: string, w: number, h: number) => {
    // Find any figure in this window and send resize
    const windowId = parseInt(id, 10)
    const win = state.windows.get(windowId)
    if (!win) return
    const figId = win.figures[0]?.figId
    if (figId) {
      window.electron.resizeFigure(figId, Math.max(100, w - 2), Math.max(100, h - 68))
    }
  }, [state.windows])

  const handleAction = useCallback((action: string, windowId: number, params: Record<string, unknown> = {}) => {
    // Toolbar buttons map to the generic toolbar_action dispatcher in Python,
    // which resolves the YAML-configured action function by name.
    sendAction('toolbar_action', { name: action, params }, windowId)
  }, [sendAction])

  const visibleWindows = Array.from(state.windows.values()).filter(w => w.visible)
  const gridPositions = (i: number) => ({
    x: 60 + (i % 5) * 40,
    y: 40 + (i % 5) * 40,
  })

  return (
    <div data-testid="mdi-area" style={styles.area}>
      {visibleWindows.map((win, i) => {
        const pos = gridPositions(i)
        const id = String(win.windowId)
        // Squarer + a bit bigger default (DPs and most navigator grids are
        // ~square). When the backend reports an image aspect (e.g. a wide 208×64
        // navigator), size the window's content to it so the image FILLS the
        // window — otherwise anyplotlib letterboxes it into a strip and the
        // crosshair/axes no longer line up with the image.
        const TITLE = 32
        let initW = 400, initH = 392
        if (win.aspect && win.aspect > 0) {
          const innerH = Math.round(Math.min(360, Math.max(150, 560 / Math.max(win.aspect, 0.0001))))
          const innerW = Math.round(Math.min(760, Math.max(220, innerH * win.aspect)))
          initW = innerW
          initH = innerH + TITLE
        }
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
            stuck={groupOf.has(id)}
          >
            {/* IPF windows carry a second `view:"3d"` figure → a 2D/3D toggle.
                Both iframes stay mounted; only the active view is shown so
                toggling is instant and the 3-D canvas isn't rebuilt. */}
            {(() => {
              const has3d = win.figures.some(f => f.view === '3d')
              const mode = ipfView[win.windowId] ?? '2d'
              return (
                <>
                  {has3d && (
                    <div style={styles.ipfToggle} data-testid={`ipf-view-toggle-${id}`}>
                      {(['2d', '3d'] as const).map(m => (
                        <button
                          key={m}
                          data-testid={`ipf-view-${m}-${id}`}
                          onClick={() => setIpfView(v => ({ ...v, [win.windowId]: m }))}
                          style={m === mode ? styles.ipfBtnActive : styles.ipfBtn}
                        >{m.toUpperCase()}</button>
                      ))}
                    </div>
                  )}
                  <div style={{ display: 'flex', width: '100%', height: '100%' }}>
                    {win.figures.map(fig => {
                      const show = !has3d || ((mode === '3d') === (fig.view === '3d'))
                      return (
                        <iframe
                          key={fig.figId}
                          ref={el => {
                            if (el) iframeRefs.current.set(fig.figId, el)
                            else iframeRefs.current.delete(fig.figId)
                          }}
                          src={fig.filePath ?? undefined}
                          // Replay any state (image data, selectors) that arrived before
                          // this iframe was listening — fixes the black-image race. Also
                          // size the figure to the iframe's ACTUAL box on load.
                          onLoad={(e) => {
                            replayState(fig.figId)
                            const el = e.currentTarget
                            window.electron.resizeFigure(
                              fig.figId,
                              Math.max(80, el.clientWidth),
                              Math.max(80, el.clientHeight),
                            )
                          }}
                          style={{ flex: 1, border: 'none', display: show ? 'block' : 'none', minWidth: 0 }}
                          title={fig.title}
                          data-testid={`figure-${fig.figId}`}
                        />
                      )
                    })}
                  </div>
                </>
              )
            })()}
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
  ipfToggle: {
    position: 'absolute', top: 4, right: 6, zIndex: 5,
    display: 'flex', gap: 2, background: 'rgba(24,24,37,0.85)',
    border: '1px solid #313244', borderRadius: 6, padding: 2,
  },
  ipfBtn: {
    background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
    fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 4,
  },
  ipfBtnActive: {
    background: '#89b4fa', border: 'none', color: '#11111b', cursor: 'pointer',
    fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 4,
  },
}
