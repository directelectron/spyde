import React, { useCallback, useRef, useState } from 'react'
import type { ToolbarAction } from '../kernel/SpyDEContext'
import { FloatingToolbar } from './FloatingToolbar'

interface Props {
  id: string
  title: string
  initialX: number
  initialY: number
  initialW: number
  initialH: number
  toolbarActions: ToolbarAction[]
  onClose: (id: string) => void
  onFocus: (id: string) => void
  onResize: (id: string, w: number, h: number) => void
  onAction: (action: string, windowId: number, params: Record<string, unknown>) => void
  zIndex: number
  windowId: number
  children: React.ReactNode
  // ── Stick / edge-snap grouping (optional) ──────────────────────────────────
  /** Live geometry of every window (incl. self) for edge snapping. */
  peers?: React.MutableRefObject<Map<string, Rect>>
  /** Report this window's geometry whenever it moves/resizes. */
  reportGeom?: (id: string, rect: Rect) => void
  /** During a drag, tell the host how far we moved so it can move stuck partners. */
  onStuckMove?: (id: string, dx: number, dy: number) => void
  /** Drag/resize finished — host decides whether new edges form a stick group. */
  onGestureEnd?: (id: string) => void
  /** A nudge applied by a stuck partner's drag: apply (dx,dy) when nonce changes. */
  groupNudge?: { dx: number; dy: number; nonce: number }
  /** An absolute rect pushed by a stuck partner's RESIZE (linked dim + follow). */
  rectOverride?: { x: number; y: number; w: number; h: number; nonce: number }
  /** Live resize → host links the shared dimension across the stick group. */
  onStuckResize?: (id: string, w: number, h: number) => void
  /** A vigorous back-and-forth "shake" → break the stick group apart. */
  onShake?: (id: string) => void
  /** True when this window belongs to a stick group (shows a small link badge). */
  stuck?: boolean
}

export interface Rect { x: number; y: number; w: number; h: number }

const TITLE_H = 32
const MIN_W = 300
const MIN_H = 200
const SNAP = 9        // px: edge-snap capture distance while dragging

// Snap the dragged rect's edges to any peer's edges (when the perpendicular
// spans overlap), so windows align cleanly. Returns the adjusted (x, y).
function snapToPeers(id: string, x: number, y: number, w: number, h: number,
                     peers: Map<string, Rect>): { x: number; y: number } {
  let sx = x, sy = y
  let bestDx = SNAP + 1, bestDy = SNAP + 1
  const r = { l: x, r: x + w, t: y, b: y + h }
  for (const [pid, p] of peers) {
    if (pid === id) continue
    const pr = { l: p.x, r: p.x + p.w, t: p.y, b: p.y + p.h }
    const vOverlap = r.t < pr.b + SNAP && r.b > pr.t - SNAP
    const hOverlap = r.l < pr.r + SNAP && r.r > pr.l - SNAP
    if (vOverlap) {
      for (const [a, b] of [[r.l, pr.r], [r.l, pr.l], [r.r, pr.l], [r.r, pr.r]]) {
        const d = b - a
        if (Math.abs(d) <= SNAP && Math.abs(d) < Math.abs(bestDx)) { bestDx = d; sx = x + d }
      }
    }
    if (hOverlap) {
      for (const [a, b] of [[r.t, pr.b], [r.t, pr.t], [r.b, pr.t], [r.b, pr.b]]) {
        const d = b - a
        if (Math.abs(d) <= SNAP && Math.abs(d) < Math.abs(bestDy)) { bestDy = d; sy = y + d }
      }
    }
  }
  return { x: sx, y: sy }
}

// Self-contained MDI subwindow. We do NOT use react-rnd: its
// getDerivedStateFromProps calls a dev `log()` that references `process`, which
// is undefined in the Electron renderer sandbox and crashes on controlled
// position/size updates. Drag AND resize are implemented manually with Pointer
// Capture so the gesture is delivered at the browser level even while the
// cursor is over the out-of-process figure iframe (whose canvas would otherwise
// steal the native pointer and freeze the interaction).
export function SubWindow({
  id, title, initialX, initialY, initialW, initialH,
  toolbarActions, onClose, onFocus, onResize, onAction,
  zIndex, windowId, children,
  peers, reportGeom, onStuckMove, onGestureEnd, groupNudge, stuck,
  rectOverride, onStuckResize, onShake,
}: Props) {
  const [maximized, setMaximized] = useState(false)
  const [pos, setPos] = useState({ x: initialX, y: initialY })
  const [size, setSize] = useState({ width: initialW, height: initialH })
  const [busy, setBusy] = useState(false)   // dragging or resizing → shield on
  const posRef = useRef(pos); posRef.current = pos
  // Honour a manually-set size from now on (don't re-adopt the backend default).
  const userResized = useRef(false)

  // Report live geometry so the host can edge-snap + move stick groups.
  React.useEffect(() => {
    reportGeom?.(id, { x: pos.x, y: pos.y, w: size.width, h: size.height })
  }, [pos.x, pos.y, size.width, size.height])  // eslint-disable-line react-hooks/exhaustive-deps

  // A stuck partner dragged → apply its delta to us.
  React.useEffect(() => {
    if (!groupNudge) return
    setPos(p => ({ x: Math.max(0, p.x + groupNudge.dx), y: Math.max(0, p.y + groupNudge.dy) }))
  }, [groupNudge?.nonce])  // eslint-disable-line react-hooks/exhaustive-deps

  // A stuck partner RESIZED → adopt the rect it computed for us (linked
  // dimension + follow the shared edge so we stay touching, never overlapping).
  React.useEffect(() => {
    if (!rectOverride) return
    userResized.current = true
    setPos({ x: rectOverride.x, y: rectOverride.y })
    setSize({ width: rectOverride.w, height: rectOverride.h })
    onResize(id, rectOverride.w, rectOverride.h)   // re-fit the figure to the new box
  }, [rectOverride?.nonce])  // eslint-disable-line react-hooks/exhaustive-deps

  // Adopt a new default size when the backend supplies it (e.g. the navigator's
  // image aspect arrives just after the window opens) — but never fight a size
  // the user has set by hand.
  React.useEffect(() => {
    if (!userResized.current) setSize({ width: initialW, height: initialH })
  }, [initialW, initialH])

  // The floating toolbar sits BELOW the window and reveals on hover (over the
  // window or the toolbar itself). A short hide delay covers the gap between the
  // window's bottom edge and the toolbar so moving toward it doesn't hide it.
  const [tbVisible, setTbVisible] = useState(false)
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const showToolbar = useCallback(() => {
    if (hideTimer.current) { clearTimeout(hideTimer.current); hideTimer.current = null }
    setTbVisible(true)
  }, [])
  const hideToolbar = useCallback(() => {
    if (hideTimer.current) clearTimeout(hideTimer.current)
    hideTimer.current = setTimeout(() => setTbVisible(false), 350)
  }, [])
  const gesture = useRef<
    | { kind: 'drag'; px: number; py: number; ox: number; oy: number }
    | { kind: 'resize'; px: number; py: number; w: number; h: number }
    | null
  >(null)
  // Horizontal direction-reversal tracking for the shake-to-break gesture.
  const shake = useRef({ dir: 0, count: 0, t0: 0, lastX: 0 })

  const hasToolbar = toolbarActions.length > 0
  const headerH = TITLE_H   // toolbar is now a right-edge rail, not a top bar

  // ── Drag (title bar) ────────────────────────────────────────────────────────
  const onTitleDown = (e: React.PointerEvent) => {
    if (maximized) return
    if ((e.target as HTMLElement).closest('button')) return  // let buttons click
    onFocus(id)
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId) } catch { /* */ }
    gesture.current = { kind: 'drag', px: e.clientX, py: e.clientY, ox: pos.x, oy: pos.y }
    shake.current = { dir: 0, count: 0, t0: performance.now(), lastX: e.clientX }
    setBusy(true)
  }
  // Count rapid horizontal reversals — a vigorous shake breaks the stick group.
  const detectShake = (clientX: number) => {
    const sh = shake.current
    const dd = clientX - sh.lastX
    if (Math.abs(dd) < 7) return
    const dir = Math.sign(dd)
    if (sh.dir !== 0 && dir !== sh.dir) {
      const now = performance.now()
      if (now - sh.t0 > 900) { sh.count = 0; sh.t0 = now }
      if (++sh.count >= 5) { onShake?.(id); sh.count = 0 }
    }
    sh.dir = dir
    sh.lastX = clientX
  }
  const onTitleMove = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g || g.kind !== 'drag') return
    if (onShake) detectShake(e.clientX)
    let nx = Math.max(0, g.ox + (e.clientX - g.px))
    let ny = Math.max(0, g.oy + (e.clientY - g.py))
    if (peers?.current) {
      const s = snapToPeers(id, nx, ny, size.width, size.height, peers.current)
      nx = s.x; ny = s.y
    }
    const dx = nx - posRef.current.x, dy = ny - posRef.current.y
    if (onStuckMove && (dx !== 0 || dy !== 0)) onStuckMove(id, dx, dy)
    setPos({ x: nx, y: ny })
  }

  // ── Resize (corner handle) ──────────────────────────────────────────────────
  const onResizeDown = (e: React.PointerEvent) => {
    if (maximized) return
    e.stopPropagation()
    userResized.current = true   // honour manual size from now on
    onFocus(id)
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId) } catch { /* */ }
    gesture.current = { kind: 'resize', px: e.clientX, py: e.clientY, w: size.width, h: size.height }
    setBusy(true)
  }
  const onResizeMove = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g || g.kind !== 'resize') return
    const w = Math.max(MIN_W, g.w + (e.clientX - g.px))
    const h = Math.max(MIN_H, g.h + (e.clientY - g.py))
    setSize({ width: w, height: h })
    // Tell the host so it links the shared dimension across the stick group
    // (and slides partners to follow the moving edge → they never overlap).
    onStuckResize?.(id, w, h)
  }

  const endGesture = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g) return
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId) } catch { /* */ }
    gesture.current = null
    setBusy(false)
    if (g.kind === 'resize') onResize(id, size.width, size.height)
    onGestureEnd?.(id)
  }

  const frame: React.CSSProperties = maximized
    ? { left: 0, top: 0, width: '100%', height: '100%' }
    : { left: pos.x, top: pos.y, width: size.width, height: size.height }

  return (
    <div
      data-testid="subwindow"
      style={{ ...styles.window, ...frame, zIndex }}
      onMouseDown={() => onFocus(id)}
      onMouseEnter={showToolbar}
      onMouseLeave={hideToolbar}
    >
      {/* Title bar (drag handle) */}
      <div
        className="spyde-titlebar"
        data-testid="subwindow-titlebar"
        style={styles.titleBar}
        onPointerDown={onTitleDown}
        onPointerMove={onTitleMove}
        onPointerUp={endGesture}
        onPointerCancel={endGesture}
        onDoubleClick={() => setMaximized(m => !m)}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 0 }}>
          {stuck && (
            <span data-testid="stuck-badge" title="Stuck to neighbours (moves as a group)"
              style={styles.stuckBadge}>🔗</span>
          )}
          <span data-testid="subwindow-title" style={styles.title}>{title}</span>
        </span>
        <div style={styles.controls}>
          <button style={styles.btn} onClick={() => setMaximized(m => !m)}>
            {maximized ? '❐' : '□'}
          </button>
          <button
            data-testid="close-btn"
            style={{ ...styles.btn, color: '#f38ba8' }}
            onClick={() => onClose(id)}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Figure body (clipped). The figure fills it; the toolbar floats over. */}
      <div style={{ ...styles.body, height: `calc(100% - ${headerH}px)` }}>
        {children}
        {busy && <div data-testid="drag-shield" style={styles.dragShield} />}
      </div>

      {/* Floating toolbar is parented to the window root (overflow:visible), so
          it tracks move/resize live and its popouts can extend past the edges. */}
      {hasToolbar && (
        <FloatingToolbar
          actions={toolbarActions}
          windowId={windowId}
          onAction={onAction}
          visible={tbVisible}
          onHoverShow={showToolbar}
          onHoverHide={hideToolbar}
        />
      )}

      {/* Resize handle (bottom-right) */}
      {!maximized && (
        <div
          data-testid="resize-handle"
          style={styles.resizeHandle}
          onPointerDown={onResizeDown}
          onPointerMove={onResizeMove}
          onPointerUp={endGesture}
          onPointerCancel={endGesture}
        />
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  window: {
    position: 'absolute',
    display: 'flex', flexDirection: 'column',
    background: '#1e1e2e',
    border: '1px solid #313244',
    borderRadius: 8,
    boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
    // visible (not hidden) so the action rail's param drawer can pop OUT the
    // right edge. The figure body does its own clipping + corner rounding.
    overflow: 'visible',
  },
  titleBar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '0 10px',
    height: TITLE_H,
    borderTopLeftRadius: 8, borderTopRightRadius: 8,
    background: '#181825',
    cursor: 'move', userSelect: 'none', touchAction: 'none',
    borderBottom: '1px solid #313244',
    flexShrink: 0,
  },
  title: { fontSize: 12, color: '#cdd6f4', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  stuckBadge: { fontSize: 10, lineHeight: 1, filter: 'grayscale(0.2)' },
  controls: { display: 'flex', gap: 4 },
  btn: {
    background: 'none', border: 'none', color: '#6c7086',
    cursor: 'pointer', fontSize: 13, padding: '2px 6px', borderRadius: 4,
  },
  body: {
    flex: 1, overflow: 'hidden', position: 'relative',
    borderBottomLeftRadius: 8, borderBottomRightRadius: 8,
  },
  dragShield: {
    position: 'absolute', inset: 0, zIndex: 10, cursor: 'grabbing',
  },
  resizeHandle: {
    position: 'absolute', right: 0, bottom: 0, width: 16, height: 16,
    cursor: 'nwse-resize', touchAction: 'none', zIndex: 11,
    // subtle corner grip
    background:
      'linear-gradient(135deg, transparent 50%, #45475a 50%, #45475a 60%, transparent 60%, transparent 70%, #45475a 70%, #45475a 80%, transparent 80%)',
  },
}
