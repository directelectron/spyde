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
}

export interface Rect { x: number; y: number; w: number; h: number }

const TITLE_H = 32
const MIN_W = 300
const MIN_H = 200

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
}: Props) {
  const [maximized, setMaximized] = useState(false)
  const [pos, setPos] = useState({ x: initialX, y: initialY })
  const [size, setSize] = useState({ width: initialW, height: initialH })
  const [busy, setBusy] = useState(false)   // dragging or resizing → shield on
  // Honour a manually-set size from now on (don't re-adopt the backend default).
  const userResized = useRef(false)

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

  const hasToolbar = toolbarActions.length > 0
  const headerH = TITLE_H   // toolbar is now a right-edge rail, not a top bar

  // ── Drag (title bar) ────────────────────────────────────────────────────────
  const onTitleDown = (e: React.PointerEvent) => {
    if (maximized) return
    if ((e.target as HTMLElement).closest('button')) return  // let buttons click
    onFocus(id)
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId) } catch { /* */ }
    gesture.current = { kind: 'drag', px: e.clientX, py: e.clientY, ox: pos.x, oy: pos.y }
    setBusy(true)
  }
  const onTitleMove = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g || g.kind !== 'drag') return
    const nx = Math.max(0, g.ox + (e.clientX - g.px))
    const ny = Math.max(0, g.oy + (e.clientY - g.py))
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
  }

  const endGesture = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g) return
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId) } catch { /* */ }
    gesture.current = null
    setBusy(false)
    if (g.kind === 'resize') onResize(id, size.width, size.height)
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
