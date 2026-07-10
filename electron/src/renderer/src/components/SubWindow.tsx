import React, { useCallback, useRef, useState } from 'react'
import type { ToolbarAction } from '../kernel/SpyDEContext'
import { FloatingToolbar } from './FloatingToolbar'
import { WINDOW_DRAG_MIME, SIGNAL_REF_DRAG_MIME } from '../kernel/dnd'

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
  onMinimize?: (id: string) => void
  onResize: (id: string, w: number, h: number) => void
  onAction: (action: string, windowId: number, params: Record<string, unknown>) => void
  zIndex: number
  windowId: number
  children: React.ReactNode
  /** Minimized: keep the window (and its figure iframe) mounted but invisible —
   *  it is listed in the MDI top bar and restored from there. */
  hidden?: boolean
  /** Navigator windows accept a dragged signal window on their titlebar —
   *  the dropped signal becomes a named navigator of this window's tree. */
  acceptSignalDrop?: boolean
  onSignalDrop?: (sourceWindowId: number) => void
  // Live layout coordination with MDIArea (all optional so older callers/tests
  // that don't pass them still work with sane defaults).
  areaSize?: { w: number; h: number }
  otherRects?: Rect[]
  onLiveRect?: (id: string, rect: Rect) => void
  // Bumped by MDIArea's Tile action: a new `gen` forces this rect to be
  // adopted even if the user had manually moved/resized the window.
  forced?: { gen: number; rect: Rect }
}

export interface Rect { x: number; y: number; w: number; h: number }

const TITLE_H = 32
const MIN_W = 300
const MIN_H = 200
// Distance (px) within which a dragged/resized edge snaps to another window's
// edge or to the area bounds.
const SNAP_DIST = 10
// However close to the edge the user drags, at least this many px of the
// titlebar must stay reachable within the visible area.
const MIN_VISIBLE_TITLEBAR = 40

// Snap a moving window's position against other windows' edges + the area
// bounds: for each axis, if any candidate target edge is within SNAP_DIST of
// the corresponding moving edge, pull the whole rect flush to it.
function snapPosition(
  x: number, y: number, w: number, h: number,
  others: Rect[], areaW: number, areaH: number,
): { x: number; y: number } {
  const xTargets: number[] = [0, areaW - w]
  const yTargets: number[] = [0, areaH - h]
  for (const o of others) {
    xTargets.push(o.x, o.x - w, o.x + o.w, o.x + o.w - w)
    yTargets.push(o.y, o.y - h, o.y + o.h, o.y + o.h - h)
  }
  let sx = x, sy = y, bestDx = SNAP_DIST, bestDy = SNAP_DIST
  for (const t of xTargets) {
    const d = Math.abs(x - t)
    if (d < bestDx) { bestDx = d; sx = t }
  }
  for (const t of yTargets) {
    const d = Math.abs(y - t)
    if (d < bestDy) { bestDy = d; sy = t }
  }
  return { x: sx, y: sy }
}

// Snap a resizing window's bottom-right corner against other windows' facing
// edges + the area bounds, independently per axis.
function snapSize(
  x: number, y: number, w: number, h: number,
  others: Rect[], areaW: number, areaH: number,
): { w: number; h: number } {
  const rightTargets: number[] = [areaW]
  const bottomTargets: number[] = [areaH]
  for (const o of others) {
    rightTargets.push(o.x, o.x + o.w)
    bottomTargets.push(o.y, o.y + o.h)
  }
  let sw = w, sh = h, bestDw = SNAP_DIST, bestDh = SNAP_DIST
  for (const t of rightTargets) {
    const d = Math.abs((x + w) - t)
    if (d < bestDw) { bestDw = d; sw = t - x }
  }
  for (const t of bottomTargets) {
    const d = Math.abs((y + h) - t)
    if (d < bestDh) { bestDh = d; sh = t - y }
  }
  return { w: Math.max(MIN_W, sw), h: Math.max(MIN_H, sh) }
}

// Clamp so the titlebar (the only drag handle) is always at least partly
// reachable within the visible area — a window can never be dragged fully off
// screen and become unrecoverable.
function clampToVisible(x: number, y: number, w: number, areaW: number, areaH: number): { x: number; y: number } {
  const minX = -w + MIN_VISIBLE_TITLEBAR
  const maxX = areaW - MIN_VISIBLE_TITLEBAR
  const minY = 0
  const maxY = Math.max(0, areaH - TITLE_H)
  return {
    x: Math.min(Math.max(x, minX), Math.max(minX, maxX)),
    y: Math.min(Math.max(y, minY), maxY),
  }
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
  toolbarActions, onClose, onFocus, onMinimize, onResize, onAction,
  zIndex, windowId, children, hidden = false,
  acceptSignalDrop = false, onSignalDrop,
  areaSize, otherRects, onLiveRect, forced,
}: Props) {
  const [maximized, setMaximized] = useState(false)
  const [dropHover, setDropHover] = useState(false)
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

  // Tile (or any other forced-layout action) bumps `forced.gen` — adopt its
  // rect unconditionally, overriding a manual move/resize.
  const lastForcedGen = useRef<number | null>(null)
  React.useEffect(() => {
    if (!forced || forced.gen === lastForcedGen.current) return
    lastForcedGen.current = forced.gen
    userResized.current = true
    setPos({ x: forced.rect.x, y: forced.rect.y })
    setSize({ width: forced.rect.w, height: forced.rect.h })
  }, [forced])

  // Report the live rect on every change so MDIArea can snap OTHER windows
  // against this one's current position/size (not just its rest position) and
  // so a freshly-opened window's free-slot search sees real sizes.
  React.useEffect(() => {
    onLiveRect?.(id, { x: pos.x, y: pos.y, w: size.width, h: size.height })
  }, [id, pos.x, pos.y, size.width, size.height, onLiveRect])

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
  const headerH = TITLE_H
  const areaW = areaSize?.w ?? 4000   // generous fallback so clamping is a no-op
  const areaH = areaSize?.h ?? 4000   // if MDIArea hasn't measured yet
  const others = otherRects ?? []

  // ── Drag (title bar) ────────────────────────────────────────────────────────
  const onTitleDown = (e: React.PointerEvent) => {
    if (maximized) return
    if ((e.target as HTMLElement).closest('button')) return  // let buttons click
    // The signal-drag grips (navigator-add + console-ref) start an HTML5 drag,
    // not a window move.
    if ((e.target as HTMLElement).closest('[data-testid="signal-drag-handle"]')) return
    if ((e.target as HTMLElement).closest('[data-testid="console-ref-handle"]')) return
    onFocus(id)
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId) } catch { /* */ }
    gesture.current = { kind: 'drag', px: e.clientX, py: e.clientY, ox: pos.x, oy: pos.y }
    setBusy(true)
  }
  const onTitleMove = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g || g.kind !== 'drag') return
    const rawX = g.ox + (e.clientX - g.px)
    const rawY = g.oy + (e.clientY - g.py)
    const snapped = snapPosition(rawX, rawY, size.width, size.height, others, areaW, areaH)
    const clamped = clampToVisible(snapped.x, snapped.y, size.width, areaW, areaH)
    setPos(clamped)
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
    const rawW = Math.max(MIN_W, g.w + (e.clientX - g.px))
    const rawH = Math.max(MIN_H, g.h + (e.clientY - g.py))
    const snapped = snapSize(pos.x, pos.y, rawW, rawH, others, areaW, areaH)
    setSize({ width: snapped.w, height: snapped.h })
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

  // The toolbar (and its caret) render inside this window's stacking context,
  // so they share the window's z-level — clicking the window/toolbar focuses
  // it, which raises it via the managed focus z-order. (An always-on hover
  // raise was tried and rejected: windows jumping above siblings on mere
  // hover made the stacking feel random.)
  const rect = maximized
    ? { x: 0, y: 0, w: areaW, h: areaH }
    : { x: pos.x, y: pos.y, w: size.width, h: size.height }
  // The bar hangs below the window; fall back to inside the bottom edge only
  // when there is no room below (maximized / dragged to the area bottom).
  const barInside = maximized || rect.y + rect.h + 46 > areaH

  return (
    <div
      data-testid="subwindow"
      style={{ ...styles.window, ...frame, zIndex, ...(hidden ? { display: 'none' } : {}) }}
      onMouseDown={() => onFocus(id)}
      onMouseEnter={showToolbar}
      onMouseLeave={hideToolbar}
    >
      {/* Title bar (drag handle) */}
      <div
        className="spyde-titlebar"
        data-testid="subwindow-titlebar"
        style={{ ...styles.titleBar, ...(dropHover ? styles.titleBarDrop : {}) }}
        onPointerDown={onTitleDown}
        onPointerMove={onTitleMove}
        onPointerUp={endGesture}
        onPointerCancel={endGesture}
        onDoubleClick={() => setMaximized(m => !m)}
        onDragOver={(e) => {
          if (!acceptSignalDrop || !e.dataTransfer.types.includes(WINDOW_DRAG_MIME)) return
          e.preventDefault()
          e.dataTransfer.dropEffect = 'copy'
          setDropHover(true)
        }}
        onDragLeave={() => setDropHover(false)}
        onDrop={(e) => {
          setDropHover(false)
          if (!acceptSignalDrop) return
          const raw = e.dataTransfer.getData(WINDOW_DRAG_MIME)
          if (!raw) return
          e.preventDefault()
          const src = parseInt(raw, 10)
          if (Number.isFinite(src) && src !== windowId) onSignalDrop?.(src)
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 0 }}>
          {/* Grip: drag this signal onto a navigator's titlebar to add it as a
              named navigator (a plain titlebar drag still moves the window). */}
          <span
            data-testid="signal-drag-handle"
            title="Drag onto a navigator to add this signal as a navigator"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData(WINDOW_DRAG_MIME, String(windowId))
              e.dataTransfer.effectAllowed = 'copy'
            }}
            style={styles.dragHandle}
          >⠿</span>
          {/* Grip: drag this signal into the console input to insert its variable
              name at the caret (resolved from the latest console_vars "signal"
              rows by windowId). Distinct MIME/handle from the navigator grip
              above — dropping this one on a navigator titlebar does nothing. */}
          <span
            data-testid="console-ref-handle"
            title="Drag into the console to insert this signal's variable name"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData(SIGNAL_REF_DRAG_MIME, JSON.stringify({ windowId }))
              e.dataTransfer.effectAllowed = 'copy'
            }}
            style={styles.consoleRefHandle}
          >»</span>
          <span data-testid="subwindow-title" style={styles.title}>{title}</span>
        </span>
        <div style={styles.controls}>
          {onMinimize && (
            <button
              data-testid="minimize-btn"
              style={styles.btn}
              title="Minimize"
              onClick={() => onMinimize(id)}
            >
              –
            </button>
          )}
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
          winRect={rect}
          areaSize={{ w: areaW, h: areaH }}
          inside={barInside}
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
  titleBarDrop: { background: '#2a2a44', borderBottom: '1px solid #89b4fa' },
  dragHandle: {
    color: '#6c7086', fontSize: 11, cursor: 'grab', flexShrink: 0,
    userSelect: 'none',
  },
  consoleRefHandle: {
    color: '#89b4fa', fontSize: 11, fontWeight: 700, cursor: 'grab', flexShrink: 0,
    userSelect: 'none',
  },
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
