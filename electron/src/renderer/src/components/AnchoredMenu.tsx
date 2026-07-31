/**
 * AnchoredMenu.tsx — a dropdown menu panel anchored to the button that opened
 * it, `position: fixed` so no scrolling ancestor can clip it.
 *
 * WHY THIS EXISTS. The Report sidebar's body is `overflowY:'auto'`, so an
 * `absolute` menu inside it is CUT at the scroller's edge: the "+ Add slide ▾"
 * starter menu (`bottom:100%`, i.e. opening upward) sat close enough to the top
 * of the body that its first item — "Add text slide" — was sliced through the
 * middle. `toBeVisible()` and a typecheck both pass on that; only pixels show it.
 *
 * This is the CaretBox idiom (`position: fixed`, anchored to the element that
 * opened it, clamped to the window, dismissed by Escape or an outside press)
 * minus the caret and plus FLIPPING: it opens on the `prefer` side when the menu
 * fits there and on the other side when it does not, so a button at the very
 * bottom of the window opens upward and one at the very top opens downward. If
 * neither side can hold it (a very short window) it takes the roomier side and
 * scrolls internally — never overflowing the viewport.
 *
 * The ANCHOR is deliberately excluded from the outside-press check, for the same
 * reason CaretBox excludes it: the anchor's own onClick toggles the menu shut, so
 * closing on its pointerdown first would let that click immediately re-open it.
 *
 * Position is measured from the LIVE anchor rect and recomputed on scroll/resize,
 * so the panel tracks its button instead of detaching from it.
 */
import React from 'react'

/** Gap between the anchor and the panel. */
const GAP = 4
/** Space kept clear of every window edge. */
const MARGIN = 8

export type MenuAlign = 'stretch' | 'left' | 'right'

interface Box {
  top: number
  left: number
  width?: number
  maxHeight: number
  /** Which side it ended up on — exposed as data-placement for tests. */
  up: boolean
}

export function AnchoredMenu({
  anchorEl, testid, onClose, children,
  prefer = 'down', align = 'stretch', minWidth = 0, zIndex = 9400,
}: {
  /** The element the menu hangs off (its own click toggles the menu). */
  anchorEl: HTMLElement
  testid: string
  onClose: () => void
  children: React.ReactNode
  /** Side to open on when it fits there. Flips to the other side otherwise. */
  prefer?: 'down' | 'up'
  /** 'stretch' matches the anchor's width (the old left:0/right:0 look). */
  align?: MenuAlign
  minWidth?: number
  zIndex?: number
}) {
  const ref = React.useRef<HTMLDivElement>(null)
  // Natural (unconstrained) size, measured on the first hidden layout pass and
  // reused afterwards — once maxHeight is applied the element can no longer
  // report how tall it WANTS to be.
  const natural = React.useRef<{ h: number; w: number } | null>(null)
  const [box, setBox] = React.useState<Box | null>(null)

  const place = React.useCallback(() => {
    const el = ref.current
    if (!el || !anchorEl.isConnected) return
    if (!natural.current) natural.current = { h: el.offsetHeight, w: el.offsetWidth }
    const { h: wantH, w: naturalW } = natural.current
    const a = anchorEl.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight

    // Usable room on each side of the anchor: preferred side if it fits, else
    // the other side if IT fits, else whichever is roomier (and scroll inside).
    const roomUp = a.top - GAP - MARGIN
    const roomDown = vh - a.bottom - GAP - MARGIN
    const up = prefer === 'up'
      ? (wantH <= roomUp ? true : wantH <= roomDown ? false : roomUp >= roomDown)
      : (wantH <= roomDown ? false : wantH <= roomUp ? true : roomUp > roomDown)

    const h = Math.max(0, Math.min(wantH, up ? roomUp : roomDown))
    const top = up
      ? Math.max(MARGIN, a.top - GAP - h)
      : Math.min(a.bottom + GAP, Math.max(MARGIN, vh - MARGIN - h))

    const width = align === 'stretch' ? Math.max(minWidth, a.width) : Math.max(minWidth, naturalW)
    const rawLeft = align === 'right' ? a.right - width : a.left
    const left = Math.min(Math.max(MARGIN, rawLeft), Math.max(MARGIN, vw - width - MARGIN))

    setBox({ top, left, width: align === 'stretch' ? width : undefined, maxHeight: h, up })
  }, [anchorEl, prefer, align, minWidth])

  // Measure + position before paint, so the panel is never seen in the wrong spot.
  React.useLayoutEffect(() => { place() }, [place])

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node
      if (ref.current?.contains(t)) return
      if (anchorEl.contains(t)) return   // see the header note
      onClose()
    }
    const reflow = () => place()
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onDown)
    // capture:true — the scroll that moves the anchor is an inner container's,
    // and those do not bubble to window.
    window.addEventListener('scroll', reflow, true)
    window.addEventListener('resize', reflow)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onDown)
      window.removeEventListener('scroll', reflow, true)
      window.removeEventListener('resize', reflow)
    }
  }, [anchorEl, onClose, place])

  return (
    <div
      ref={ref}
      data-testid={testid}
      data-placement={box ? (box.up ? 'up' : 'down') : undefined}
      role="menu"
      style={{
        ...S.panel,
        zIndex,
        ...(box
          ? { top: box.top, left: box.left, width: box.width, maxHeight: box.maxHeight }
          // First pass: laid out but not yet placed — measured, not painted.
          : { top: 0, left: 0, visibility: 'hidden' }),
      }}
    >
      {children}
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  panel: {
    position: 'fixed',
    background: '#1e1e2e', border: '1px solid #45475a', borderRadius: 6,
    padding: 4, display: 'flex', flexDirection: 'column', gap: 1,
    overflowY: 'auto',
    boxShadow: '0 6px 22px rgba(0,0,0,0.5)',
  },
}
