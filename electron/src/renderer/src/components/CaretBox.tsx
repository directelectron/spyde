/**
 * CaretBox.tsx — the dock's small anchored popover.
 *
 * A caret-pointed box under whatever was clicked: the shape the metadata panel
 * uses for field details and the chunk viewer. `position: fixed` so the dock's
 * scrolling metadata panel can't clip it, clamped to the window, and dismissed
 * by Escape or a press outside.
 *
 * The ANCHOR is deliberately excluded from that outside-press check: the anchor's
 * own onClick toggles the box shut, and closing here first would let that click
 * re-open it — so it could never be dismissed by clicking the thing that opened
 * it. (A full-screen backdrop would do the same by swallowing the click; that is
 * why this is not a modal.)
 */
import React from 'react'
import { CHROME_Z } from './WizardShell'

export function CaretBox({ anchor, el, width = 236, testid, onClose, children }: {
  /** Anchor rect, captured at click time. */
  anchor: DOMRect
  /** Anchor element, so a press on it is not an "outside" press. */
  el: HTMLElement
  width?: number
  testid: string
  onClose: () => void
  children: React.ReactNode
}) {
  const left = Math.min(Math.max(8, anchor.left - 8), window.innerWidth - width - 8)
  const top = Math.min(anchor.bottom + 8, window.innerHeight - 120)

  React.useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    const down = (e: PointerEvent) => {
      if (!el.contains(e.target as Node)) onClose()
    }
    window.addEventListener('keydown', key)
    window.addEventListener('pointerdown', down)
    return () => {
      window.removeEventListener('keydown', key)
      window.removeEventListener('pointerdown', down)
    }
  }, [onClose, el])

  return (
    <div data-testid={testid} style={{ ...S.pop, left, top, width }}
      onPointerDown={(e) => e.stopPropagation()}>
      <span style={{ ...S.caret, left: Math.min(anchor.left + anchor.width / 2 - left - 5, width - 16) }} />
      {/* The scroll lives on this INNER box: the caret sits outside the border,
          so an `overflow` on the outer one would clip it away. */}
      <div style={S.body}>{children}</div>
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  pop: {
    position: 'fixed', zIndex: CHROME_Z + 40,
    background: '#181825', border: '1px solid #45475a', borderRadius: 6,
    boxShadow: '0 6px 20px rgba(0,0,0,0.45)', padding: '8px 10px',
  },
  body: {
    display: 'flex', flexDirection: 'column', gap: 4,
    maxHeight: '60vh', overflowY: 'auto',
  },
  caret: {
    position: 'absolute', top: -5, width: 8, height: 8,
    background: '#181825', borderLeft: '1px solid #45475a',
    borderTop: '1px solid #45475a', transform: 'rotate(45deg)',
  },
}
