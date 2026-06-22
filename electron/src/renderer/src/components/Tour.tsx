/**
 * Tour.tsx — the in-app interactive coachmark tour.
 *
 * Renders a Guide (from guides/, the single source shared with the docs site) as
 * a step-by-step walkthrough: a dimmed full-screen overlay with a "spotlight"
 * hole cut over the real UI element named by the current step's `anchor`, plus a
 * callout bubble (title + markdown body + Back/Next) placed next to it.
 *
 * The element is found live by its data-testid, so the tour points at the actual
 * running UI — not a screenshot. Anchors that can't be found (e.g. a wizard not
 * yet open) fall back to a centered bubble, so a tour never dead-ends.
 */
import React, { useEffect, useLayoutEffect, useState } from 'react'
import type { Guide, GuideStep, Placement } from '@guides/index'
import { Markdown } from '@guides/markdown'

const ACCENT = '#89b4fa'

function resolveAnchor(anchor: string | null): DOMRect | null {
  if (!anchor) return null
  const sel = /^[.#[]/.test(anchor) ? anchor : `[data-testid="${anchor}"]`
  const el = document.querySelector(sel) as HTMLElement | null
  if (!el) return null
  const r = el.getBoundingClientRect()
  // Treat an off-screen / zero-size element as "not found" so we center instead.
  if (r.width === 0 && r.height === 0) return null
  return r
}

/** Bubble position from the spotlight rect + desired placement (viewport-clamped). */
function bubblePos(
  rect: DOMRect | null,
  placement: Placement,
  bubbleW: number,
  bubbleH: number,
): { left: number; top: number } {
  const M = 14 // gap between spotlight and bubble
  const vw = window.innerWidth
  const vh = window.innerHeight
  if (!rect || placement === 'center') {
    return { left: (vw - bubbleW) / 2, top: (vh - bubbleH) / 2 }
  }
  let left = rect.left + rect.width / 2 - bubbleW / 2
  let top = rect.bottom + M
  if (placement === 'top') top = rect.top - bubbleH - M
  else if (placement === 'left') {
    left = rect.left - bubbleW - M
    top = rect.top + rect.height / 2 - bubbleH / 2
  } else if (placement === 'right') {
    left = rect.right + M
    top = rect.top + rect.height / 2 - bubbleH / 2
  }
  // Clamp into the viewport with an 8px margin.
  left = Math.max(8, Math.min(left, vw - bubbleW - 8))
  top = Math.max(8, Math.min(top, vh - bubbleH - 8))
  return { left, top }
}

export function Tour({ guide, onClose }: { guide: Guide; onClose: () => void }) {
  const [i, setI] = useState(0)
  const [rect, setRect] = useState<DOMRect | null>(null)
  const step: GuideStep = guide.steps[i]
  const last = i === guide.steps.length - 1

  // Re-measure the anchor on step change, on resize, and on a short poll (so a
  // wizard that opens slightly after the step advances still gets spotlighted).
  useLayoutEffect(() => {
    const measure = () => setRect(resolveAnchor(step.anchor))
    measure()
    const id = window.setInterval(measure, 250)
    window.addEventListener('resize', measure)
    return () => {
      window.clearInterval(id)
      window.removeEventListener('resize', measure)
    }
  }, [step.anchor])

  // Esc closes; ←/→ navigate.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowRight' && !last) setI((n) => n + 1)
      else if (e.key === 'ArrowLeft' && i > 0) setI((n) => n - 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [i, last, onClose])

  const bubbleW = 320
  const bubbleH = 200
  const { left, top } = bubblePos(rect, step.placement ?? 'bottom', bubbleW, bubbleH)
  const pad = 6 // spotlight padding around the element

  return (
    <div data-testid="tour-overlay" style={styles.overlay} onClick={onClose}>
      {/* Spotlight: a transparent ring with a huge box-shadow dims everything
          else, leaving the anchored element visible and un-clickable-through. */}
      {rect && (
        <div
          data-testid="tour-spotlight"
          style={{
            ...styles.spotlight,
            left: rect.left - pad,
            top: rect.top - pad,
            width: rect.width + pad * 2,
            height: rect.height + pad * 2,
          }}
        />
      )}
      {!rect && <div style={styles.dimAll} />}

      {/* Callout bubble. Stop click-through so buttons work. */}
      <div
        data-testid="tour-bubble"
        style={{ ...styles.bubble, left, top, width: bubbleW }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={styles.header}>
          <span style={styles.stepCount}>
            {i + 1} / {guide.steps.length}
          </span>
          <button data-testid="tour-close" style={styles.closeBtn} onClick={onClose}>
            ✕
          </button>
        </div>
        <h3 style={styles.title}>{step.title}</h3>
        <div style={styles.body}>
          <Markdown text={step.body} styles={{ paragraph: styles.p, callout: styles.callout }} />
        </div>
        <div style={styles.footer}>
          <button
            data-testid="tour-back"
            style={{ ...styles.navBtn, visibility: i > 0 ? 'visible' : 'hidden' }}
            onClick={() => setI((n) => n - 1)}
          >
            ‹ Back
          </button>
          {last ? (
            <button data-testid="tour-done" style={styles.primaryBtn} onClick={onClose}>
              Done
            </button>
          ) : (
            <button
              data-testid="tour-next"
              style={styles.primaryBtn}
              onClick={() => setI((n) => n + 1)}
            >
              Next ›
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: { position: 'fixed', inset: 0, zIndex: 9000 },
  dimAll: { position: 'absolute', inset: 0, background: 'rgba(17,17,27,0.72)' },
  spotlight: {
    position: 'absolute',
    borderRadius: 8,
    boxShadow: '0 0 0 9999px rgba(17,17,27,0.72)',
    border: `2px solid ${ACCENT}`,
    pointerEvents: 'none',
    transition: 'left 140ms ease, top 140ms ease, width 140ms ease, height 140ms ease',
  },
  bubble: {
    position: 'absolute',
    background: '#1e1e2e',
    border: '1px solid #313244',
    borderRadius: 10,
    padding: 14,
    boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
    color: '#cdd6f4',
    fontSize: 13,
    transition: 'left 140ms ease, top 140ms ease',
  },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  stepCount: { fontSize: 11, color: '#6c7086', letterSpacing: 0.5 },
  closeBtn: {
    background: 'transparent', border: 'none', color: '#6c7086',
    cursor: 'pointer', fontSize: 13, padding: 2,
  },
  title: { margin: '6px 0 4px', fontSize: 15, color: '#cdd6f4', fontWeight: 600 },
  body: { lineHeight: 1.5, color: '#bac2de' },
  p: { margin: '6px 0' },
  callout: {
    margin: '8px 0', padding: '8px 10px', borderRadius: 6,
    background: 'rgba(137,180,250,0.10)', borderLeft: `3px solid ${ACCENT}`,
    color: '#cdd6f4',
  },
  footer: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12,
  },
  navBtn: {
    background: 'transparent', border: '1px solid #313244', color: '#cdd6f4',
    borderRadius: 6, padding: '5px 12px', cursor: 'pointer', fontSize: 12,
  },
  primaryBtn: {
    background: ACCENT, border: 'none', color: '#11111b', fontWeight: 600,
    borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontSize: 12,
  },
}
