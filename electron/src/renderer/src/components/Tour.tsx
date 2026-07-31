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
 *
 * THE TOUR IS NOT A MODAL — it is CLICK-THROUGH, and only ✕/Done exits.
 * ------------------------------------------------------------------
 * The overlay used to be `onClick={onClose}` over an opaque hit-target, which
 * broke the tour two ways at once: a stray click anywhere killed the walkthrough
 * mid-way, AND the step "click the peak-finding tool" was impossible to follow
 * because the overlay swallowed that very click. Both are the same root cause,
 * so both get the same fix: every layer of the overlay is `pointerEvents:'none'`
 * (the dim, the spotlight ring, the root), and ONLY the callout bubble takes
 * pointer events. So clicks land on the REAL UI underneath — the user performs
 * each step for real while the spotlight follows — and the tour is dismissed
 * exclusively by its ✕ (or Done on the last step, or Esc). Do not put an
 * `onClick={onClose}` back on a full-screen layer here; the app's modal idiom
 * (PeriodicTable/GpuHelpDialog backdrops close on outside click) deliberately
 * does NOT apply to a coachmark tour, whose whole job is to sit over live UI.
 */
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Guide, GuideStep, Placement } from '@guides/index'
import { Markdown } from '@guides/markdown'
import { useSpyDE } from '../kernel/SpyDEContext'
import { runDrive } from '../kernel/guideDriver'

const ACCENT = '#89b4fa'

function resolveAnchor(anchor: string | null): DOMRect | null {
  if (!anchor) return null
  const sel = /^[.#[]/.test(anchor) ? anchor : `[data-testid="${anchor}"]`
  const el = document.querySelector(sel) as HTMLElement | null
  if (!el) return null
  const r = el.getBoundingClientRect()
  // Treat an off-screen / zero-size element as "not found" so we center instead.
  if (r.width === 0 && r.height === 0) return null
  // Treat an element that is present in the DOM but not actually VISIBLE
  // (opacity 0 / hidden / display none) as "not found" too — otherwise we draw a
  // spotlight ring around an invisible control. The floating plot toolbar, for
  // example, is laid out but opacity:0 until the window is hovered; without this
  // check the "plot toolbar" step would highlight an empty box.
  const cs = window.getComputedStyle(el)
  if (
    cs.visibility === 'hidden' ||
    cs.display === 'none' ||
    Number(cs.opacity) === 0
  ) {
    return null
  }
  return r
}

/** Do two boxes overlap? Used to keep the bubble off the spotlit control. */
function overlaps(
  a: { left: number; top: number; w: number; h: number },
  r: DOMRect,
): boolean {
  return (
    a.left < r.right && a.left + a.w > r.left &&
    a.top < r.bottom && a.top + a.h > r.top
  )
}

/**
 * Bubble position from the spotlight rect + desired placement (viewport-clamped).
 *
 * The bubble is the ONE part of the tour that takes pointer events, so wherever
 * it lands is un-clickable. It must therefore never come to rest on top of the
 * element the step is telling you to click. The authored `placement` puts it
 * beside the anchor, but the viewport clamp can shove it back over — so after
 * clamping we check for an overlap and, if there is one, try the other
 * placements in order and keep the first that clears the anchor.
 */
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
  const at = (p: Placement) => {
    let left = rect.left + rect.width / 2 - bubbleW / 2
    let top = rect.bottom + M
    if (p === 'top') top = rect.top - bubbleH - M
    else if (p === 'left') {
      left = rect.left - bubbleW - M
      top = rect.top + rect.height / 2 - bubbleH / 2
    } else if (p === 'right') {
      left = rect.right + M
      top = rect.top + rect.height / 2 - bubbleH / 2
    }
    // Clamp into the viewport with an 8px margin.
    left = Math.max(8, Math.min(left, vw - bubbleW - 8))
    top = Math.max(8, Math.min(top, vh - bubbleH - 8))
    return { left, top }
  }
  // Preferred placement first, then the alternatives.
  const order: Placement[] = [placement, 'bottom', 'top', 'right', 'left']
  for (const p of order) {
    const pos = at(p)
    if (!overlaps({ ...pos, w: bubbleW, h: bubbleH }, rect)) return pos
  }
  // Nothing clears it (a huge anchor, or a tiny window) — keep the authored
  // placement rather than jumping somewhere arbitrary.
  return at(placement)
}

/**
 * The synthetic final step built from `guide.info` — the "More info" the user
 * asked to land on at the END of a walkthrough. It is NOT authored in the guide's
 * `steps` (the docs page and the Help → Info dialog render the same `guide.info`
 * their own way), so it can never drift between the three surfaces.
 */
function moreInfoStep(guide: Guide): GuideStep | null {
  if (!guide.info) return null
  return {
    anchor: null,
    title: 'More info',
    body: guide.info.blurb,
    placement: 'center',
  }
}

export function Tour({ guide, onClose }: { guide: Guide; onClose: () => void }) {
  const [i, setI] = useState(0)
  const [rect, setRect] = useState<DOMRect | null>(null)
  // Auto-load (guide.autoload) lifecycle: 'idle' → 'loading' → 'done'|'error'.
  const [autoloadState, setAutoloadState] =
    useState<'idle' | 'loading' | 'done' | 'error'>(guide.autoload ? 'loading' : 'done')
  // The walked sequence = the authored steps + (when the guide has background
  // reading) one trailing "More info" step.
  const infoStep = moreInfoStep(guide)
  const steps: GuideStep[] = infoStep ? [...guide.steps, infoStep] : guide.steps
  const step: GuideStep = steps[Math.min(i, steps.length - 1)]
  const last = i === steps.length - 1
  const onInfoStep = infoStep != null && i === steps.length - 1

  const { sendAction } = useSpyDE()
  // sendAction identity changes each render; route through a ref so the effects
  // below don't re-fire on every render.
  const sendRef = useRef(sendAction)
  sendRef.current = sendAction
  // The tour is purely descriptive: it loads the tutorial dataset once on open,
  // spotlights each step's UI, and closes the dataset again on exit. There is no
  // per-step "Show me" auto-drive — the user follows the highlighted controls
  // themselves (the auto-drive was removed; it also double-loaded the data).

  // Track whether this tour actually loaded a tutorial dataset, so teardown only
  // closes data the tour opened (never the user's own). A ref (not state) so the
  // unmount cleanup reads the latest value without re-running.
  const didAutoloadRef = useRef(false)
  // StrictMode double-invokes effects in dev; guard the autoload so a re-mount
  // doesn't fire a second load.
  const autoloadStartedRef = useRef(false)
  // The callout bubble, measured for placement (see `bubbleH` below).
  const bubbleRef = useRef<HTMLDivElement>(null)

  // Auto-load the tutorial dataset ONCE on open, before showing step 1. Errors
  // are swallowed into an 'error' state (the tour still works; the dataset just
  // wasn't loaded for the user). On tour EXIT, close the tutorial dataset(s) so
  // the dummy data doesn't linger after the walkthrough.
  //
  // SELF-CONTAINED, both ends: `tutorial_session_begin` is sent BEFORE the load
  // so the backend also records everything the walkthrough goes on to CREATE
  // (a find-vectors result tree, a virtual image, an orientation map — see
  // Session._add_signal). `tutorial_close_all` on unmount then closes the whole
  // set, not just the dataset that was auto-loaded. Data backed by a real file
  // on disk is never recorded, so a user's own open file is never closed.
  useEffect(() => {
    if (!guide.autoload) return
    if (autoloadStartedRef.current) return
    autoloadStartedRef.current = true
    let cancelled = false
    setAutoloadState('loading')
    // Ordered ahead of the load on the same serial stdin pipe.
    sendRef.current('tutorial_session_begin', {})
    didAutoloadRef.current = true
    runDrive(guide.autoload, guide.steps[0], { sendAction: sendRef.current })
      .then(() => { if (!cancelled) setAutoloadState('done') })
      .catch(() => { if (!cancelled) setAutoloadState('error') })
    return () => {
      cancelled = true
      // Tour is unmounting (Done / ✕ / Esc): tear down the tutorial data it
      // loaded AND anything the walkthrough derived from it, so the user is left
      // with a clean workspace.
      if (didAutoloadRef.current) sendRef.current('tutorial_close_all', {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [guide.id])

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

  // Esc closes; ←/→ navigate. (Esc is the keyboard equivalent of ✕ — a
  // deliberate exit, not the accidental "clicked away" the tour must survive.)
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
  // MEASURE the bubble rather than guessing its height. Step bodies vary a lot
  // (the trailing "More info" step carries a blurb plus a link list and is
  // several times taller than a one-line step), and a fixed estimate positioned
  // a tall bubble so its bottom — the Further-reading links and the Done button
  // — fell off the screen. Measuring means `bubblePos`'s viewport clamp gets the
  // real height and the whole bubble is always on-screen; anything taller than
  // the viewport still scrolls inside itself (maxHeight/overflowY below).
  const [bubbleH, setBubbleH] = useState(320)
  useLayoutEffect(() => {
    const el = bubbleRef.current
    if (!el) return
    const measure = () => {
      const h = el.getBoundingClientRect().height
      // Only commit real changes — an unconditional setState here would loop
      // (position change → layout → measure → position…).
      setBubbleH((cur) => (Math.abs(cur - h) > 1 ? h : cur))
    }
    measure()
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(measure) : null
    ro?.observe(el)
    window.addEventListener('resize', measure)
    return () => { ro?.disconnect(); window.removeEventListener('resize', measure) }
  }, [i, autoloadState])
  const { left, top } = bubblePos(rect, step.placement ?? 'bottom', bubbleW, bubbleH)
  const pad = 6 // spotlight padding around the element

  return (
    // No onClick here, and pointerEvents:'none' on this root — see the file
    // header. Clicking "away" must NOT exit the tour, and must reach the real
    // UI so the user can actually perform the highlighted step.
    <div data-testid="tour-overlay" style={styles.overlay}>
      {/* Keyframes for the "Loading tutorial data…" spinner (inline styles can't
          declare @keyframes). */}
      <style>{'@keyframes spyde-tour-spin{to{transform:rotate(360deg)}}'}</style>
      {/* Spotlight: a transparent ring with a huge box-shadow dims everything
          else, leaving the anchored element visible AND clickable through. */}
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

      {/* Callout bubble — the ONLY layer that takes pointer events. */}
      <div
        ref={bubbleRef}
        data-testid="tour-bubble"
        style={{ ...styles.bubble, left, top, width: bubbleW }}
      >
        <div style={styles.header}>
          <span style={styles.stepCount}>
            {i + 1} / {steps.length}
          </span>
          <button
            data-testid="tour-close"
            style={styles.closeBtn}
            title="Exit the tour"
            aria-label="Exit the tour"
            onClick={onClose}
          >
            ✕
          </button>
        </div>
        <h3 style={styles.title}>{step.title}</h3>
        <div style={styles.body}>
          <Markdown text={step.body} styles={{ paragraph: styles.p, callout: styles.callout }} />
        </div>

        {/* Final "More info" step: the technique's further reading, opened in the
            user's browser. Same `guide.info` the Help → Info dialog and the docs
            page render. */}
        {onInfoStep && guide.info && (
          <div data-testid="tour-more-info" style={styles.links}>
            <div style={styles.linksLabel}>Further reading</div>
            {guide.info.links.map((l) => (
              <button
                key={l.url}
                data-testid={`tour-info-link-${l.url}`}
                style={styles.linkBtn}
                onClick={() => window.electron?.openExternal?.(l.url)}
              >
                <span style={styles.linkLabel}>{l.label} ↗</span>
                {l.note && <span style={styles.linkNote}>{l.note}</span>}
              </button>
            ))}
          </div>
        )}

        {/* Auto-load status (step 1 only, while the tutorial dataset loads). */}
        {i === 0 && autoloadState === 'loading' && (
          <div data-testid="tour-autoload-loading" style={styles.loadNote}>
            <span style={styles.spinner} /> Loading tutorial data…
          </div>
        )}
        {i === 0 && autoloadState === 'error' && (
          <div data-testid="tour-autoload-error" style={styles.errNote}>
            Couldn’t auto-load the tutorial data — open it from{' '}
            <strong>Examples → Dummy Data</strong> and follow along.
          </div>
        )}
        {/* Say once, up front, that the app stays usable and how to leave —
            the tour is click-through, which is not the convention users expect
            from a dimmed overlay. */}
        {i === 0 && (
          <div data-testid="tour-usage-hint" style={styles.hint}>
            Keep working while the tour is open — clicking the app won’t close
            it. Exit with <strong>✕</strong>.
          </div>
        )}

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
  // pointerEvents:'none' on the root + every dim layer. The tour draws OVER the
  // app but never intercepts it: clicks reach the real controls, and no
  // full-screen hit-target exists that could dismiss the tour by accident.
  // Only `bubble` re-enables pointer events.
  overlay: { position: 'fixed', inset: 0, zIndex: 9000, pointerEvents: 'none' },
  dimAll: {
    position: 'absolute', inset: 0, background: 'rgba(17,17,27,0.72)',
    pointerEvents: 'none',
  },
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
    // The one interactive layer (the root is pointerEvents:'none').
    pointerEvents: 'auto',
    background: '#1e1e2e',
    border: '1px solid #313244',
    borderRadius: 10,
    padding: 14,
    boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
    color: '#cdd6f4',
    fontSize: 13,
    transition: 'left 140ms ease, top 140ms ease',
    // Never taller than the viewport — a long step body scrolls INSIDE the
    // bubble so the Back/Next/Done footer is always on-screen and clickable.
    maxHeight: 'calc(100vh - 16px)',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
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
    // Sticky at the bottom of a scrolling bubble so Back/Done is always visible
    // even when a long step body overflows.
    position: 'sticky', bottom: -14, background: '#1e1e2e',
    paddingTop: 8, marginBottom: -14, paddingBottom: 14,
  },
  navBtn: {
    background: 'transparent', border: '1px solid #313244', color: '#cdd6f4',
    borderRadius: 6, padding: '5px 12px', cursor: 'pointer', fontSize: 12,
  },
  primaryBtn: {
    background: ACCENT, border: 'none', color: '#11111b', fontWeight: 600,
    borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontSize: 12,
  },
  loadNote: {
    display: 'flex', alignItems: 'center', gap: 8, marginTop: 10,
    fontSize: 12, color: '#a6adc8',
  },
  errNote: {
    marginTop: 10, padding: '7px 9px', borderRadius: 6, fontSize: 11.5,
    lineHeight: 1.4, color: '#f9c0c9',
    background: 'rgba(243,139,168,0.10)', border: '1px solid rgba(243,139,168,0.35)',
  },
  spinner: {
    width: 12, height: 12, borderRadius: '50%', flexShrink: 0,
    border: '2px solid rgba(137,180,250,0.3)', borderTopColor: ACCENT,
    display: 'inline-block', animation: 'spyde-tour-spin 0.7s linear infinite',
  },
  hint: {
    marginTop: 10, paddingTop: 8, borderTop: '1px solid #313244',
    fontSize: 11, lineHeight: 1.45, color: '#7f849c',
  },
  links: { marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 },
  linksLabel: {
    fontSize: 10.5, fontWeight: 700, letterSpacing: 0.6, color: '#6c7086',
    textTransform: 'uppercase', marginBottom: 2,
  },
  linkBtn: {
    display: 'block', width: '100%', textAlign: 'left', cursor: 'pointer',
    background: 'rgba(137,180,250,0.08)', border: '1px solid #313244',
    borderRadius: 6, padding: '6px 8px', color: '#cdd6f4',
  },
  linkLabel: { display: 'block', fontSize: 12, color: ACCENT, fontWeight: 500 },
  linkNote: { display: 'block', fontSize: 11, color: '#7f849c', marginTop: 2, lineHeight: 1.35 },
}
