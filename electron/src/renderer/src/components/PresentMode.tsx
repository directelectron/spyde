/**
 * PresentMode.tsx — present a Report as SLIDES (Phase 6, Present mode).
 *
 * A full-screen overlay (z above everything, like the update dialog) that renders
 * ONE slide at a time from the report doc's slides — cells grouped by the per-cell
 * `slide_break` flag (the SAME grouping `ReportDoc.slides()` does on the backend).
 *
 * THE DESIGN'S WHOLE POINT: three SEPARATE surfaces.
 *   • Static slides — freely navigable back/forward. Slides are RENDERED CONTENT,
 *     not live app state, so navigation is always safe (nothing to rewind).
 *   • Interactive EMBEDS baked into slides — a figure cell mounts the SAME live
 *     SeamlessFigureFrame the Report sidebar uses (vectors explorer, anyplotlib
 *     widgets), so it stays interactive INSIDE the slide on frozen data. To keep
 *     it from tearing down destructively on navigation, ALL figure frames stay
 *     MOUNTED (each slide is rendered; only the active one is visible) — the
 *     iframe never unmounts as you move back/forth.
 *   • "Go live" excursions — a slide carrying `live_action` shows a "Launch live ▶"
 *     button that exits Present mode and fires the live action (a tutorial load +
 *     optional guide tour) in the REAL app, then a floating "⤺ Back" pill (owned by
 *     App) RE-ENTERS Present mode at the SAME slide index. The live side-trip does
 *     not BECOME the slide (live compute state doesn't rewind).
 *
 * Controls: → / Space / PageDown next, ← / PageUp prev, Home/End first/last, ESC
 * exit. A presentation clicker (remote) sends arrow / PageUp/PageDown keys, so
 * those Just Work. A slide counter (n / N) shows position.
 *
 * Graceful degradation (Phase 1/2 may not be merged): `onLaunchLive` is wired to
 * `sendAction('tutorial_load', …)` — if that action isn't available the backend
 * simply no-ops and the excursion is just "exit Present mode"; we never hard-
 * depend on the tutorial/guide phases landing.
 */
import React, { useEffect, useMemo } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { renderMarkdown } from '../kernel/markdown'
import { SeamlessFigureFrame } from './ReportFigureCell'
import type { ReportCell } from '../kernel/protocol'

// Present-mode markdown sizing: the `.spyde-md` base stylesheet (injected by
// ReportCell) is em-relative off `--spyde-md-fs` (13px in the sidebar). On the
// full-screen stage we want big readable type, so a scoped `present-md` override
// bumps the base font and heading rhythm. Injected once (the ConsoleBar idiom).
// Also (re)inject the base `.spyde-md` rules in case ReportCell wasn't imported
// yet (Present mode can be opened before the sidebar mounted a markdown cell).
if (typeof document !== 'undefined' && !document.getElementById('spyde-present-md-css')) {
  const el = document.createElement('style')
  el.id = 'spyde-present-md-css'
  el.textContent = `
.present-md { font-size: 1.15rem; line-height: 1.6; color: #e8e8f0; word-break: break-word; }
.present-md > *:first-child { margin-top: 0; }
.present-md h1 { font-size: 2.4rem; line-height: 1.15; margin: 0 0 1.2rem; font-weight: 700;
  border-bottom: none; padding-bottom: 0; }
.present-md h2 { font-size: 1.8rem; margin: 1.4rem 0 0.7rem; border-bottom: none; padding-bottom: 0; }
.present-md h3 { font-size: 1.35rem; margin: 1.1rem 0 0.5rem; }
.present-md p { margin: 0 0 0.9rem; }
.present-md ul, .present-md ol { margin: 0.6rem 0; padding-left: 1.6rem; }
.present-md li { margin: 0.3rem 0; }
.present-md code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.9em; background: #22222f; padding: 0.1em 0.35em; border-radius: 4px; color: #f5c2e7; }
.present-md pre { background: #1c1c28; padding: 1rem; border-radius: 8px; overflow-x: auto; }
.present-md pre code { background: none; padding: 0; color: #cdd6f4; }
.present-md blockquote { border-left: 4px solid #45475a; margin: 0.8rem 0; padding: 0.2rem 1rem;
  color: #a6adc8; }
.present-md a { color: #89b4fa; }
.present-md strong { color: #ffffff; }
.present-md .katex-display { display: block; margin: 1rem 0; text-align: center;
  overflow-x: auto; overflow-y: hidden; }
`
  document.head.appendChild(el)
}

export interface LiveAction {
  tutorial?: string
  guide?: string
}

interface Props {
  /** The slide index to open on (persisted across a go-live excursion by App). */
  initialSlide: number
  /** Called on every slide change so App can persist the index for re-entry. */
  onSlideChange: (index: number) => void
  /** ESC / the exit button — close Present mode entirely. */
  onExit: () => void
  /** Launch a slide's go-live excursion: App exits Present mode, fires the live
   *  action, and shows the "Back to presentation" pill. */
  onLaunchLive: (action: LiveAction) => void
}

/** Group the mirrored report cells into slides by `slide_break` — the renderer
 *  mirror of `ReportDoc.slides()`. A break STARTS a new slide; the first cell
 *  always begins slide 0. */
function groupSlides(cells: ReportCell[]): ReportCell[][] {
  const groups: ReportCell[][] = []
  for (const c of cells) {
    if (c.slide_break && groups.length) groups.push([c])
    else if (!groups.length) groups.push([c])
    else groups[groups.length - 1].push(c)
  }
  return groups
}

export function PresentMode({ initialSlide, onSlideChange, onExit, onLaunchLive }: Props) {
  const { state, iframeRefs, replayState } = useSpyDE()
  const report = state.report && state.report.open ? state.report : null
  const cells = report?.cells ?? []

  const slides = useMemo(() => groupSlides(cells), [cells])
  const count = slides.length

  // Clamp the incoming index into range (a deck edited down mid-excursion could
  // leave it past the end).
  const [index, setIndex] = React.useState(() =>
    Math.max(0, Math.min(initialSlide, Math.max(0, count - 1))))

  // Report every change up so App can persist it for re-entry.
  useEffect(() => { onSlideChange(index) }, [index, onSlideChange])
  // Keep the index valid as the deck changes (cells added/removed while open).
  useEffect(() => {
    setIndex(i => Math.max(0, Math.min(i, Math.max(0, count - 1))))
  }, [count])

  const go = React.useCallback((n: number) => {
    setIndex(i => Math.max(0, Math.min(n, Math.max(0, count - 1))))
  }, [count])

  // Keyboard: clicker-friendly. Arrows/Space/PageDown advance, PageUp/← back,
  // Home/End jump, ESC exits. Attached to window so it works regardless of focus
  // (a figure iframe might otherwise steal it — but keydown on window still fires
  // for the top document's controls).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key
      if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'Spacebar') {
        e.preventDefault(); go(index + 1)
      } else if (k === 'ArrowLeft' || k === 'PageUp') {
        e.preventDefault(); go(index - 1)
      } else if (k === 'Home') { e.preventDefault(); go(0) }
      else if (k === 'End') { e.preventDefault(); go(count - 1) }
      else if (k === 'Escape') { e.preventDefault(); onExit() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [index, count, go, onExit])

  if (report == null || count === 0) {
    // Nothing to present — surface a tiny message instead of a blank overlay.
    return (
      <div style={styles.overlay} data-testid="present-mode">
        <div style={styles.emptyMsg} data-testid="present-empty">
          This report has no slides to present.
          <button style={styles.exitBtn} onClick={onExit} data-testid="present-exit">Exit</button>
        </div>
      </div>
    )
  }

  return (
    <div style={styles.overlay} data-testid="present-mode">
      {/* Every slide is RENDERED (so figure iframes stay mounted across
          navigation and never tear down); only the active one is displayed. */}
      {slides.map((group, si) => (
        <Slide
          key={si}
          cells={group}
          active={si === index}
          reportFigures={state.reportFigures}
          iframeRefs={iframeRefs}
          replayState={replayState}
          onLaunchLive={onLaunchLive}
        />
      ))}

      {/* Top-right controls: exit. */}
      <div style={styles.topBar}>
        <button
          data-testid="present-exit"
          style={styles.iconBtn}
          title="Exit presentation (Esc)"
          onClick={onExit}
        >✕</button>
      </div>

      {/* Bottom bar: prev / counter / next. */}
      <div style={styles.bottomBar}>
        <button
          data-testid="present-prev"
          style={{ ...styles.navBtn, ...(index === 0 ? styles.navBtnDisabled : {}) }}
          title="Previous (←)"
          disabled={index === 0}
          onClick={() => go(index - 1)}
        >‹</button>
        <span data-testid="present-counter" style={styles.counter}>
          {index + 1} / {count}
        </span>
        <button
          data-testid="present-next"
          style={{ ...styles.navBtn, ...(index >= count - 1 ? styles.navBtnDisabled : {}) }}
          title="Next (→ / Space)"
          disabled={index >= count - 1}
          onClick={() => go(index + 1)}
        >›</button>
      </div>
    </div>
  )
}

// One slide: its cells' rendered content, plus (if any cell carries a
// live_action) a "Launch live ▶" button. Kept always-mounted; visibility toggled
// so figure embeds survive navigation.
function Slide({ cells, active, reportFigures, iframeRefs, replayState, onLaunchLive }: {
  cells: ReportCell[]
  active: boolean
  reportFigures: ReturnType<typeof useSpyDE>['state']['reportFigures']
  iframeRefs: ReturnType<typeof useSpyDE>['iframeRefs']
  replayState: ReturnType<typeof useSpyDE>['replayState']
  onLaunchLive: (action: LiveAction) => void
}) {
  // The go-live handle for this slide: the first cell that carries one.
  const live = cells.find(c => c.live_action)?.live_action as LiveAction | undefined

  return (
    <section
      data-testid="present-slide"
      data-active={active ? '1' : '0'}
      style={{ ...styles.slide, ...(active ? styles.slideActive : {}) }}
    >
      <div style={styles.slideInner}>
        {cells.map(cell => (
          <SlideCell
            key={cell.id}
            cell={cell}
            reportFigures={reportFigures}
            iframeRefs={iframeRefs}
            replayState={replayState}
          />
        ))}
        {live && (
          <div style={styles.liveRow}>
            <button
              data-testid="present-launch-live"
              style={styles.liveBtn}
              title="Open this dataset in the app and demo live"
              onClick={() => onLaunchLive(live)}
            >Launch live ▶</button>
          </div>
        )}
      </div>
    </section>
  )
}

// One cell inside a slide: markdown → sanitized HTML (reusing the report's own
// render pipeline); figure → the live SeamlessFigureFrame (interactive embed),
// its baked PNG when offline, or a skipped placeholder. Dispatches to two
// sub-components so hooks are never called conditionally (cell_type is stable
// per cell id, but keep the split for React-rules correctness).
function SlideCell({ cell, reportFigures, iframeRefs, replayState }: {
  cell: ReportCell
  reportFigures: ReturnType<typeof useSpyDE>['state']['reportFigures']
  iframeRefs: ReturnType<typeof useSpyDE>['iframeRefs']
  replayState: ReturnType<typeof useSpyDE>['replayState']
}) {
  if (cell.cell_type === 'markdown') return <SlideMarkdown cell={cell} />
  if (cell.cell_type === 'image') return <SlideImage cell={cell} />
  return (
    <SlideFigure
      cell={cell}
      reportFigures={reportFigures}
      iframeRefs={iframeRefs}
      replayState={replayState}
    />
  )
}

// A photo on a slide — large + centered, using the same data URL the sidebar
// renders. Sized to fit the slide (max-height caps it so a tall image doesn't
// push the caption off-screen).
function SlideImage({ cell }: { cell: ReportCell }) {
  const caption = (cell.caption ?? '').trim()
  if (!cell.image) return null
  return (
    <figure data-testid={`present-img-${cell.id}`} style={styles.figure}>
      <img src={cell.image} alt={caption} style={styles.slideImg} />
      {caption && <figcaption style={styles.figCaption}>{caption}</figcaption>}
    </figure>
  )
}

function SlideMarkdown({ cell }: { cell: ReportCell }) {
  const html = useMemo(() => renderMarkdown(cell.source ?? ''), [cell.source])
  if (!(cell.source ?? '').trim()) return null
  return (
    <div
      data-testid={`present-md-${cell.id}`}
      className="spyde-md present-md"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

function SlideFigure({ cell, reportFigures, iframeRefs, replayState }: {
  cell: ReportCell
  reportFigures: ReturnType<typeof useSpyDE>['state']['reportFigures']
  iframeRefs: ReturnType<typeof useSpyDE>['iframeRefs']
  replayState: ReturnType<typeof useSpyDE>['replayState']
}) {
  if (cell.placeholder) return null
  const fig = reportFigures.get(cell.id)
  const caption = (cell.caption ?? '').trim()
  return (
    <figure data-testid={`present-fig-${cell.id}`} style={styles.figure}>
      <div style={styles.figBox}>
        {fig ? (
          <SeamlessFigureFrame
            figId={fig.figId}
            filePath={fig.filePath}
            title={fig.title}
            iframeRefs={iframeRefs}
            replayState={replayState}
          />
        ) : cell.png ? (
          <img src={cell.png} alt={caption} style={styles.figImg} />
        ) : (
          <div style={styles.figPending}>rendering…</div>
        )}
      </div>
      {caption && <figcaption style={styles.figCaption}>{caption}</figcaption>}
    </figure>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9500,
    background: '#14141f', color: '#e8e8f0',
    fontSize: 22, lineHeight: 1.6,
  },
  slide: {
    position: 'absolute', inset: 0, display: 'none',
    flexDirection: 'column', justifyContent: 'center',
    padding: '5vh 8vw', overflowY: 'auto',
  },
  slideActive: { display: 'flex' },
  slideInner: { maxWidth: '60rem', margin: '0 auto', width: '100%' },
  figure: { margin: '1rem 0', textAlign: 'center' },
  figBox: {
    width: '100%', height: '58vh',
    border: '1px solid #313244', borderRadius: 8, overflow: 'hidden',
    background: '#0e0e16',
  },
  figImg: { maxWidth: '100%', maxHeight: '100%', height: 'auto' },
  // A photo cell on a slide — large + centered, capped so it stays on-screen
  // with room for the caption.
  slideImg: {
    display: 'block', margin: '0 auto',
    maxWidth: '100%', maxHeight: '68vh', height: 'auto', borderRadius: 8,
  },
  figPending: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    height: '100%', color: '#585b70', fontSize: 14,
  },
  figCaption: {
    marginTop: '0.5rem', fontSize: '0.85rem', color: '#a6adc8', fontStyle: 'italic',
  },
  liveRow: { marginTop: '1.5rem', textAlign: 'center' },
  liveBtn: {
    background: '#89b4fa', color: '#11111b', border: 'none',
    borderRadius: 8, padding: '10px 22px', fontSize: 18, fontWeight: 700,
    cursor: 'pointer', boxShadow: '0 4px 16px rgba(137,180,250,0.35)',
  },
  topBar: {
    position: 'fixed', top: 16, right: 20, zIndex: 10,
    display: 'flex', gap: 8,
  },
  iconBtn: {
    background: 'rgba(30,30,46,0.8)', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 8,
    width: 36, height: 36, fontSize: 16, cursor: 'pointer',
  },
  bottomBar: {
    position: 'fixed', bottom: 18, left: '50%', transform: 'translateX(-50%)',
    zIndex: 10, display: 'flex', alignItems: 'center', gap: 14,
    background: 'rgba(20,20,31,0.75)', borderRadius: 22, padding: '6px 14px',
  },
  navBtn: {
    background: 'transparent', color: '#cdd6f4', border: 'none',
    fontSize: 28, lineHeight: 1, cursor: 'pointer', padding: '0 8px',
  },
  navBtnDisabled: { color: '#45475a', cursor: 'default' },
  counter: { fontSize: 14, color: '#a6adc8', minWidth: 60, textAlign: 'center' },
  emptyMsg: {
    position: 'absolute', top: '50%', left: '50%',
    transform: 'translate(-50%, -50%)', textAlign: 'center',
    fontSize: 18, color: '#a6adc8', display: 'flex', flexDirection: 'column', gap: 16,
  },
  exitBtn: {
    background: '#1e1e2e', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 8, padding: '8px 18px', fontSize: 14, cursor: 'pointer',
  },
}
