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
 * PRESENTER VIEW (S to toggle, or the header button): swaps THIS screen between
 * the clean audience slide and a presenter DASHBOARD — the current slide (live,
 * scaled), the NEXT slide (a smaller dimmed preview), the current slide's SPEAKER
 * NOTES (the big readable panel), and an elapsed-time TIMER (start/pause/reset) +
 * the slide position. SpyDE is a single Electron window, so this is a same-screen
 * toggle, not a true dual-monitor audience/presenter split (a real second-window
 * popout is a future extension). Advancing slides (arrows) works in BOTH views and
 * keeps them in sync — the presenter dashboard reads the SAME `index`. Notes are
 * speaker-private: they render ONLY in the presenter view, never on the audience
 * slide or in the exported deck.
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
import { SlideOverview } from './SlideOverview'
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
/* ── presentation polish: TITLE / SECTION slides ──────────────────────────────
   A title slide centers a large title block — first heading huge, the rest a
   muted subtitle. Scoped to .present-title-md so a content slide is unchanged. */
.present-title-md { text-align: center; }
.present-title-md h1 { font-size: 4.2rem; line-height: 1.08; margin: 0 0 0.6rem;
  font-weight: 800; letter-spacing: -0.01em; }
.present-title-md h2 { font-size: 2.2rem; margin: 0.2rem 0; font-weight: 600; color: #cdd6f4; }
.present-title-md h3 { font-size: 1.6rem; color: #a6adc8; font-weight: 500; }
.present-title-md p { font-size: 1.6rem; color: #a6adc8; margin: 0.3rem 0; }
.present-title-md h1::after { content: ""; display: block; width: 4rem; height: 3px;
  margin: 1.2rem auto 0; background: #89b4fa; border-radius: 2px; }
/* ── presenter-view speaker notes ─────────────────────────────────────────────
   The big readable notes panel in the presenter dashboard — larger, roomy line
   height, scoped so it doesn't affect the audience slide markdown. */
.present-notes-md { font-size: 1.25rem; line-height: 1.7; color: #e8e8f0; }
.present-notes-md > *:first-child { margin-top: 0; }
.present-notes-md h1, .present-notes-md h2, .present-notes-md h3 {
  color: #cdd6f4; margin: 0.8rem 0 0.4rem; border-bottom: none; padding-bottom: 0; }
.present-notes-md h1 { font-size: 1.7rem; }
.present-notes-md h2 { font-size: 1.45rem; }
.present-notes-md h3 { font-size: 1.25rem; }
.present-notes-md p { margin: 0 0 0.7rem; }
.present-notes-md ul, .present-notes-md ol { margin: 0.5rem 0; padding-left: 1.5rem; }
.present-notes-md li { margin: 0.25rem 0; }
.present-notes-md strong { color: #ffffff; }
.present-notes-md code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.9em; background: #22222f; padding: 0.1em 0.35em; border-radius: 4px; color: #f5c2e7; }
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

// The per-slide presentation attributes — read off the slide's FIRST cell (the
// renderer mirror of `model.slide_meta`). kind '' (content) / 'title' (a
// big-centered title slide); style '' (default) / 'plain' / 'accent'.
export type SlideKind = '' | 'title'
export type SlideStyle = '' | 'plain' | 'accent'
export function slideMeta(cells: ReportCell[]): { kind: SlideKind; style: SlideStyle } {
  const first = cells[0]
  const k = (first?.slide_kind ?? '').trim().toLowerCase()
  const s = (first?.slide_style ?? '').trim().toLowerCase()
  return {
    kind: k === 'title' ? 'title' : '',
    style: s === 'plain' || s === 'accent' ? (s as SlideStyle) : '',
  }
}

/** A slide's SPEAKER NOTES — read off its FIRST cell (the renderer mirror of
 *  `model.slide_notes`). '' when the slide has no notes. */
export function slideNotes(cells: ReportCell[]): string {
  return (cells[0]?.notes ?? '').toString()
}

/** Group the mirrored report cells into slides by `slide_break` — re-exported so
 *  the Slide Overview grid uses the SAME grouping as Present mode. */
export { groupSlides }

/** mm:ss for an elapsed-seconds count (the presenter timer). */
function fmtElapsed(sec: number): string {
  const s = Math.max(0, Math.floor(sec))
  const mm = Math.floor(s / 60)
  const ss = s % 60
  return `${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
}

export function PresentMode({ initialSlide, onSlideChange, onExit, onLaunchLive }: Props) {
  const { state, iframeRefs, replayState, sendAction } = useSpyDE()
  const report = state.report && state.report.open ? state.report : null
  const cells = report?.cells ?? []

  const slides = useMemo(() => groupSlides(cells), [cells])
  const count = slides.length

  // Clamp the incoming index into range (a deck edited down mid-excursion could
  // leave it past the end).
  const [index, setIndex] = React.useState(() =>
    Math.max(0, Math.min(initialSlide, Math.max(0, count - 1))))

  // Presenter view: swap THIS screen between the clean audience slide and the
  // presenter dashboard (current + next + notes + timer). Same window, toggled
  // with `S` or the header button. Advancing keeps both in sync (shared index).
  const [presenter, setPresenter] = React.useState(false)

  // Slide overview grid: a thumbnail grid of ALL slides (the presenter's "jump
  // around" + drag-reorder tool). Toggled with `O` or the header grid button.
  // While it's open, present-mode navigation keys are suppressed (the overview
  // owns the keyboard) so arrows/Esc don't leak through to the deck behind it.
  const [overview, setOverview] = React.useState(false)

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
      // While the overview grid is open it OWNS the keyboard (it has its own
      // capture-phase Esc → close). Only `O` (toggle back off) reaches here.
      if (overview) {
        if (k === 'o' || k === 'O') { e.preventDefault(); setOverview(false) }
        return
      }
      if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'Spacebar') {
        e.preventDefault(); go(index + 1)
      } else if (k === 'ArrowLeft' || k === 'PageUp') {
        e.preventDefault(); go(index - 1)
      } else if (k === 'Home') { e.preventDefault(); go(0) }
      else if (k === 'End') { e.preventDefault(); go(count - 1) }
      else if (k === 's' || k === 'S') { e.preventDefault(); setPresenter(p => !p) }
      else if (k === 'o' || k === 'O' || k === 'g' || k === 'G') {
        e.preventDefault(); setOverview(true)
      }
      else if (k === 'Escape') { e.preventDefault(); onExit() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [index, count, go, onExit, overview])

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
    <div style={styles.overlay} data-testid="present-mode" data-presenter={presenter ? '1' : '0'}>
      {/* Every slide is RENDERED (so figure iframes stay mounted across
          navigation and never tear down); only the active one is displayed.
          In presenter mode the whole audience stack is hidden (kept MOUNTED so
          the live iframes survive) and the presenter dashboard renders on top. */}
      <div style={presenter ? styles.audienceHidden : undefined} aria-hidden={presenter}>
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
      </div>

      {presenter && (
        <PresenterView
          slides={slides}
          index={index}
          count={count}
          onGo={go}
        />
      )}

      {/* Slide overview grid: thumbnails of every slide. Click a thumbnail to
          jump (and close); drag one onto another to reorder the WHOLE slide via
          `report_move_slide`. Thumbnails are STATIC (baked PNGs), not live
          iframes — the live embeds stay in the audience stack behind this. */}
      {overview && (
        <SlideOverview
          slides={slides}
          index={index}
          onJump={(i) => { go(i); setOverview(false) }}
          onClose={() => setOverview(false)}
          onMoveSlide={(from, to) => {
            sendAction('report_move_slide', { from, to })
            // Keep the CURRENT slide highlighted at its new position: if the
            // moved slide is the one we're on, follow it; otherwise adjust for
            // the block shift so the same slide stays "current".
            setIndex(cur => {
              if (cur === from) return to
              // A slide moved out of `from` and into `to`: recompute where our
              // current index lands after the splice.
              let n = cur
              if (from < cur) n -= 1          // our slide shifted down by the removal
              if (to <= n) n += 1             // …and up by the insertion at/<= it
              return Math.max(0, Math.min(n, Math.max(0, count - 1)))
            })
          }}
        />
      )}

      {/* Top-right controls: overview grid + presenter-view toggle + exit. */}
      <div style={styles.topBar}>
        <button
          data-testid="present-overview-toggle"
          data-active={overview ? '1' : '0'}
          style={{ ...styles.iconBtn, ...(overview ? styles.iconBtnActive : {}) }}
          title="Slide overview: jump around + reorder slides (O)"
          onClick={() => setOverview(o => !o)}
        >▦</button>
        <button
          data-testid="present-presenter-toggle"
          data-active={presenter ? '1' : '0'}
          style={{ ...styles.iconBtn, ...(presenter ? styles.iconBtnActive : {}) }}
          title={presenter
            ? 'Presenter view ON — show the clean audience slide (S)'
            : 'Presenter view: current + next + notes + timer (S)'}
          onClick={() => setPresenter(p => !p)}
        >🗣</button>
        <button
          data-testid="present-exit"
          style={styles.iconBtn}
          title="Exit presentation (Esc)"
          onClick={onExit}
        >✕</button>
      </div>

      {/* Bottom bar: prev / counter / next. Hidden in presenter mode (the
          dashboard has its own nav + counter). */}
      {!presenter && (
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
      )}
    </div>
  )
}

// ── Presenter view (single-window dashboard) ───────────────────────────────────

/** The presenter dashboard: a header (timer + position + nav), the CURRENT slide
 *  preview (largest, top-left), the NEXT slide preview (smaller, dimmed, top-right),
 *  and the current slide's SPEAKER NOTES filling the bottom (the big readable
 *  panel). Slide previews are STATIC (markdown + the figure's baked PNG / photo) so
 *  we never duplicate the live audience iframes; the live embeds stay in the hidden
 *  audience stack. Advancing (arrows / the header nav) drives the SAME `index`, so
 *  audience + presenter stay in sync. */
function PresenterView({ slides, index, count, onGo }: {
  slides: ReportCell[][]
  index: number
  count: number
  onGo: (n: number) => void
}) {
  const current = slides[index] ?? []
  const next = index + 1 < count ? slides[index + 1] : null
  const notes = slideNotes(current)
  const notesHtml = React.useMemo(
    () => (notes.trim() ? renderMarkdown(notes) : ''), [notes])

  // Elapsed timer: running/paused + start epoch, mm:ss. Starts running the
  // moment the presenter view first opens.
  const [running, setRunning] = React.useState(true)
  const [elapsed, setElapsed] = React.useState(0)
  const startedAt = React.useRef<number>(Date.now())
  const baseElapsed = React.useRef(0)   // accumulated seconds across pauses
  React.useEffect(() => {
    if (!running) return
    startedAt.current = Date.now()
    const id = setInterval(() => {
      setElapsed(baseElapsed.current + (Date.now() - startedAt.current) / 1000)
    }, 250)
    return () => clearInterval(id)
  }, [running])
  const togglePause = () => {
    setRunning(r => {
      if (r) { baseElapsed.current = baseElapsed.current + (Date.now() - startedAt.current) / 1000 }
      else { startedAt.current = Date.now() }
      return !r
    })
  }
  const resetTimer = () => {
    baseElapsed.current = 0
    startedAt.current = Date.now()
    setElapsed(0)
  }

  return (
    <div style={styles.presenter} data-testid="presenter-view">
      {/* Header: timer + slide position + prev/next. */}
      <div style={styles.presHeader}>
        <div style={styles.timerBox}>
          <span data-testid="presenter-timer" style={styles.timerText}>
            {fmtElapsed(elapsed)}
          </span>
          <button
            data-testid="presenter-timer-pause"
            style={styles.timerBtn}
            title={running ? 'Pause timer' : 'Resume timer'}
            onClick={togglePause}
          >{running ? '⏸' : '▶'}</button>
          <button
            data-testid="presenter-timer-reset"
            style={styles.timerBtn}
            title="Reset timer"
            onClick={resetTimer}
          >⟲</button>
        </div>
        <div style={styles.presTitle}>Presenter view</div>
        <div style={styles.presNav}>
          <button
            data-testid="presenter-prev"
            style={{ ...styles.presNavBtn, ...(index === 0 ? styles.navBtnDisabled : {}) }}
            disabled={index === 0}
            title="Previous (←)"
            onClick={() => onGo(index - 1)}
          >‹</button>
          <span data-testid="presenter-counter" style={styles.presCounter}>
            {index + 1} / {count}
          </span>
          <button
            data-testid="presenter-next"
            style={{ ...styles.presNavBtn, ...(index >= count - 1 ? styles.navBtnDisabled : {}) }}
            disabled={index >= count - 1}
            title="Next (→ / Space)"
            onClick={() => onGo(index + 1)}
          >›</button>
        </div>
      </div>

      {/* Top row: current slide (large) + next slide (smaller, dimmed). */}
      <div style={styles.presPreviews}>
        <div style={styles.presCurrentWrap}>
          <div style={styles.presPreviewLabel}>Current</div>
          <div style={styles.presCurrentBox} data-testid="presenter-current">
            <SlidePreview cells={current} />
          </div>
        </div>
        <div style={styles.presNextWrap}>
          <div style={styles.presPreviewLabel}>Next</div>
          <div style={styles.presNextBox} data-testid="presenter-next-preview">
            {next
              ? <SlidePreview cells={next} dimmed />
              : <div style={styles.presEndCard}>End of deck</div>}
          </div>
        </div>
      </div>

      {/* Bottom: the current slide's speaker notes (the big readable panel). */}
      <div style={styles.presNotes} data-testid="presenter-notes">
        <div style={styles.presNotesLabel}>Speaker notes</div>
        {notesHtml
          ? <div className="spyde-md present-notes-md"
              data-testid="presenter-notes-body"
              dangerouslySetInnerHTML={{ __html: notesHtml }} />
          : <div style={styles.presNotesEmpty} data-testid="presenter-notes-empty">
              No notes for this slide.
            </div>}
      </div>
    </div>
  )
}

/** A STATIC, scaled-down preview of a slide for the presenter dashboard: markdown
 *  cells render their HTML; figure cells show their baked PNG (offline snapshot);
 *  image cells show their photo. It deliberately does NOT mount the live figure
 *  iframe (that lives in the hidden audience stack) so the presenter panel is cheap
 *  and never duplicates/steals an embed. `dimmed` softens the NEXT preview. */
export function SlidePreview({ cells, dimmed }: { cells: ReportCell[]; dimmed?: boolean }) {
  const meta = React.useMemo(() => slideMeta(cells), [cells])
  const isTitle = meta.kind === 'title'
  const styleBg =
    meta.style === 'plain' ? styles.slideBgPlain
      : meta.style === 'accent' ? styles.slideBgAccent : {}
  return (
    <div style={{ ...styles.previewStage, ...styleBg, ...(dimmed ? styles.previewDimmed : {}) }}>
      <div style={{ ...styles.previewInner, ...(isTitle ? styles.previewInnerTitle : {}) }}>
        {cells.map(cell => (
          <PreviewCell key={cell.id} cell={cell} titleSlide={isTitle} />
        ))}
      </div>
    </div>
  )
}

function PreviewCell({ cell, titleSlide }: { cell: ReportCell; titleSlide: boolean }) {
  if (cell.cell_type === 'markdown') {
    const html = renderMarkdown(cell.source ?? '')
    if (!(cell.source ?? '').trim()) return null
    const cls = 'spyde-md present-md' + (titleSlide ? ' present-title-md' : '')
    return <div className={cls} dangerouslySetInnerHTML={{ __html: html }} />
  }
  if (cell.cell_type === 'image') {
    if (!cell.image) return null
    return (
      <figure style={styles.previewFigure}>
        <img src={cell.image} alt={cell.caption ?? ''} style={styles.previewImg} />
        {(cell.caption ?? '').trim() &&
          <figcaption style={styles.previewCaption}>{cell.caption}</figcaption>}
      </figure>
    )
  }
  // figure cell — the baked PNG snapshot (a live iframe isn't mounted here).
  if (cell.placeholder) return null
  return (
    <figure style={styles.previewFigure}>
      {cell.png
        ? <img src={cell.png} alt={cell.caption ?? ''} style={styles.previewImg} />
        : <div style={styles.previewFigPending}>figure</div>}
      {(cell.caption ?? '').trim() &&
        <figcaption style={styles.previewCaption}>{cell.caption}</figcaption>}
    </figure>
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
  // Per-slide presentation polish: a title slide big-centers its markdown; a
  // style preset paints the background.
  const meta = React.useMemo(() => slideMeta(cells), [cells])
  const isTitle = meta.kind === 'title'
  const styleBg =
    meta.style === 'plain' ? styles.slideBgPlain
      : meta.style === 'accent' ? styles.slideBgAccent : {}

  const renderCell = (cell: ReportCell) => (
    <SlideCell
      key={cell.id}
      cell={cell}
      titleSlide={isTitle}
      reportFigures={reportFigures}
      iframeRefs={iframeRefs}
      replayState={replayState}
    />
  )

  return (
    <section
      data-testid="present-slide"
      data-active={active ? '1' : '0'}
      data-kind={isTitle ? 'title' : 'content'}
      data-style={meta.style || 'default'}
      style={{ ...styles.slide, ...styleBg,
        ...(isTitle ? styles.slideTitle : {}),
        ...(active ? styles.slideActive : {}) }}
    >
      <div style={{ ...styles.slideInner, ...(isTitle ? styles.slideInnerTitle : {}) }}>
        {cells.map(cell => (
          <React.Fragment key={cell.id}>{renderCell(cell)}</React.Fragment>
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
function SlideCell({ cell, titleSlide, reportFigures, iframeRefs, replayState }: {
  cell: ReportCell
  titleSlide: boolean
  reportFigures: ReturnType<typeof useSpyDE>['state']['reportFigures']
  iframeRefs: ReturnType<typeof useSpyDE>['iframeRefs']
  replayState: ReturnType<typeof useSpyDE>['replayState']
}) {
  if (cell.cell_type === 'markdown') return <SlideMarkdown cell={cell} titleSlide={titleSlide} />
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

function SlideMarkdown({ cell, titleSlide }: { cell: ReportCell; titleSlide: boolean }) {
  const html = useMemo(() => renderMarkdown(cell.source ?? ''), [cell.source])
  if (!(cell.source ?? '').trim()) return null
  // A title slide adds `present-title-md` (big-centered heading treatment).
  const cls = 'spyde-md present-md' + (titleSlide ? ' present-title-md' : '')
  return (
    <div
      data-testid={`present-md-${cell.id}`}
      data-title-slide={titleSlide ? '1' : '0'}
      className={cls}
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
  // A title slide: content vertically + horizontally centered, tighter column.
  slideTitle: { justifyContent: 'center', textAlign: 'center' },
  slideInnerTitle: { maxWidth: '48rem' },
  // Per-slide background presets.
  slideBgPlain: { background: '#0e0e16' },
  slideBgAccent: {
    background:
      'radial-gradient(ellipse at 50% 30%, rgba(137,180,250,0.18), transparent 70%), #14141f',
  },
  figure: { margin: '1rem 0', textAlign: 'center' },
  figBox: {
    // position:relative is LOAD-BEARING — SeamlessFigureFrame's frameHost is
    // `position:absolute; inset:0`, so it anchors to the nearest positioned
    // ancestor. Without this the iframe escaped its box and filled the whole
    // slide.
    position: 'relative',
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
  iconBtnActive: {
    background: '#89b4fa', color: '#11111b', borderColor: '#89b4fa',
  },
  // The whole audience slide stack is hidden (but kept MOUNTED) while the
  // presenter dashboard is up, so the live figure iframes never tear down.
  audienceHidden: { visibility: 'hidden', pointerEvents: 'none' },
  // ── presenter dashboard ──────────────────────────────────────────────────────
  presenter: {
    position: 'absolute', inset: 0, zIndex: 20,
    background: '#0e0e16', color: '#e8e8f0',
    display: 'flex', flexDirection: 'column',
    padding: '56px 3vw 2.5vh', gap: '1.6vh',
    fontSize: 16,
  },
  presHeader: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    gap: 16, paddingBottom: 8, borderBottom: '1px solid #313244',
  },
  timerBox: {
    display: 'flex', alignItems: 'center', gap: 8,
    minWidth: 220,
  },
  timerText: {
    fontSize: 34, fontWeight: 700, letterSpacing: 1,
    color: '#89b4fa', fontVariantNumeric: 'tabular-nums',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
  },
  timerBtn: {
    background: 'rgba(137,180,250,0.12)', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 6,
    width: 30, height: 30, fontSize: 14, cursor: 'pointer',
  },
  presTitle: {
    fontSize: 14, color: '#7f849c', fontWeight: 600, letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  presNav: { display: 'flex', alignItems: 'center', gap: 12, minWidth: 220, justifyContent: 'flex-end' },
  presNavBtn: {
    background: 'rgba(30,30,46,0.9)', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 8,
    width: 40, height: 40, fontSize: 24, lineHeight: 1, cursor: 'pointer',
  },
  presCounter: { fontSize: 18, color: '#cdd6f4', minWidth: 66, textAlign: 'center', fontWeight: 600 },
  presPreviews: {
    display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '2vw',
    flex: '0 0 48%', minHeight: 0,
  },
  presCurrentWrap: { display: 'flex', flexDirection: 'column', minHeight: 0 },
  presNextWrap: { display: 'flex', flexDirection: 'column', minHeight: 0 },
  presPreviewLabel: {
    fontSize: 12, color: '#7f849c', fontWeight: 600, letterSpacing: 0.5,
    textTransform: 'uppercase', marginBottom: 5,
  },
  presCurrentBox: {
    flex: 1, minHeight: 0, borderRadius: 10, overflow: 'hidden',
    border: '2px solid #45475a', background: '#14141f',
  },
  presNextBox: {
    flex: 1, minHeight: 0, borderRadius: 10, overflow: 'hidden',
    border: '1px solid #313244', background: '#14141f',
  },
  presEndCard: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    height: '100%', color: '#585b70', fontSize: 15, fontStyle: 'italic',
  },
  presNotes: {
    flex: 1, minHeight: 0, borderRadius: 10, padding: '14px 20px',
    background: '#181825', border: '1px solid #313244',
    overflowY: 'auto',
  },
  presNotesLabel: {
    fontSize: 12, color: '#7f849c', fontWeight: 600, letterSpacing: 0.5,
    textTransform: 'uppercase', marginBottom: 8,
  },
  presNotesEmpty: { color: '#585b70', fontSize: 16, fontStyle: 'italic' },
  // ── slide preview (static, scaled) ───────────────────────────────────────────
  previewStage: {
    width: '100%', height: '100%', overflow: 'hidden',
    display: 'flex', flexDirection: 'column', justifyContent: 'center',
    padding: '3% 5%',
  },
  previewDimmed: { opacity: 0.65 },
  previewInner: { width: '100%', maxWidth: '100%', margin: '0 auto' },
  previewInnerTitle: { textAlign: 'center' },
  previewFigure: { margin: '0.4rem 0', textAlign: 'center' },
  previewImg: {
    display: 'block', margin: '0 auto',
    maxWidth: '100%', maxHeight: '30vh', height: 'auto', borderRadius: 6,
  },
  previewFigPending: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    height: 80, color: '#585b70', fontSize: 12,
    border: '1px dashed #45475a', borderRadius: 6,
  },
  previewCaption: {
    marginTop: 3, fontSize: '0.7rem', color: '#a6adc8', fontStyle: 'italic',
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
