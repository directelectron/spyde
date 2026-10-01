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
import { DECK_THEME_DEFAULTS, type DeckTheme, type ReportCell } from '../kernel/protocol'

// Crisp inline SVG icons for the top-right present controls (replacing the emoji
// glyphs — the 🗣 speaker in particular read as "voice/audio", not "presenter").
// `currentColor` so they inherit the button's text colour (incl. the active state).
const svgProps = {
  width: 17, height: 17, viewBox: '0 0 24 24', fill: 'none',
  stroke: 'currentColor', strokeWidth: 2,
  strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const,
}
// A 3×3 grid — the slide-overview toggle.
const GridIcon = () => (
  <svg {...svgProps} aria-hidden>
    <rect x="3" y="3" width="7" height="7" rx="1" />
    <rect x="14" y="3" width="7" height="7" rx="1" />
    <rect x="3" y="14" width="7" height="7" rx="1" />
    <rect x="14" y="14" width="7" height="7" rx="1" />
  </svg>
)
// A presenter at a screen/board with a pointer — the presenter-view toggle.
const PresenterIcon = () => (
  <svg {...svgProps} aria-hidden>
    <rect x="3" y="3" width="18" height="12" rx="1.5" />
    <path d="M12 15v3" />
    <path d="M8 21h8" />
    <circle cx="8.5" cy="8.5" r="1.4" />
    <path d="M11.5 11l3-4 3 4" />
  </svg>
)
const CloseIcon = () => (
  <svg {...svgProps} aria-hidden>
    <path d="M6 6l12 12M18 6L6 18" />
  </svg>
)

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
/* Colours come from the deck THEME via custom properties set on the present
   overlay (themeVars). The literals are the fallback for anything rendering
   this markdown outside a themed deck. */
.present-md { font-size: 1.15rem; line-height: 1.6;
  color: var(--spyde-deck-text, #e8e8f0); word-break: break-word; }
.present-md > *:first-child { margin-top: 0; }
/* Headings carry an explicit colour: the BASE .spyde-md sheet hard-codes
   #cdd6f4 on h1-h6, which wins over the colour .present-md inherits, so
   without this a themed deck restyled its chrome and its body text while every
   heading stayed the old lavender.
   (No backticks in this block — it lives inside a JS template literal.) */
.present-md h1, .present-md h2, .present-md h3,
.present-md h4, .present-md h5, .present-md h6 { color: var(--spyde-deck-text, #cdd6f4); }
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
.present-md blockquote { border-left: 4px solid var(--spyde-deck-accent, #45475a);
  margin: 0.8rem 0; padding: 0.2rem 1rem;
  color: var(--spyde-deck-muted, #a6adc8); }
.present-md a { color: var(--spyde-deck-accent, #89b4fa); }
.present-md strong { color: var(--spyde-deck-text, #ffffff); }
.present-md .katex-display { display: block; margin: 1rem 0; text-align: center;
  overflow-x: auto; overflow-y: hidden; }
/* ── text-only slides FILL the stage ──────────────────────────────────────────
   A heading and a few bullets at prose size, centred in a 60rem column, is a
   small block adrift in a large dark rectangle on a projector. These tiers scale
   the type up so a sparse slide uses the room it has; a dense slide gets no
   class at all and keeps prose size, which is what stops it overflowing.

   vh-based with a clamp, so it tracks the projector rather than a fixed px
   guess, and cannot run away on a very tall or very short display. */
.present-fill .present-md { font-size: clamp(1.25rem, 2.55vh, 2.05rem); line-height: 1.55; }
.present-fill .present-md h1 { font-size: clamp(2.6rem, 5.6vh, 4.4rem); margin: 0 0 1.4rem; }
.present-fill .present-md h2 { font-size: clamp(2rem, 4.3vh, 3.4rem); margin: 0 0 1rem; }
.present-fill .present-md h3 { font-size: clamp(1.5rem, 3.2vh, 2.4rem); }
.present-fill .present-md li { margin: 0.55em 0; }
.present-fill .present-md ul, .present-fill .present-md ol { margin: 0.8em 0; }
.present-fill .present-md p { margin: 0 0 0.9em; }
/* The sparse tier goes further — this is the "title + four bullets" slide. */
.present-fill-lg .present-md { font-size: clamp(1.45rem, 3.05vh, 2.5rem); }
.present-fill-lg .present-md h2 { font-size: clamp(2.3rem, 5vh, 3.9rem); }
.present-fill-lg .present-md li { margin: 0.7em 0; }
/* ── presentation polish: TITLE / SECTION slides ──────────────────────────────
   A title slide centers a large title block — first heading huge, the rest a
   muted subtitle. Scoped to .present-title-md so a content slide is unchanged. */
.present-title-md { text-align: center; }
.present-title-md h1 { color: var(--spyde-deck-text, #cdd6f4);
  font-size: 4.2rem; line-height: 1.08; margin: 0 0 0.6rem;
  font-weight: 800; letter-spacing: -0.01em; }
.present-title-md h2 { font-size: 2.2rem; margin: 0.2rem 0; font-weight: 600;
  color: var(--spyde-deck-text, #cdd6f4); }
.present-title-md h3 { font-size: 1.6rem; color: var(--spyde-deck-muted, #a6adc8); font-weight: 500; }
.present-title-md p { font-size: 1.6rem; color: var(--spyde-deck-muted, #a6adc8); margin: 0.3rem 0; }
.present-title-md h1::after { content: ""; display: block; width: 4rem; height: 3px;
  margin: 1.2rem auto 0; background: var(--spyde-deck-accent, #89b4fa); border-radius: 2px; }
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

/**
 * Resolve a split cell's layout into the two booleans that actually drive the
 * grid — the renderer mirror of `model._SPLIT_LAYOUTS` and of the EDITOR's own
 * resolution in ReportSplitCell.
 *
 * All FOUR layouts, not the left/right pair this used to test with a bare
 * `!== 'text-right'`: that read `text-top` and `text-bottom` as text-left, so a
 * stacked split round-tripped through the document perfectly and then rendered
 * side-by-side on the slide. The editor offered a layout the deck could not
 * show.
 */
function splitLayoutOf(raw: string | undefined): {
  stacked: boolean; textFirst: boolean; layout: string
} {
  const layout = SPLIT_LAYOUTS.includes(raw ?? '') ? (raw as string) : 'text-left'
  return {
    layout,
    stacked: layout === 'text-top' || layout === 'text-bottom',
    textFirst: layout === 'text-left' || layout === 'text-top',
  }
}
const SPLIT_LAYOUTS: string[] = ['text-left', 'text-right', 'text-top', 'text-bottom']

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

/**
 * The theme as CSS CUSTOM PROPERTIES on the deck root.
 *
 * Variables rather than inline styles on each element: the markdown inside a
 * slide is `dangerouslySetInnerHTML`, so its h1/p/li/code are styled by the
 * injected `.present-md` stylesheet and can't take React inline styles at all.
 * A variable set on the overlay reaches them, and the same value drives the
 * chrome — one source of truth, no per-element plumbing.
 */
function themeVars(t: DeckTheme): React.CSSProperties {
  return {
    ['--spyde-deck-bg' as any]: t.bg,
    ['--spyde-deck-text' as any]: t.text,
    ['--spyde-deck-muted' as any]: t.muted,
    ['--spyde-deck-accent' as any]: t.accent,
    background: t.bg,
    color: t.text,
    ...(t.font ? { fontFamily: t.font } : {}),
  }
}

/**
 * Is this footer worth drawing at all?
 *
 * IDENTITY only — name, email, note, logo. Deliberately NOT the slide number:
 * `slide_numbers` defaults to true, so counting it would give every untouched
 * deck a footer bar it never asked for (and move its page count out of the
 * pager). The number rides along once there IS a footer; on a deck with no
 * identity set, the pager keeps showing it exactly as before.
 */
function footerHasContent(t: DeckTheme): boolean {
  return Boolean(t.footer_name || t.footer_email || t.footer_note || t.logo)
}

/**
 * The footer bar: logo, name / email / note, and the slide number.
 *
 * NOT drawn on a title slide — a title card carries its own attribution and a
 * repeated footer under it reads as clutter. That is the usual convention and
 * what was asked for.
 */
function SlideFooter({ theme, slideNumber, slideCount }: {
  theme: DeckTheme
  slideNumber: number
  slideCount: number
}) {
  const bits = [theme.footer_name, theme.footer_email, theme.footer_note]
    .map(s => s.trim()).filter(Boolean)
  return (
    <div style={styles.footer} data-testid="present-footer">
      <div style={styles.footerLeft}>
        {theme.logo && (
          <img
            src={theme.logo}
            alt=""
            data-testid="present-footer-logo"
            style={{ ...styles.footerLogo, height: theme.logo_height }}
          />
        )}
        {bits.length > 0 && (
          <span style={styles.footerText} data-testid="present-footer-text">
            {bits.map((b, i) => (
              <React.Fragment key={i}>
                {i > 0 && <span style={styles.footerSep}>·</span>}
                {b}
              </React.Fragment>
            ))}
          </span>
        )}
      </div>
      {theme.slide_numbers && (
        <span style={styles.footerNum} data-testid="present-footer-number">
          {slideNumber} / {slideCount}
        </span>
      )}
    </div>
  )
}

/**
 * The figure diagnostic, drawn ON THE SLIDE (press D while presenting).
 *
 * A report figure is mounted twice — sidebar cell and presented slide — and both
 * register under the SAME figId, so the presented copy draws from whatever
 * `replayState` re-sends to whichever element holds that registration. When a
 * panel comes up blank, the question is simply: did the PRESENTED iframe win
 * the registration, and does the pixel stash still hold a state per panel.
 *
 * Rendered rather than logged so answering it needs a keypress and a
 * screenshot, not a DevTools session mid-talk.
 */
function FigureDiag({ slideCells }: { slideCells: ReportCell[] }) {
  const rows = React.useMemo(() => {
    const fn = (window as unknown as Record<string, unknown>).__spydeFigureDump
    if (typeof fn !== 'function') return null
    try { return (fn as () => Record<string, unknown>[])() } catch { return null }
  }, [])
  // Panels expected on THIS slide, from the report doc — the number each
  // figure's stash has to be compared against.
  const expected = slideCells
    .filter(c => c.cell_type === 'figure' || c.cell_type === 'split')
    .map(c => `${c.id}: ${c.figure?.panels?.length ?? 0} panel(s)`)

  return (
    <div style={styles.diag} data-testid="present-figure-diag">
      <div style={styles.diagTitle}>Figure diagnostic — press D to hide</div>
      <div style={styles.diagLine}>this slide → {expected.join(' · ') || 'no figure cells'}</div>
      {rows == null ? (
        <div style={styles.diagLine}>__spydeFigureDump unavailable (old build?)</div>
      ) : rows.length === 0 ? (
        <div style={styles.diagLine}>no figures registered at all</div>
      ) : rows.map((r, i) => (
        <div key={i} style={styles.diagLine}>
          {String(r.figId)} · in={String(r.registeredIn)} · size={String(r.size)}
          {' · json='}{String(r.jsonKeys)}{' · binary='}{String(r.binaryKeys)}
          {r.binaryKeyNames ? ` (${String(r.binaryKeyNames)})` : ''}
        </div>
      ))}
    </div>
  )
}

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
  // The deck's look. The backend always ships a full theme; DECK_THEME_DEFAULTS
  // covers an older backend that ships none.
  const theme: DeckTheme = { ...DECK_THEME_DEFAULTS, ...(report?.theme ?? {}) }
  // Whether the PAGER shows "n / N". Off when the deck's footer already does,
  // so the projected screen never carries the same number twice.
  const showPagerCount = !(theme.footer_show && theme.slide_numbers
                           && footerHasContent(theme))

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
  // The on-screen figure diagnostic (press D while presenting).
  const [diag, setDiag] = React.useState(false)
  // On Windows the native title-bar overlay (min/max/close) sits at the very
  // top-right, ~38px tall. Push the presentation top controls BELOW it so they
  // aren't clipped/covered (macOS traffic lights are on the left, no conflict).
  const isMac = window.electron?.platform === 'darwin'
  const topBarStyle = isMac ? styles.topBar : { ...styles.topBar, top: 46, right: 12 }

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
      // D — the figure DIAGNOSTIC readout, on screen. Not in DevTools: asking
      // someone to open a console mid-presentation to debug a panel that
      // didn't draw is a bad trade, and console.table prints NOTHING at all
      // when the array is empty, which is indistinguishable from "the function
      // isn't there".
      else if (k === 'd' || k === 'D') {
        e.preventDefault(); setDiag(v => !v)
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
    <div
      style={{ ...styles.overlay, ...themeVars(theme) }}
      data-testid="present-mode"
      data-presenter={presenter ? '1' : '0'}
    >
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
            theme={theme}
            slideNumber={si + 1}
            slideCount={count}
          />
        ))}
      </div>

      {diag && <FigureDiag slideCells={slides[index] ?? []} />}

      {presenter && (
        <PresenterView
          slides={slides}
          index={index}
          count={count}
          onGo={go}
          onExitPresenter={() => setPresenter(false)}
          onExit={onExit}
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

      {/* Top-right controls: diagnostic + overview grid + presenter toggle + exit. */}
      <div style={topBarStyle}>
        <button
          data-testid="present-diag-toggle"
          data-active={diag ? '1' : '0'}
          style={{ ...styles.iconBtn, ...(diag ? styles.iconBtnActive : {}) }}
          title="Figure diagnostic — why a panel didn't draw (D)"
          onClick={() => setDiag(v => !v)}
        >{'⚠'}</button>
        <button
          data-testid="present-overview-toggle"
          data-active={overview ? '1' : '0'}
          style={{ ...styles.iconBtn, ...(overview ? styles.iconBtnActive : {}) }}
          title="Slide overview: jump around + reorder slides (O)"
          onClick={() => setOverview(o => !o)}
        ><GridIcon /></button>
        <button
          data-testid="present-presenter-toggle"
          data-active={presenter ? '1' : '0'}
          style={{ ...styles.iconBtn, ...(presenter ? styles.iconBtnActive : {}) }}
          title={presenter
            ? 'Presenter view ON — show the clean audience slide (S)'
            : 'Presenter view: current + next + notes + timer (S)'}
          onClick={() => setPresenter(p => !p)}
        ><PresenterIcon /></button>
        <button
          data-testid="present-exit"
          style={styles.iconBtn}
          title="Exit presentation (Esc)"
          onClick={onExit}
        ><CloseIcon /></button>
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
          {/* The pager's counter is SUPPRESSED when the deck's own footer is
              already showing slide numbers — the whole window is what gets
              projected, so both drawing "2 / 2" reads as a bug. The arrows
              stay: they are navigation, not information. */}
          {showPagerCount ? (
            <span data-testid="present-counter" style={styles.counter}>
              {index + 1} / {count}
            </span>
          ) : (
            <span data-testid="present-counter" style={styles.counterQuiet} aria-hidden />
          )}
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
function PresenterView({ slides, index, count, onGo, onExitPresenter, onExit }: {
  slides: ReportCell[][]
  index: number
  count: number
  onGo: (n: number) => void
  onExitPresenter: () => void   // back to the clean audience slide (leave presenter)
  onExit: () => void            // exit Present mode entirely
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
          {/* Explicit exits — the presenter dashboard covers the top-right
              controls, so these are the only on-screen way out. */}
          <button
            data-testid="presenter-exit-view"
            style={styles.presExitBtn}
            title="Leave presenter view — show the clean audience slide (S)"
            onClick={onExitPresenter}
          >Audience view</button>
          <button
            data-testid="presenter-exit-present"
            style={styles.presExitBtnStrong}
            title="Exit the presentation (Esc)"
            onClick={onExit}
          >✕ Exit</button>
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
  if (cell.cell_type === 'split') {
    // A split preview (presenter dashboard / overview thumbnail): text + the
    // baked figure/photo, side by side per split_layout. No live iframe here.
    const html = renderMarkdown(cell.source ?? '')
    const src = cell.image || cell.png
    const textPane = (
      <div key="t" className="spyde-md present-md" style={{ minWidth: 0 }}
           dangerouslySetInnerHTML={{ __html: html }} />
    )
    const figPane = (
      <div key="f" style={{ minWidth: 0, textAlign: 'center' }}>
        {src ? <img src={src} alt={cell.caption ?? ''} style={styles.previewImg} />
             : <div style={styles.previewFigPending}>figure</div>}
      </div>
    )
    const { stacked, textFirst } = splitLayoutOf(cell.split_layout)
    return (
      <div style={{ display: 'grid',
                    gridTemplateColumns: stacked ? '1fr' : '1fr 1fr',
                    gap: '1rem',
                    alignItems: 'center', margin: '0.5rem 0' }}>
        {textFirst ? [textPane, figPane] : [figPane, textPane]}
      </div>
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
function Slide({ cells, active, reportFigures, iframeRefs, replayState, onLaunchLive,
                 theme, slideNumber, slideCount }: {
  cells: ReportCell[]
  active: boolean
  reportFigures: ReturnType<typeof useSpyDE>['state']['reportFigures']
  iframeRefs: ReturnType<typeof useSpyDE>['iframeRefs']
  replayState: ReturnType<typeof useSpyDE>['replayState']
  onLaunchLive: (action: LiveAction) => void
  theme: DeckTheme
  slideNumber: number
  slideCount: number
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

  // SHARE the vertical budget between the visual cells on this slide.
  //
  // Every figure used to get a fixed 58vh box, so a slide holding a navigator
  // AND its signal asked for 116vh of an ~85vh stage: the deck overflowed, the
  // second figure was clipped, and the pager sat on top of it. Dividing the
  // budget keeps the whole slide on screen, which is the thing a presentation
  // has to guarantee.
  //
  // It rides on a CSS custom property rather than a computed inline height so
  // the split cell's two panes inherit the same budget without plumbing a prop
  // through every cell type.
  const visualCells = cells.filter(
    c => c.cell_type === 'figure' || c.cell_type === 'split' || c.cell_type === 'image'
      || c.cell_type === 'movie').length
  // Leave room for the markdown around them; never shrink below something
  // readable, and never grow past the single-figure case.
  //
  // `vh` is only the STARTING point — each figure also carries a caption and
  // margins (~28px measured), and hand-computing that against the viewport was
  // wrong twice (24vh and 18vh both still overflowed at five figures). So the
  // boxes also FLEX: `slideInner` is a flex column and each box is
  // `flex: 1 1 <figVh>`, which lets the browser do the arithmetic against the
  // real stage height whatever the captions and markdown around them cost.
  const figVh = visualCells <= 1 ? 58 : Math.max(16, Math.round(62 / visualCells))

  // FILL THE STAGE on a text-only slide.
  //
  // A heading plus a few bullets used to render at prose size, vertically
  // centred in a 60rem column, which on a 1900px projector is a small block
  // floating in a large dark rectangle with dead space above AND below. A slide
  // is not a paragraph — it should use the room it has.
  //
  // Scaled by CONTENT LENGTH rather than by measuring: a sparse slide gets the
  // big type, a dense one keeps prose size so it can't overflow into the
  // pager. Deterministic, no layout thrash, no reflow loop — and the tier is
  // exported as `data-fill` so a spec can assert it without reading font sizes.
  // SPLIT slides qualify too. Their text side is its own column, so scaling it
  // cannot crowd the picture — and a split left at prose size was the same
  // complaint: a heading and four bullets adrift in half a dark rectangle.
  // A slide with a full-width figure/image/movie does NOT qualify: there the
  // text shares the vertical budget with the visual and bigger type pushes it
  // off the stage.
  const splitCells = cells.filter(c => c.cell_type === 'split').length
  const splitOnly = splitCells > 0 && visualCells === splitCells
  const fillable = !isTitle && (visualCells === 0 || splitOnly)
  const textLen = cells.reduce((n, c) => n + (c.source ?? '').length, 0)
  // A split's text lives in HALF the width, so the same character count fills
  // twice the height — the tiers step down accordingly.
  const [lgMax, mdMax] = splitOnly ? [260, 520] : [400, 850]
  const fill = !fillable ? '' : textLen < lgMax ? 'lg' : textLen < mdMax ? 'md' : ''

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
      data-fig-vh={figVh}
      data-fill={fill || 'base'}
      style={{ ...styles.slide, ...styleBg,
        ...(isTitle ? styles.slideTitle : {}),
        ...(active ? styles.slideActive : {}),
        ['--spyde-fig-vh' as any]: `${figVh}vh` }}
    >
      <div
        className={fill ? `present-fill present-fill-${fill}` : undefined}
        style={{ ...styles.slideInner,
        // A slide carrying a figure gets the WIDE column: 60rem is a prose
        // measure (right for text, wrong for data), and capping a figure slide
        // at it left most of a wide screen empty while the plot rendered small.
        ...(visualCells > 0 ? styles.slideInnerWide : {}),
        // A scaled-up text slide needs the column to grow WITH the type:
        // max-width is in `rem` (root-relative), so it is a fixed pixel box —
        // leaving it at 60rem while the body goes 18px → 27px would SHORTEN the
        // measure to ~45 characters and undo the point.
        ...(fill ? styles.slideInnerFill : {}),
        // TOP-align content slides. Centring is right for a title card, but on a
        // content slide it floats the heading at a height that depends on how
        // much text follows it — so consecutive slides visibly jump. Anchoring
        // the heading is what makes a deck read as one deck. A slide whose
        // figures grow to fill has no free space to distribute, so this is a
        // no-op there.
        ...(isTitle ? {} : styles.slideInnerTop),
        ...(isTitle ? styles.slideInnerTitle : {}) }}>
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
      {/* Footer on every slide EXCEPT the title card (which carries its own
          attribution). Outside slideInner so it pins to the slide's bottom
          rather than joining the centred content column. */}
      {!isTitle && theme.footer_show && footerHasContent(theme) && (
        <SlideFooter theme={theme} slideNumber={slideNumber} slideCount={slideCount} />
      )}
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
  if (cell.cell_type === 'split') {
    return (
      <SlideSplit
        cell={cell}
        reportFigures={reportFigures}
        iframeRefs={iframeRefs}
        replayState={replayState}
      />
    )
  }
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

// A SPLIT cell on a slide: text on one side, a figure/photo on the other, in the
// order given by split_layout (text-left | text-right). Renders BOTH panes — the
// earlier bug routed split cells to SlideFigure, which dropped the text entirely.
function SlideSplit({ cell, reportFigures, iframeRefs, replayState }: {
  cell: ReportCell
  reportFigures: ReturnType<typeof useSpyDE>['state']['reportFigures']
  iframeRefs: ReturnType<typeof useSpyDE>['iframeRefs']
  replayState: ReturnType<typeof useSpyDE>['replayState']
}) {
  const html = React.useMemo(() => renderMarkdown(cell.source ?? ''), [cell.source])
  const textPane = (
    <div key="text" className="spyde-md present-md" style={styles.splitText}
         dangerouslySetInnerHTML={{ __html: html }} />
  )
  // The figure side reuses the same figure/photo/pending logic as SlideFigure.
  const fig = reportFigures.get(cell.id)
  const figPane = (
    <div key="fig" style={styles.splitFig}>
      {fig ? (
        <div style={styles.splitFigBox}>
          <SeamlessFigureFrame
            figId={fig.figId} filePath={fig.filePath} title={fig.title}
            iframeRefs={iframeRefs} replayState={replayState} />
        </div>
      ) : cell.image ? (
        <img src={cell.image} alt={cell.caption ?? ''} style={styles.figImg} />
      ) : cell.png ? (
        <img src={cell.png} alt={cell.caption ?? ''} style={styles.figImg} />
      ) : (
        <div style={styles.splitFigBox}><div style={styles.figPending}>rendering…</div></div>
      )}
    </div>
  )
  const { stacked, textFirst, layout } = splitLayoutOf(cell.split_layout)
  return (
    <div data-testid={`present-split-${cell.id}`}
         data-layout={layout}
         style={{ ...styles.splitRow,
                  // Stacked: ONE column, and the text row sizes to its content
                  // while the figure row takes the rest. `1fr 1fr` here would
                  // give a two-line caption half the slide.
                  gridTemplateColumns: stacked ? '1fr' : '1fr 1fr',
                  ...(stacked
                    ? { gridTemplateRows: textFirst ? 'auto 1fr' : '1fr auto' }
                    : {}) }}>
      {textFirst ? [textPane, figPane] : [figPane, textPane]}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9500,
    background: '#14141f', color: '#e8e8f0',
    fontSize: 22, lineHeight: 1.6,
  },
  splitRow: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem',
    // STRETCH, not center: the figure pane has to fill the row's height or the
    // grid sizes to the tallest CONTENT and the slide keeps a band of dead
    // space under it. The text pane opts back out via splitText's alignSelf.
    alignItems: 'stretch', margin: '1rem 0',
    // Grow into the stage like a plain figure cell does.
    flex: '1 1 var(--spyde-fig-vh, 58vh)', minHeight: 0,
  },
  // START, not center: the row is stretched to the stage, so centring the text
  // inside it floats the heading at a height that depends on how many bullets
  // follow — the same drift the content slides had, and it puts the heading out
  // of line with the top of the picture beside it. Long text fills the column
  // either way, so this only changes the sparse case, which is the broken one.
  splitText: { minWidth: 0, alignSelf: 'start', maxHeight: '100%', overflow: 'hidden' },
  splitFig: {
    minWidth: 0, minHeight: 0, display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
  },
  splitFigBox: {
    position: 'relative', width: '100%',
    // Fill the stretched pane rather than a fixed vh (see splitRow).
    flex: '1 1 auto', minHeight: 0,
    border: '1px solid #313244', borderRadius: 8, overflow: 'hidden',
    background: '#0e0e16',
  },
  slide: {
    position: 'absolute', inset: 0, display: 'none',
    flexDirection: 'column', justifyContent: 'center',
    padding: '5vh 8vw', overflowY: 'auto',
  },
  slideActive: { display: 'flex' },
  slideInner: {
    maxWidth: '60rem', margin: '0 auto', width: '100%',
    // A flex COLUMN so the figure boxes above can shrink to share the stage.
    // minHeight:0 lets it actually shrink inside the slide's own flex context.
    // flex:1 1 0 (not the default 1 1 auto) so this TAKES the slide's height
    // instead of sizing to its content — only then can the figure boxes inside
    // shrink to share it rather than overflowing.
    display: 'flex', flexDirection: 'column', minHeight: 0,
    flex: '1 1 0',
    // CENTER the column's content. Load-bearing since `flex: 1 1 0` above: this
    // box now takes the slide's FULL height, so the slide's own
    // `justifyContent:center` has no free space left to distribute and stopped
    // centering anything — a title slide's text jammed against the top of the
    // stage. Centering HERE restores it, and costs nothing on a slide whose
    // figures grow to fill (no free space → nothing to center).
    justifyContent: 'center',
  },
  // A slide with a figure on it: let the data have the screen.
  slideInnerWide: { maxWidth: '96rem' },
  // A scaled-up text slide: wider column so the bigger type keeps a sane
  // measure, and a little top padding so the heading isn't jammed to the edge.
  slideInnerFill: { maxWidth: '78rem', paddingTop: '2vh' },
  // Content slides anchor their heading at the top (see the call site).
  slideInnerTop: { justifyContent: 'flex-start' },
  // A title slide: content vertically + horizontally centered, tighter column.
  slideTitle: { justifyContent: 'center', textAlign: 'center' },
  slideInnerTitle: { maxWidth: '48rem' },
  // Per-slide background presets.
  slideBgPlain: { background: '#0e0e16' },
  slideBgAccent: {
    background:
      'radial-gradient(ellipse at 50% 30%, rgba(137,180,250,0.18), transparent 70%), #14141f',
  },
  // A visual cell is a flex COLUMN that GROWS. The `--spyde-fig-vh` basis still
  // divides the stage between N figures, but grow:1 means one figure on a
  // half-empty slide expands into the dead space instead of stopping at 58vh
  // and leaving the bottom third of the screen blank.
  figure: {
    margin: '1rem 0', textAlign: 'center',
    display: 'flex', flexDirection: 'column',
    flex: '1 1 var(--spyde-fig-vh, 58vh)', minHeight: 0,
  },
  figBox: {
    // position:relative is LOAD-BEARING — SeamlessFigureFrame's frameHost is
    // `position:absolute; inset:0`, so it anchors to the nearest positioned
    // ancestor. Without this the iframe escaped its box and filled the whole
    // slide.
    position: 'relative',
    // Fill whatever height the <figure> above was given, minus the caption.
    // (The vh budget lives on the <figure> now — a fixed height here would
    // pin the box and re-open the dead-space gap.)
    width: '100%', flex: '1 1 auto', minHeight: 0,
    border: '1px solid #313244', borderRadius: 8, overflow: 'hidden',
    background: '#0e0e16',
  },
  figImg: { maxWidth: '100%', maxHeight: '100%', height: 'auto' },
  // A photo cell on a slide — fills its share of the stage, aspect preserved.
  slideImg: {
    display: 'block', margin: '0 auto',
    maxWidth: '100%', maxHeight: '100%',
    flex: '1 1 auto', minHeight: 0,
    objectFit: 'contain', borderRadius: 8,
  },
  figPending: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    height: '100%', color: '#585b70', fontSize: 14,
  },
  figCaption: {
    marginTop: '0.5rem', fontSize: '0.85rem', color: '#a6adc8', fontStyle: 'italic',
    // Never let the caption be squeezed away when the box above it grows.
    flex: '0 0 auto',
  },
  // ── Footer bar ──────────────────────────────────────────────────────────────
  // Pinned to the slide's bottom, INSIDE the slide's horizontal padding, and
  // clear of the pager (which is centred). flex:0 0 auto so it never competes
  // with the figures for the vertical budget.
  footer: {
    flex: '0 0 auto',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    gap: '1.5rem', marginTop: '1.2rem', paddingTop: '0.7rem',
    borderTop: '1px solid var(--spyde-deck-accent, #89b4fa)',
    // The keyline is a hint, not a rule — full-strength accent across the whole
    // slide width reads as a divider competing with the content.
    borderTopColor: 'color-mix(in srgb, var(--spyde-deck-accent, #89b4fa) 35%, transparent)',
    fontSize: '0.8rem', lineHeight: 1.3,
    color: 'var(--spyde-deck-muted, #a6adc8)',
  },
  footerLeft: { display: 'flex', alignItems: 'center', gap: '0.9rem', minWidth: 0 },
  footerLogo: { display: 'block', width: 'auto', objectFit: 'contain', flex: '0 0 auto' },
  footerText: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  footerSep: { margin: '0 0.5rem', opacity: 0.55 },
  footerNum: {
    flex: '0 0 auto', fontVariantNumeric: 'tabular-nums',
    color: 'var(--spyde-deck-muted, #a6adc8)',
  },
  // ── figure diagnostic (D) ───────────────────────────────────────────────────
  diag: {
    position: 'fixed', left: 16, bottom: 16, zIndex: 40,
    maxWidth: '70vw', padding: '10px 12px',
    background: 'rgba(10,10,16,0.94)', border: '1px solid #f9e2af',
    borderRadius: 8, color: '#f9e2af',
    font: '12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
    whiteSpace: 'pre-wrap', wordBreak: 'break-all',
  },
  diagTitle: { fontWeight: 700, marginBottom: 6, color: '#fab387' },
  diagLine: { margin: '1px 0' },
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
    width: 36, height: 36, cursor: 'pointer',
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    padding: 0,
  },
  iconBtnActive: {
    background: '#89b4fa', color: '#11111b', border: '1px solid #89b4fa',
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
  presExitBtn: {
    background: 'rgba(30,30,46,0.9)', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 8, padding: '0 14px', height: 40, fontSize: 14, cursor: 'pointer',
  },
  presExitBtnStrong: {
    background: '#f38ba8', color: '#11111b', border: 'none',
    borderRadius: 8, padding: '0 16px', height: 40, fontSize: 14, fontWeight: 700, cursor: 'pointer',
  },
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
  // Same footprint, no text — keeps the two arrows from jumping together when
  // the count moves to the deck footer.
  counterQuiet: { display: 'inline-block', minWidth: 60 },
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
