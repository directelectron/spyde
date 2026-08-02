/**
 * CellChrome.tsx — the shared hover-chrome pill (absolute-positioned, top-right)
 * shown on a Report cell (ReportCell: markdown; ReportFigureCell: figure;
 * ReportImageCell: photo; ReportSplitCell: split block).
 *
 * All cells show Copy / Delete; a FIGURE-bearing cell also gets ＋ "Add another
 * figure". Each caller adds its own extras via the `leading` / `trailing` slots
 * (a drag handle, a figure Edit toggle + Refresh, a split's layout switch).
 *
 * The ＋ used to be "Duplicate cell". Duplicating a slide's cell is a thing
 * nobody asked for, while COMBINING two figures into a subplot grid had no
 * button at all — it was reachable only by dragging a window pill onto an
 * existing figure and hitting a 28%-wide edge strip, which is fiddly to aim at
 * and easy to miss entirely. ＋ now opens a window picker and tiles the choice
 * in, so the grid has a click path that can't be fumbled.
 *
 * Wave B de-clutter: the per-cell SLIDE chrome (title-slide 'T', background
 * style '◐', and speaker-notes '📝') was REMOVED. Those roles are re-surfaced
 * slide-natively in Wave C; the backend fields (slide_kind/slide_style/notes)
 * remain untouched.
 *
 * Testids: `cell-copy-<id>`, `cell-add-figure-<id>` (was `cell-duplicate-<id>`),
 * plus a caller-supplied delete testid (`report-cell-delete-<id>` /
 * `report-figcell-delete-<id>` / …).
 */
import React from 'react'

// Shared hover feedback for a chrome button (they use inline styles, not CSS
// classes, so hover is wired per-button). A subtle raised background on hover
// makes the small icon buttons feel like real, clickable targets.
const hoverProps = {
  onMouseEnter: (e: React.MouseEvent<HTMLButtonElement>) => {
    e.currentTarget.style.background = 'rgba(137,180,250,0.18)'
  },
  onMouseLeave: (e: React.MouseEvent<HTMLButtonElement>) => {
    e.currentTarget.style.background = 'none'
  },
}

export interface CellChromeStyles {
  /** The absolute-positioned wrapper pill. */
  chrome: React.CSSProperties
  /** A plain (non-active) chrome button — copy/duplicate default to this. */
  chromeBtn: React.CSSProperties
  /** Delete button style, if it differs from `chromeBtn` (ReportCell's original
   *  delete button was 1px smaller than its copy/duplicate buttons). Defaults
   *  to `chromeBtn` when omitted. */
  deleteBtn?: React.CSSProperties
}

interface Props {
  cellId: string
  styles: CellChromeStyles
  onCopy: () => void
  /** Figure-bearing cells only: the ＋ button. Opens a picker of open windows
   *  and tiles the chosen one in beside this figure (→ a subplot grid). Omitted
   *  on cells with no figure, which simply don't render a ＋. */
  onAddFigure?: () => void
  onDelete: () => void
  deleteTestid: string
  deleteTitle?: string
  /** Extra buttons rendered BEFORE Copy (e.g. a drag handle, a figure Edit
   *  toggle). */
  leading?: React.ReactNode
  /** Extra buttons rendered AFTER Duplicate, BEFORE Delete (e.g. a figure's
   *  Refresh, a split block's layout switch). */
  trailing?: React.ReactNode
}

/** How long an armed delete stays armed before disarming itself. Long enough to
 *  move the pointer a few px and click again; short enough that a stray armed
 *  button never survives to the next time you reach for the chrome. */
const ARM_MS = 2600

export function CellChrome({
  cellId, styles, onCopy, onAddFigure, onDelete, deleteTestid,
  deleteTitle = 'Delete cell', leading, trailing,
}: Props) {
  // TWO-STEP DELETE. This chrome is shown on `hover || showEditor`, so opening a
  // figure's edit toolbar MAKES this ✕ appear — top-right, exactly where the
  // editor's own "Close" × sits, and one click from destroying the slide. The
  // first click now only ARMS it; the second confirms. Undo (report_undo) is the
  // safety net underneath, but not losing the slide in the first place beats
  // getting it back.
  const [armed, setArmed] = React.useState(false)
  const timer = React.useRef<number | null>(null)

  const disarm = React.useCallback(() => {
    if (timer.current != null) { window.clearTimeout(timer.current); timer.current = null }
    setArmed(false)
  }, [])

  React.useEffect(() => disarm, [disarm])   // never leave a timer behind

  const clickDelete = () => {
    if (armed) { disarm(); onDelete(); return }
    setArmed(true)
    if (timer.current != null) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => { timer.current = null; setArmed(false) }, ARM_MS)
  }

  return (
    <div style={styles.chrome} onMouseLeave={disarm}>
      {leading}
      <button
        data-testid={`cell-copy-${cellId}`}
        style={styles.chromeBtn}
        title="Copy cell"
        onClick={onCopy}
        {...hoverProps}
      >⧉</button>
      {onAddFigure && (
        <button
          data-testid={`cell-add-figure-${cellId}`}
          style={styles.chromeBtn}
          title="Add another figure — tiles it beside this one"
          onClick={onAddFigure}
          {...hoverProps}
        >＋</button>
      )}
      {trailing}
      <button
        data-testid={deleteTestid}
        data-armed={armed ? 'true' : 'false'}
        style={armed
          ? { ...(styles.deleteBtn ?? styles.chromeBtn), ...armedStyle }
          : (styles.deleteBtn ?? styles.chromeBtn)}
        title={armed ? 'Click again to delete' : deleteTitle}
        onClick={clickDelete}
        {...hoverProps}
      >{armed ? 'Delete?' : '✕'}</button>
    </div>
  )
}

// Armed state reads as a warning, and the label change is the real signal — a
// recolour alone is easy to miss on a small glyph in a dark UI.
const armedStyle: React.CSSProperties = {
  background: '#f38ba8', color: '#11111b', fontWeight: 700,
  borderRadius: 4, padding: '0 6px', fontSize: 10.5, width: 'auto',
}
