/**
 * CellChrome.tsx — the shared hover-chrome pill (absolute-positioned, top-right)
 * shown on a Report cell (ReportCell: markdown; ReportFigureCell: figure;
 * ReportImageCell: photo; ReportSplitCell: split block).
 *
 * All cells show the SAME Copy / Duplicate / Delete trio; each caller adds its
 * own extras via the `leading` / `trailing` slots (a drag handle, a figure Edit
 * toggle + Refresh, a split's layout switch). CellChrome owns just the chrome
 * wrapper + the three shared buttons so the copy/duplicate/delete markup +
 * styling lives once.
 *
 * Wave B de-clutter: the per-cell SLIDE chrome (title-slide 'T', background
 * style '◐', speaker-notes '📝', and the 2-column '▭/◧/◨' toggle + ColumnBadge)
 * was REMOVED. Those roles are re-surfaced slide-natively in Wave C; the backend
 * fields (slide_kind/slide_style/notes/column) remain untouched.
 *
 * Every surviving `data-testid` is preserved EXACTLY (both e2e suites select on
 * them): `cell-copy-<id>`, `cell-duplicate-<id>`, plus a caller-supplied delete
 * testid (`report-cell-delete-<id>` / `report-figcell-delete-<id>` / …).
 */
import React from 'react'

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
  onDuplicate: () => void
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

export function CellChrome({
  cellId, styles, onCopy, onDuplicate, onDelete, deleteTestid,
  deleteTitle = 'Delete cell', leading, trailing,
}: Props) {
  return (
    <div style={styles.chrome}>
      {leading}
      <button
        data-testid={`cell-copy-${cellId}`}
        style={styles.chromeBtn}
        title="Copy cell"
        onClick={onCopy}
      >⧉</button>
      <button
        data-testid={`cell-duplicate-${cellId}`}
        style={styles.chromeBtn}
        title="Duplicate cell"
        onClick={onDuplicate}
      >＋</button>
      {trailing}
      <button
        data-testid={deleteTestid}
        style={styles.deleteBtn ?? styles.chromeBtn}
        title={deleteTitle}
        onClick={onDelete}
      >✕</button>
    </div>
  )
}
