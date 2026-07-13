/**
 * CellChrome.tsx — the shared hover-chrome pill (absolute-positioned, top-right)
 * shown on a Report cell (ReportCell.tsx: markdown; ReportFigureCell.tsx: figure).
 *
 * Both cells show the SAME Copy / Duplicate / Delete trio; ReportCell also has a
 * leading drag handle, ReportFigureCell also has a leading Edit toggle and a
 * trailing Refresh button. Rather than force one fixed button set, CellChrome
 * takes `leading`/`trailing` slots for the cell-specific extras and owns just the
 * chrome wrapper + the three shared buttons — so each caller keeps its own extra
 * affordances while the copy/duplicate/delete markup (and styling) lives once.
 *
 * Every existing `data-testid` is preserved EXACTLY as it was before this
 * extraction (both e2e suites select on them):
 *   `cell-copy-<id>`, `cell-duplicate-<id>`, plus a caller-supplied delete
 *   testid (`report-cell-delete-<id>` / `report-figcell-delete-<id>`).
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
  /** Extra buttons rendered BEFORE Copy (e.g. ReportCell's drag handle,
   *  ReportFigureCell's Edit toggle). */
  leading?: React.ReactNode
  /** Extra buttons rendered AFTER Duplicate, BEFORE Delete (e.g.
   *  ReportFigureCell's Refresh-from-live). */
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
