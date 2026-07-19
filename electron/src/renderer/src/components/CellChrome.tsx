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
  /** Style for the column toggle when a column is active (left/right). Defaults
   *  to `chromeBtn` when omitted. */
  columnBtnActive?: React.CSSProperties
}

/** The per-cell column value + cycle handler for the slide 2-column layout.
 *  '' / 'full' = full-width (default); 'left' / 'right' = a slide column. The
 *  toggle button cycles full → left → right → full. */
export type CellColumn = '' | 'full' | 'left' | 'right'

// The cycle order + the glyph/label per state.
const COLUMN_CYCLE: CellColumn[] = ['', 'left', 'right']
const COLUMN_GLYPH: Record<string, string> = {
  '': '▭', full: '▭', left: '◧', right: '◨',
}
const COLUMN_LABEL: Record<string, string> = {
  '': 'Full width', full: 'Full width', left: 'Left column', right: 'Right column',
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
  /** The cell's slide column ('' / 'full' / 'left' / 'right'). When
   *  `onSetColumn` is also given, a 3-way toggle button (▭ full → ◧ left →
   *  ◨ right) renders between `leading` and Copy. Omit both to hide it. */
  column?: string
  onSetColumn?: (column: CellColumn) => void
}

/** A small ALWAYS-VISIBLE badge (top-left of a cell) marking its slide column —
 *  "◧ Left" / "◨ Right" — so the 2-column intent reads in the vertical cell
 *  list even when the cell isn't hovered. Renders nothing for a full-width cell.
 *  The sidebar keeps its linear DnD list (a real side-by-side grid would break
 *  the per-cell reorder + the vertical drop-index math), so the badge is how the
 *  authoring view signals columns; Present mode + export render them for real. */
export function ColumnBadge({ column }: { column?: string }) {
  const cur = column === 'left' || column === 'right' ? column : ''
  if (!cur) return null
  return (
    <span
      data-testid="cell-column-badge"
      data-column={cur}
      title={`This cell is the ${COLUMN_LABEL[cur].toLowerCase()} of its slide`}
      style={badgeStyle}
    >{COLUMN_GLYPH[cur]} {cur === 'left' ? 'Left' : 'Right'}</span>
  )
}

const badgeStyle: React.CSSProperties = {
  position: 'absolute', top: 2, left: 6, zIndex: 2,
  display: 'inline-flex', alignItems: 'center', gap: 3,
  fontSize: 9, fontWeight: 700, letterSpacing: 0.2,
  color: '#89b4fa', background: 'rgba(137,180,250,0.12)',
  border: '1px solid rgba(137,180,250,0.35)', borderRadius: 4,
  padding: '1px 5px', lineHeight: 1.3, userSelect: 'none',
  pointerEvents: 'none',
}

export function CellChrome({
  cellId, styles, onCopy, onDuplicate, onDelete, deleteTestid,
  deleteTitle = 'Delete cell', leading, trailing, column, onSetColumn,
}: Props) {
  const cur: CellColumn =
    column === 'left' || column === 'right' ? column : ''
  const cycleColumn = () => {
    if (!onSetColumn) return
    const idx = COLUMN_CYCLE.indexOf(cur)
    onSetColumn(COLUMN_CYCLE[(idx + 1) % COLUMN_CYCLE.length])
  }
  return (
    <div style={styles.chrome}>
      {leading}
      {onSetColumn && (
        <button
          data-testid={`cell-column-${cellId}`}
          data-column={cur || 'full'}
          style={cur ? (styles.columnBtnActive ?? styles.chromeBtn) : styles.chromeBtn}
          title={`Slide layout: ${COLUMN_LABEL[cur]} — click to cycle (full → left → right)`}
          onClick={cycleColumn}
        >{COLUMN_GLYPH[cur]}</button>
      )}
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
