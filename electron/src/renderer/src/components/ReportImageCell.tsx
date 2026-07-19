/**
 * ReportImageCell.tsx — a PHOTO/IMAGE cell in the Report sidebar.
 *
 * A plain image the user dropped a file for, pasted from the clipboard, or
 * browsed to (ReportSidebar owns those three add paths → report_add_image_cell).
 * The backend ships the bytes as a data URL (`cell.image`); this renders them
 * inline as a resizable `<img>` with an editable caption below — the same
 * click-to-edit caption + hover chrome (Copy / Duplicate / Delete + a reorder
 * drag handle) the figure cell uses.
 *
 * RESIZE: a bottom-right drag handle scales the image's display width as a
 * PERCENT of the cell (10–100%), persisted per-cell in localStorage so it
 * survives re-renders / reopen. The stored width is display-only (never sent to
 * the backend — the report is authored on this machine and the width is a view
 * preference, like the markdown font size).
 */
import React, { useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { reportClipboard, type SerializedImageCell } from '../kernel/reportClipboard'
import type { ReportCell } from '../kernel/protocol'
import { CellChrome, ColumnBadge, type CellColumn } from './CellChrome'
import { SlideNotesEditor } from './SlideNotesEditor'

const WIDTH_KEY = (id: string) => `spyde-report-imgw-${id}`
const DEFAULT_WIDTH_PCT = 100
const MIN_WIDTH_PCT = 10

interface Props {
  cell: ReportCell
  onRemove: () => void
  /** Own index in the cell list (Duplicate → insert at index+1). */
  index: number
  /** This cell STARTS a slide (first cell or a slide_break) — offer the
   *  per-slide "Title slide" toggle in the chrome. */
  slideStart?: boolean
  /** HTML5 DnD reorder wiring supplied by the parent list (same shape as the
   *  markdown/figure cells' — ReportSidebar.makeDragProps). */
  dragProps: {
    onDragStart: (e: React.DragEvent) => void
    onDragOver: (e: React.DragEvent) => void
    onDrop: (e: React.DragEvent) => void
    onDragEnd: () => void
    dragging: boolean
    dropBefore: boolean
  }
}

export function ReportImageCell({ cell, onRemove, index, slideStart, dragProps }: Props) {
  const { sendAction } = useSpyDE()
  const [hover, setHover] = useState(false)
  const [notesOpen, setNotesOpen] = useState(false)
  const [captionEditing, setCaptionEditing] = useState(false)
  const [captionDraft, setCaptionDraft] = useState(cell.caption ?? '')
  const [widthPct, setWidthPct] = useState<number>(() => {
    const v = Number(localStorage.getItem(WIDTH_KEY(cell.id)))
    return Number.isFinite(v) && v >= MIN_WIDTH_PCT && v <= 100 ? v : DEFAULT_WIDTH_PCT
  })
  const rootRef = React.useRef<HTMLDivElement>(null)
  const boxRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (!captionEditing) setCaptionDraft(cell.caption ?? '')
  }, [cell.caption, captionEditing])

  const commitCaption = () => {
    setCaptionEditing(false)
    if (captionDraft !== (cell.caption ?? '')) {
      sendAction('report_set_caption', { cell_id: cell.id, caption: captionDraft })
    }
  }

  // ── Resize (Pointer-Capture on a bottom-right handle) ─────────────────────
  const resizeGesture = React.useRef<{ px: number; boxW: number; startPct: number } | null>(null)
  const onResizeDown = (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId) } catch { /* */ }
    const boxW = boxRef.current?.parentElement?.clientWidth ?? 1
    resizeGesture.current = { px: e.clientX, boxW: Math.max(1, boxW), startPct: widthPct }
  }
  const onResizeMove = (e: React.PointerEvent) => {
    const g = resizeGesture.current
    if (!g) return
    const deltaPct = ((e.clientX - g.px) / g.boxW) * 100
    const next = Math.min(100, Math.max(MIN_WIDTH_PCT, Math.round(g.startPct + deltaPct)))
    setWidthPct(next)
  }
  const onResizeUp = (e: React.PointerEvent) => {
    if (!resizeGesture.current) return
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId) } catch { /* */ }
    resizeGesture.current = null
    localStorage.setItem(WIDTH_KEY(cell.id), String(widthPct))
  }

  // ── Copy / Duplicate ──────────────────────────────────────────────────────
  const serialize = (): SerializedImageCell => ({
    cell_type: 'image',
    caption: cell.caption ?? '',
    image_ext: cell.image_ext ?? 'png',
    image: cell.image ?? '',
  })
  const doCopy = async () => {
    const ser = serialize()
    reportClipboard.set(ser)
    // Best-effort: mirror the image to the OS clipboard so a paste into an
    // external app works too. Ignore failure (internal clipboard is the truth).
    if (ser.image) {
      try { await window.electron.clipboardWritePng(ser.image) } catch { /* ignore */ }
    }
  }
  const doDuplicate = () =>
    sendAction('report_paste_cell', { cell: serialize(), index: index + 1 })

  return (
    <div
      ref={rootRef}
      data-testid={`report-imgcell-${cell.id}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onDragOver={dragProps.onDragOver}
      onDrop={dragProps.onDrop}
      onDragEnd={dragProps.onDragEnd}
      style={{
        ...styles.cell,
        ...(dragProps.dragging ? styles.cellDragging : {}),
        ...(dragProps.dropBefore ? styles.cellDropBefore : {}),
      }}
    >
      {/* Always-visible slide-column badge (◧ Left / ◨ Right). */}
      <ColumnBadge column={cell.column} />
      {hover && (
        <CellChrome
          cellId={cell.id}
          styles={{ chrome: styles.chrome, chromeBtn: styles.chromeBtn, columnBtnActive: styles.columnBtnActive }}
          onCopy={doCopy}
          onDuplicate={doDuplicate}
          onDelete={onRemove}
          column={cell.column}
          onSetColumn={(c: CellColumn) => sendAction('report_set_cell_column', { cell_id: cell.id, column: c })}
          slideStart={slideStart}
          slideKind={cell.slide_kind}
          onToggleTitle={() => sendAction('report_set_slide_kind', { cell_id: cell.id })}
          slideStyle={cell.slide_style}
          onCycleStyle={(style) => sendAction('report_set_slide_style', { cell_id: cell.id, slide_style: style })}
          slideNotes={cell.notes}
          notesOpen={notesOpen}
          onToggleNotes={() => setNotesOpen((v) => !v)}
          deleteTestid={`report-imgcell-delete-${cell.id}`}
          deleteTitle="Delete image"
          leading={
            <span
              data-testid={`report-imgcell-drag-${cell.id}`}
              style={styles.dragHandle}
              title="Drag to reorder"
              draggable
              onDragStart={(e) => {
                dragProps.onDragStart(e)
                if (rootRef.current) e.dataTransfer.setDragImage(rootRef.current, 24, 16)
              }}
              onDragEnd={dragProps.onDragEnd}
            >⠿</span>
          }
        />
      )}

      {/* The image, sized to widthPct of the cell + centered, with a resize
          handle at its bottom-right corner. */}
      <div style={styles.imgWrap}>
        <div
          ref={boxRef}
          style={{ ...styles.imgBox, width: `${widthPct}%` }}
          data-testid={`report-imgcell-box-${cell.id}`}
        >
          {cell.image ? (
            <img
              src={cell.image}
              alt={cell.caption ?? ''}
              style={styles.img}
              data-testid={`report-imgcell-img-${cell.id}`}
              draggable={false}
            />
          ) : (
            <div style={styles.missing} data-testid={`report-imgcell-missing-${cell.id}`}>
              image unavailable
            </div>
          )}
          {cell.image && (
            <div
              data-testid={`report-imgcell-resize-${cell.id}`}
              title="Drag to resize"
              style={styles.resizeHandle}
              onPointerDown={onResizeDown}
              onPointerMove={onResizeMove}
              onPointerUp={onResizeUp}
              onPointerCancel={onResizeUp}
            >⟟</div>
          )}
        </div>
      </div>

      {/* Caption line (click-to-edit → report_set_caption). */}
      {captionEditing ? (
        <input
          data-testid={`report-imgcell-caption-input-${cell.id}`}
          autoFocus
          style={styles.captionInput}
          value={captionDraft}
          onChange={(e) => setCaptionDraft(e.target.value)}
          onBlur={commitCaption}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
            else if (e.key === 'Escape') { setCaptionDraft(cell.caption ?? ''); setCaptionEditing(false) }
          }}
        />
      ) : (
        <div
          data-testid={`report-imgcell-caption-${cell.id}`}
          style={styles.caption}
          title="Click to edit caption"
          onClick={() => { setCaptionDraft(cell.caption ?? ''); setCaptionEditing(true) }}
        >
          {(cell.caption ?? '').trim()
            ? cell.caption
            : <span style={styles.captionPlaceholder}>Add a caption…</span>}
        </div>
      )}

      {/* Speaker-notes editor (slide-starting cells only), toggled from the
          chrome 📝 button. Debounced → report_set_slide_notes. */}
      {slideStart && notesOpen && (
        <SlideNotesEditor
          cellId={cell.id}
          notes={cell.notes ?? ''}
          onCommit={(notes) => sendAction('report_set_slide_notes', { cell_id: cell.id, notes })}
          onClose={() => setNotesOpen(false)}
        />
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  cell: {
    position: 'relative',
    marginBottom: 8,
    borderRadius: 6,
    borderTop: '2px solid transparent',
  },
  cellDragging: { opacity: 0.4 },
  cellDropBefore: { borderTop: '2px solid #89b4fa' },
  dragHandle: {
    cursor: 'grab', color: '#6c7086', fontSize: 13, userSelect: 'none',
    lineHeight: 1,
  },
  chrome: {
    position: 'absolute', top: 4, right: 6, zIndex: 4,
    display: 'flex', alignItems: 'center', gap: 4,
    background: 'rgba(24,24,37,0.92)', borderRadius: 5, padding: '1px 3px',
  },
  chromeBtn: {
    background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
    fontSize: 13, padding: '0 3px', lineHeight: 1,
  },
  columnBtnActive: {
    background: 'none', border: 'none', color: '#89b4fa', cursor: 'pointer',
    fontSize: 13, padding: '0 3px', lineHeight: 1,
  },
  imgWrap: {
    display: 'flex', justifyContent: 'center', width: '100%',
  },
  imgBox: {
    position: 'relative', maxWidth: '100%',
    // width set per-instance (widthPct).
  },
  img: {
    display: 'block', width: '100%', height: 'auto',
    borderRadius: 6, border: '1px solid #313244',
  },
  missing: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    minHeight: 80, color: '#6c7086', fontSize: 12,
    border: '1px dashed #45475a', borderRadius: 6,
  },
  resizeHandle: {
    position: 'absolute', right: -2, bottom: -2, zIndex: 3,
    width: 16, height: 16, cursor: 'nwse-resize',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: '#89b4fa', fontSize: 12, lineHeight: 1,
    background: 'rgba(24,24,37,0.85)', borderRadius: 4,
    transform: 'rotate(90deg)',
    userSelect: 'none', touchAction: 'none',
  },
  caption: {
    fontSize: 11.5, color: '#a6adc8', padding: '4px 2px',
    cursor: 'text', lineHeight: 1.4, textAlign: 'center',
  },
  captionPlaceholder: { color: '#585b70', fontStyle: 'italic' },
  captionInput: {
    width: '100%', boxSizing: 'border-box', marginTop: 3,
    background: '#11111b', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 5,
    padding: '3px 6px', fontSize: 11.5, outline: 'none', textAlign: 'center',
  },
}
