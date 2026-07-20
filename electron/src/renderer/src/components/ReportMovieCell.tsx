/**
 * ReportMovieCell.tsx — a MOVIE cell in the Report sidebar.
 *
 * A movie block is an editable, persistent in-situ movie (source SignalRef +
 * render/edit state → a MovieSpec, persisted to movies/<id>.yaml in the
 * .spyde-report zip). The sidebar card is compact: a poster still (baked on
 * export / loaded from the zip) OR a "drop an in-situ movie / pick a signal"
 * drop-zone when no source is assigned, a caption, and an "Edit ▶" button that
 * opens the full-screen editor (MovieEditor, mounted by MovieGate) via the
 * spyde:movie_edit CustomEvent.
 *
 * The drop-zone accepts a dragged in-situ plot window (WINDOW_DRAG_MIME) →
 * report_set_movie_source; the card also shows a small summary (frames · fps)
 * from the MovieSpec params.
 */
import React, { useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { WINDOW_DRAG_MIME } from '../kernel/dnd'
import type { ReportCell } from '../kernel/protocol'
import { CellChrome } from './CellChrome'

interface Props {
  cell: ReportCell
  index: number
  onRemove: () => void
  onEdit: () => void
  onSetSource: (windowId: number) => void
  dragProps: {
    onDragStart: (e: React.DragEvent) => void
    onDragOver: (e: React.DragEvent) => void
    onDrop: (e: React.DragEvent) => void
    onDragEnd: () => void
    dragging: boolean
    dropBefore: boolean
  }
}

export function ReportMovieCell({ cell, onRemove, onEdit, onSetSource, dragProps }: Props) {
  const { sendAction } = useSpyDE()
  const [hover, setHover] = useState(false)
  const [dropOver, setDropOver] = useState(false)
  const [captionEditing, setCaptionEditing] = useState(false)
  const [captionDraft, setCaptionDraft] = useState(cell.caption ?? '')
  const rootRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (!captionEditing) setCaptionDraft(cell.caption ?? '')
  }, [cell.caption, captionEditing])

  const commitCaption = () => {
    setCaptionEditing(false)
    if (captionDraft !== (cell.caption ?? '')) {
      sendAction('report_set_caption', { cell_id: cell.id, caption: captionDraft })
    }
  }

  const hasSource = Boolean(cell.has_source)
  const params = cell.movie?.params ?? {}
  const fps = Number(params.fps ?? 0)
  const tStart = Number(params.t_start ?? 0)
  const tEnd = Number(params.t_end ?? 0)
  const nRange = Math.max(0, tEnd - tStart + 1)

  // ── Source drop (a dragged in-situ window pill) ───────────────────────────
  const onDrop = (e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes(WINDOW_DRAG_MIME)) return
    e.preventDefault()
    e.stopPropagation()
    setDropOver(false)
    const wid = parseInt(e.dataTransfer.getData(WINDOW_DRAG_MIME), 10)
    if (!Number.isNaN(wid)) onSetSource(wid)
  }
  const onDragOver = (e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes(WINDOW_DRAG_MIME)) { dragProps.onDragOver(e); return }
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
    if (!dropOver) setDropOver(true)
  }

  return (
    <div
      ref={rootRef}
      data-testid={`report-moviecell-${cell.id}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onDragOver={onDragOver}
      onDragLeave={() => setDropOver(false)}
      onDrop={onDrop}
      onDragEnd={dragProps.onDragEnd}
      style={{
        ...styles.cell,
        ...(dragProps.dragging ? styles.cellDragging : {}),
        ...(dragProps.dropBefore ? styles.cellDropBefore : {}),
      }}
    >
      {hover && (
        <CellChrome
          cellId={cell.id}
          styles={{ chrome: styles.chrome, chromeBtn: styles.chromeBtn }}
          onCopy={() => { /* movie copy is a Phase-3 nicety; no-op for now */ }}
          onDuplicate={() => { /* handled in a later phase */ }}
          onDelete={onRemove}
          deleteTestid={`report-moviecell-delete-${cell.id}`}
          deleteTitle="Delete movie"
          leading={
            <span
              data-testid={`report-moviecell-drag-${cell.id}`}
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

      {/* Poster / drop-zone. */}
      {hasSource && cell.poster ? (
        <button
          data-testid={`report-moviecell-poster-${cell.id}`}
          style={styles.posterBtn}
          onClick={onEdit}
          title="Edit movie"
        >
          <img src={cell.poster} alt={cell.caption ?? 'movie'} style={styles.poster} draggable={false} />
          <span style={styles.playBadge}>▶</span>
        </button>
      ) : hasSource ? (
        // Source assigned but no poster baked yet — an "open the editor" prompt.
        <button
          data-testid={`report-moviecell-open-${cell.id}`}
          style={{ ...styles.dropZone, ...styles.dropZoneReady }}
          onClick={onEdit}
        >
          <span style={styles.filmIcon}>🎬</span>
          <span>Open the movie editor to preview & export</span>
        </button>
      ) : (
        <div
          data-testid={`report-moviecell-dropzone-${cell.id}`}
          style={{ ...styles.dropZone, ...(dropOver ? styles.dropZoneActive : {}) }}
        >
          <span style={styles.filmIcon}>🎬</span>
          <span>Drop an in-situ movie window here to pick a signal</span>
        </div>
      )}

      {/* Summary line (frames · fps) for a sourced movie. */}
      {hasSource && (
        <div style={styles.summary} data-testid={`report-moviecell-summary-${cell.id}`}>
          {nRange > 0 ? `${nRange} frames` : 'movie'}{fps > 0 ? ` · ${fps} fps` : ''}
          <button
            data-testid={`report-moviecell-edit-${cell.id}`}
            style={styles.editBtn}
            onClick={onEdit}
            title="Edit movie"
          >Edit ▶</button>
        </div>
      )}

      {/* Caption (click-to-edit → report_set_caption). */}
      {captionEditing ? (
        <input
          data-testid={`report-moviecell-caption-input-${cell.id}`}
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
          data-testid={`report-moviecell-caption-${cell.id}`}
          style={styles.caption}
          title="Click to edit caption"
          onClick={() => { setCaptionDraft(cell.caption ?? ''); setCaptionEditing(true) }}
        >
          {(cell.caption ?? '').trim()
            ? cell.caption
            : <span style={styles.captionPlaceholder}>Add a caption…</span>}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  cell: {
    position: 'relative', marginBottom: 8, borderRadius: 6,
    borderTop: '2px solid transparent',
  },
  cellDragging: { opacity: 0.4 },
  cellDropBefore: { borderTop: '2px solid #89b4fa' },
  dragHandle: {
    cursor: 'grab', color: '#6c7086', fontSize: 13, userSelect: 'none', lineHeight: 1,
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
  posterBtn: {
    position: 'relative', display: 'block', width: '100%', padding: 0,
    border: '1px solid #313244', borderRadius: 6, background: '#11111b',
    cursor: 'pointer', overflow: 'hidden',
  },
  poster: { display: 'block', width: '100%', height: 'auto' },
  playBadge: {
    position: 'absolute', top: '50%', left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 38, height: 38, borderRadius: '50%',
    background: 'rgba(17,17,27,0.66)', color: '#cdd6f4',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 16, paddingLeft: 3, pointerEvents: 'none',
  },
  dropZone: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 8, minHeight: 96,
    border: '1px dashed #45475a', borderRadius: 6, background: '#11111b',
    color: '#6c7086', fontSize: 11.5, textAlign: 'center', padding: '14px 10px',
    cursor: 'default', width: '100%', boxSizing: 'border-box',
  },
  dropZoneActive: { borderColor: '#89b4fa', color: '#89b4fa' },
  dropZoneReady: { cursor: 'pointer', borderStyle: 'solid', color: '#a6adc8' },
  filmIcon: { fontSize: 22, lineHeight: 1 },
  summary: {
    display: 'flex', alignItems: 'center', gap: 8,
    fontSize: 11, color: '#a6adc8', padding: '5px 2px 2px',
  },
  editBtn: {
    marginLeft: 'auto',
    background: '#89b4fa', color: '#11111b', border: 'none', borderRadius: 5,
    padding: '2px 10px', fontSize: 11, fontWeight: 700, cursor: 'pointer',
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
