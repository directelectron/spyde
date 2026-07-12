/**
 * ReportFigureCell.tsx — a figure cell in the Report sidebar.
 *
 * Three states:
 *   • placeholder (template drop zone) — dashed box with caption text; drop a
 *     compatible figure/window pill onto it to FILL it (report_add_figure
 *     {source_window_id, at_cell:id}).
 *   • data_offline — the SignalRef couldn't rebind: show the baked PNG (cell.png)
 *     + a small "data offline" badge.
 *   • live — the report figure iframe for fig_id, reusing the exact
 *     iframeRefs/replayState mounting pattern from WindowContent (sandboxed
 *     file:// src, register in iframeRefs, replayState on load).
 *
 * Below the figure: an editable caption (click-to-edit → report_set_caption) +
 * hover chrome (Refresh-from-live → report_refresh_figure, delete →
 * report_remove_cell).
 */
import React, { useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { ReportCell } from '../kernel/protocol'
import { FIGURE_DRAG_MIME, WINDOW_DRAG_MIME } from '../kernel/dnd'

// Resolve a source window id from a FIGURE_DRAG_MIME or WINDOW_DRAG_MIME drop.
function sourceWindowIdFromDrop(dt: DataTransfer): number | null {
  const fig = dt.getData(FIGURE_DRAG_MIME)
  if (fig) {
    try {
      const { windowId } = JSON.parse(fig) as { windowId?: number }
      if (typeof windowId === 'number') return windowId
    } catch { /* malformed */ }
  }
  const win = dt.getData(WINDOW_DRAG_MIME)
  if (win) {
    const n = parseInt(win, 10)
    if (Number.isFinite(n)) return n
  }
  return null
}

const DROP_MIMES = [FIGURE_DRAG_MIME, WINDOW_DRAG_MIME]

interface Props {
  cell: ReportCell
  onRemove: () => void
}

export function ReportFigureCell({ cell, onRemove }: Props) {
  const { state, iframeRefs, replayState, sendAction } = useSpyDE()
  const [captionEditing, setCaptionEditing] = useState(false)
  const [captionDraft, setCaptionDraft] = useState(cell.caption ?? '')
  const [hover, setHover] = useState(false)
  const [dropHover, setDropHover] = useState(false)

  React.useEffect(() => {
    if (!captionEditing) setCaptionDraft(cell.caption ?? '')
  }, [cell.caption, captionEditing])

  const fig = state.reportFigures.get(cell.id)

  const commitCaption = () => {
    setCaptionEditing(false)
    if (captionDraft !== (cell.caption ?? '')) {
      sendAction('report_set_caption', { cell_id: cell.id, caption: captionDraft })
    }
  }

  // A placeholder cell IS a drop target: dropping a compatible pill fills it.
  const onDragOver = (e: React.DragEvent) => {
    if (!DROP_MIMES.some(m => e.dataTransfer.types.includes(m))) return
    e.preventDefault()
    e.stopPropagation()   // don't also trigger the sidebar-body insertion logic
    e.dataTransfer.dropEffect = 'copy'
    setDropHover(true)
  }
  const onDrop = (e: React.DragEvent) => {
    if (!DROP_MIMES.some(m => e.dataTransfer.types.includes(m))) return
    e.preventDefault()
    e.stopPropagation()
    setDropHover(false)
    const src = sourceWindowIdFromDrop(e.dataTransfer)
    if (src != null) sendAction('report_add_figure', { source_window_id: src, at_cell: cell.id })
  }

  return (
    <div
      data-testid={`report-figcell-${cell.id}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={styles.cell}
    >
      {/* Hover chrome: Refresh-from-live + delete (not on a placeholder). */}
      {hover && !cell.placeholder && (
        <div style={styles.chrome}>
          <button
            data-testid={`report-figcell-refresh-${cell.id}`}
            style={styles.chromeBtn}
            title="Refresh from live figure"
            onClick={() => sendAction('report_refresh_figure', { cell_id: cell.id })}
          >⟳</button>
          <button
            data-testid={`report-figcell-delete-${cell.id}`}
            style={styles.chromeBtn}
            title="Delete figure"
            onClick={onRemove}
          >✕</button>
        </div>
      )}

      {cell.placeholder ? (
        // Template placeholder — dashed drop zone.
        <div
          data-testid={`report-figcell-placeholder-${cell.id}`}
          onDragOver={onDragOver}
          onDragLeave={() => setDropHover(false)}
          onDrop={onDrop}
          style={{ ...styles.placeholder, ...(dropHover ? styles.placeholderHot : {}) }}
        >
          <div style={styles.placeholderIcon}>▤</div>
          <div style={styles.placeholderText}>
            {cell.caption || 'Drop a figure here'}
          </div>
        </div>
      ) : cell.data_offline ? (
        // Rebind failed — show the baked snapshot + a "data offline" badge.
        <div style={styles.figBox}>
          {cell.png
            ? <img src={cell.png} alt={cell.caption ?? ''} style={styles.offlineImg} />
            : <div style={styles.offlineMissing}>snapshot unavailable</div>}
          <span style={styles.offlineBadge} data-testid={`report-figcell-offline-${cell.id}`}>
            data offline
          </span>
        </div>
      ) : fig ? (
        // Live report figure iframe — same mounting/replay pattern as WindowContent.
        <div style={styles.figBox}>
          <iframe
            key={fig.figId}
            ref={el => {
              if (el) iframeRefs.current.set(fig.figId, el)
              else iframeRefs.current.delete(fig.figId)
            }}
            src={fig.filePath ?? undefined}
            onLoad={(e) => {
              replayState(fig.figId)
              const el = e.currentTarget
              window.electron.resizeFigure(fig.figId, Math.max(80, el.clientWidth), Math.max(80, el.clientHeight))
            }}
            style={styles.frame}
            title={fig.title}
            data-testid={`figure-${fig.figId}`}
          />
        </div>
      ) : (
        // Figure cell whose iframe hasn't arrived yet (or its data-URL PNG
        // fallback is the only thing present) — show the baked PNG if any.
        <div style={styles.figBox}>
          {cell.png
            ? <img src={cell.png} alt={cell.caption ?? ''} style={styles.offlineImg} />
            : <div style={styles.pending} data-testid={`report-figcell-pending-${cell.id}`}>rendering…</div>}
        </div>
      )}

      {/* Caption line (not on a placeholder — its text lives in the drop zone). */}
      {!cell.placeholder && (
        captionEditing ? (
          <input
            data-testid={`report-figcell-caption-input-${cell.id}`}
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
            data-testid={`report-figcell-caption-${cell.id}`}
            style={styles.caption}
            title="Click to edit caption"
            onClick={() => { setCaptionDraft(cell.caption ?? ''); setCaptionEditing(true) }}
          >
            {(cell.caption ?? '').trim()
              ? cell.caption
              : <span style={styles.captionPlaceholder}>Add a caption…</span>}
          </div>
        )
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  cell: {
    position: 'relative',
    marginBottom: 8,
    borderRadius: 6,
  },
  chrome: {
    position: 'absolute', top: 4, right: 6, zIndex: 3,
    display: 'flex', alignItems: 'center', gap: 4,
    background: 'rgba(24,24,37,0.92)', borderRadius: 5, padding: '1px 3px',
  },
  chromeBtn: {
    background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
    fontSize: 13, padding: '0 3px', lineHeight: 1,
  },
  figBox: {
    position: 'relative', width: '100%',
    aspectRatio: '16 / 10',
    background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    overflow: 'hidden',
  },
  frame: {
    position: 'absolute', inset: 0, width: '100%', height: '100%',
    border: 'none',
  },
  offlineImg: {
    position: 'absolute', inset: 0, width: '100%', height: '100%',
    objectFit: 'contain',
  },
  offlineMissing: {
    position: 'absolute', inset: 0, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
    color: '#6c7086', fontSize: 12,
  },
  pending: {
    position: 'absolute', inset: 0, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
    color: '#6c7086', fontSize: 12,
  },
  offlineBadge: {
    position: 'absolute', top: 6, left: 6, zIndex: 2,
    background: 'rgba(243,139,168,0.18)', color: '#f38ba8',
    border: '1px solid rgba(243,139,168,0.4)', borderRadius: 4,
    fontSize: 9.5, fontWeight: 600, padding: '1px 6px',
  },
  placeholder: {
    width: '100%', aspectRatio: '16 / 10',
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 6,
    border: '2px dashed #45475a', borderRadius: 8,
    background: 'rgba(30,30,46,0.4)', color: '#6c7086',
    transition: 'border-color 90ms, background 90ms',
  },
  placeholderHot: {
    borderColor: '#89b4fa', background: 'rgba(137,180,250,0.08)', color: '#89b4fa',
  },
  placeholderIcon: { fontSize: 26, opacity: 0.6 },
  placeholderText: { fontSize: 12, textAlign: 'center', padding: '0 12px' },
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
