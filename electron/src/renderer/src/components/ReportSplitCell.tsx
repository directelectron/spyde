/**
 * ReportSplitCell.tsx — a SPLIT block in the Report sidebar (Wave B).
 *
 * ONE atomic cell rendered as TWO side-by-side panes: a TEXT pane (the cell's
 * `source` markdown, editable exactly like a ReportCell — double-click → a
 * monospace textarea, commit on blur / Ctrl-Enter) and a FIGURE pane. The pane
 * order follows `split_layout`:
 *   • 'text-left'  → text | figure  (the default)
 *   • 'text-right' → figure | text  (mirror)
 * A single layout-switch button on the block swaps the two sides
 * (report_set_split_layout).
 *
 * The FIGURE pane is:
 *   • a dashed DROP ZONE when `split_empty` — dropping a figure/window pill onto
 *     it fills the figure side via `report_add_figure {source_window_id,
 *     at_cell:<this cell id>}` (Wave A made report_add_figure fill a split cell's
 *     figure side when at_cell targets it, and it SKIPS the viewer-vs-image
 *     prompt for a split).
 *   • a LIVE figure embed (SeamlessFigureFrame, keyed by fig_id === cell.id) when
 *     the figure side is a snapshot spec — interactive, like a normal figure cell.
 *   • an <img> when the figure side is a photo (`image` data URL).
 *
 * Chrome (hover): the shared CellChrome trio (Copy / Duplicate / Delete) + a
 * reorder ⠿ handle + the layout switch. NO slide-toggle chrome (removed in the
 * Wave B de-clutter; slide roles are re-surfaced slide-natively in Wave C).
 */
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { renderMarkdown } from '../kernel/markdown'
import { reportClipboard } from '../kernel/reportClipboard'
import type { ReportCell } from '../kernel/protocol'
import { FIGURE_DRAG_MIME, WINDOW_DRAG_MIME } from '../kernel/dnd'
import { CellChrome } from './CellChrome'
import { SeamlessFigureFrame } from './ReportFigureCell'

const DROP_MIMES = [FIGURE_DRAG_MIME, WINDOW_DRAG_MIME]
const isComposeDrag = (dt: DataTransfer) => DROP_MIMES.some(m => dt.types.includes(m))

/** The figure payload of a pill drop: the source window id (+ the shown figure
 *  id / view tag when the FIGURE_DRAG_MIME payload carries them). */
interface DropFigurePayload { windowId: number; figId?: string; view?: string }
function figurePayloadFromDrop(dt: DataTransfer): DropFigurePayload | null {
  const fig = dt.getData(FIGURE_DRAG_MIME)
  if (fig) {
    try {
      const { windowId, figId, view } = JSON.parse(fig) as {
        windowId?: number; figId?: string; view?: string
      }
      if (typeof windowId === 'number') return { windowId, figId, view }
    } catch { /* malformed */ }
  }
  const win = dt.getData(WINDOW_DRAG_MIME)
  if (win) {
    const n = parseInt(win, 10)
    if (Number.isFinite(n)) return { windowId: n }
  }
  return null
}

interface Props {
  cell: ReportCell
  onRemove: () => void
  /** Own index in the cell list (Duplicate → insert at index+1). */
  index: number
  /** HTML5 DnD reorder wiring supplied by the parent list (ReportSidebar.
   *  makeDragProps). */
  dragProps: {
    onDragStart: (e: React.DragEvent) => void
    onDragOver: (e: React.DragEvent) => void
    onDrop: (e: React.DragEvent) => void
    onDragEnd: () => void
    dragging: boolean
    dropBefore: boolean
  }
  /** True while ANY cell reorder is in flight — mounts a transparent shield over
   *  the figure iframe so dragover/drop reach this cell (the out-of-process
   *  iframe swallows DnD otherwise). */
  reorderActive: boolean
}

export function ReportSplitCell({ cell, onRemove, index, dragProps, reorderActive }: Props) {
  const { state, iframeRefs, replayState, sendAction, dragKind } = useSpyDE()
  const [hover, setHover] = useState(false)
  const [editing, setEditing] = useState(false)
  const [figEditOpen, setFigEditOpen] = useState(false)
  const [draft, setDraft] = useState(cell.source ?? '')
  const [dropHover, setDropHover] = useState(false)
  const taRef = useRef<HTMLTextAreaElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  const LAYOUTS = ['text-left', 'text-right', 'text-top', 'text-bottom'] as const
  type Layout = typeof LAYOUTS[number]
  const layout: Layout = (LAYOUTS as readonly string[]).includes(cell.split_layout ?? '')
    ? (cell.split_layout as Layout) : 'text-left'
  const stacked = layout === 'text-top' || layout === 'text-bottom'
  const textFirst = layout === 'text-left' || layout === 'text-top'
  const [layoutMenu, setLayoutMenu] = useState(false)
  const empty = !!cell.split_empty
  // The figure side rides in the same fig_id/reportFigures plumbing as a figure
  // cell (keyed by the CELL id). A photo side ships `image` instead of a spec.
  const fig = state.reportFigures.get(cell.id)
  const hasImage = !empty && !cell.figure && !!cell.image
  const isLive = !empty && !!cell.figure && !!fig

  // Keep the text draft in sync when the backing source changes and we're not
  // actively editing (a live report_state update from elsewhere).
  useEffect(() => { if (!editing) setDraft(cell.source ?? '') }, [cell.source, editing])

  // Autosize the textarea to its content.
  useLayoutEffect(() => {
    const ta = taRef.current
    if (!ta || !editing) return
    ta.style.height = 'auto'
    ta.style.height = `${ta.scrollHeight}px`
  }, [draft, editing])

  // Figure-side actions — available only when the figure side is a live FIGURE
  // (not a photo, not an empty drop zone). The backend admits a split's figure
  // side into report_refresh_figure / repfig_set_edit_mode (it reuses the same
  // FigureSpec machinery), so these behave exactly like a figure cell's.
  const canEditFigure = isLive && !!cell.figure
  const refreshFigure = () => sendAction('report_refresh_figure', { cell_id: cell.id })
  const toggleFigEdit = () => {
    setFigEditOpen((v) => {
      const next = !v
      sendAction('repfig_set_edit_mode', { cell_id: cell.id, editing: next })
      return next
    })
  }

  const commitText = () => {
    setEditing(false)
    if (draft !== (cell.source ?? '')) {
      sendAction('report_update_cell', {
        cell_id: cell.id, source: draft, html: renderMarkdown(draft),
      })
    }
  }
  const revertText = () => { setDraft(cell.source ?? ''); setEditing(false) }

  const rendered = React.useMemo(() => renderMarkdown(cell.source ?? ''), [cell.source])
  const textEmpty = !(cell.source ?? '').trim()

  // ── Layout switch (a dropdown picker: 4 arrangements) ─────────────────────
  const setLayout = (l: Layout) => {
    setLayoutMenu(false)
    sendAction('report_set_split_layout', { cell_id: cell.id, layout: l })
  }
  const LAYOUT_OPTS: { value: Layout; label: string; glyph: string }[] = [
    { value: 'text-left', label: 'Text left', glyph: '◧' },
    { value: 'text-right', label: 'Text right', glyph: '◨' },
    { value: 'text-top', label: 'Text top', glyph: '⬒' },
    { value: 'text-bottom', label: 'Text bottom', glyph: '⬓' },
  ]

  // ── Figure-side drop (fill the empty figure side in place) ─────────────────
  const onFigDragOver = (e: React.DragEvent) => {
    if (!isComposeDrag(e.dataTransfer)) return
    e.preventDefault()
    e.stopPropagation()   // don't also trigger the sidebar-body insertion logic
    e.dataTransfer.dropEffect = 'copy'
    setDropHover(true)
  }
  const onFigDrop = (e: React.DragEvent) => {
    if (!isComposeDrag(e.dataTransfer)) return
    e.preventDefault()
    e.stopPropagation()
    setDropHover(false)
    const src = figurePayloadFromDrop(e.dataTransfer)
    if (src == null) return
    // Wave A: report_add_figure with at_cell targeting a SPLIT cell fills its
    // figure side in place (skips the viewer-vs-image prompt for a split).
    sendAction('report_add_figure', {
      source_window_id: src.windowId, at_cell: cell.id,
      ...(src.view !== undefined ? { view: src.view } : {}),
      ...(src.figId !== undefined ? { fig_id: src.figId } : {}),
    })
  }

  // ── Copy / Duplicate ──────────────────────────────────────────────────────
  // A split block's live figure side isn't round-trippable through the internal
  // clipboard (report_paste_cell has no split branch — Wave A), so Copy mirrors
  // the TEXT to the clipboard (as a markdown cell, so a cross-cell paste still
  // works) and Duplicate re-creates the split via the supported
  // report_add_split_cell verb (same text + layout, figure side re-empty for a
  // fresh drop).
  const doCopy = () => reportClipboard.set({
    cell_type: 'markdown', source: cell.source ?? '', html: rendered,
  })
  const doDuplicate = () =>
    sendAction('report_add_split_cell', {
      index: index + 1, source: cell.source ?? '',
      caption: cell.caption ?? '', layout,
    })

  // ── The two panes ─────────────────────────────────────────────────────────
  const textPane = (
    <div style={styles.pane} data-testid={`report-split-text-${cell.id}`}>
      {editing ? (
        <textarea
          ref={taRef}
          data-testid={`report-split-textarea-${cell.id}`}
          style={styles.textarea}
          value={draft}
          autoFocus
          spellCheck={false}
          placeholder="Write markdown…  ($x^2$ and $$…$$ render as math)"
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitText}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey || e.shiftKey)) {
              e.preventDefault(); (e.target as HTMLTextAreaElement).blur()
            } else if (e.key === 'Escape') {
              e.preventDefault(); revertText()
            }
          }}
        />
      ) : (
        <div
          data-testid={`report-split-rendered-${cell.id}`}
          className="spyde-md"
          onDoubleClick={() => { setDraft(cell.source ?? ''); setEditing(true) }}
          title="Double-click to edit"
          style={styles.rendered}
        >
          {textEmpty
            ? <span style={styles.emptyHint}>Empty text side — double-click to edit</span>
            : <span dangerouslySetInnerHTML={{ __html: rendered }} />}
        </div>
      )}
    </div>
  )

  const figurePane = (
    <div style={styles.pane} data-testid={`report-split-figure-${cell.id}`}>
      {empty ? (
        <div
          data-testid={`report-split-dropzone-${cell.id}`}
          onDragOver={onFigDragOver}
          onDragLeave={() => setDropHover(false)}
          onDrop={onFigDrop}
          style={{ ...styles.dropzone, ...(dropHover ? styles.dropzoneHot : {}) }}
        >
          <div style={styles.dropzoneIcon}>▤</div>
          <div style={styles.dropzoneText}>Drop a figure window here</div>
        </div>
      ) : isLive && fig ? (
        <div style={styles.figBox}>
          <SeamlessFigureFrame
            figId={fig.figId}
            filePath={fig.filePath}
            title={fig.title}
            iframeRefs={iframeRefs}
            replayState={replayState}
          />
          {/* Reorder shield: the out-of-process iframe swallows DnD, so during a
              cell reorder we cover it with a transparent layer (dragover/drop
              bubble to the cell root's dragProps). Mounted only during a drag. */}
          {(reorderActive || dragKind === 'window') && (
            <div
              data-testid={`report-split-shield-${cell.id}`}
              style={styles.shield}
              {...(dragKind === 'window' ? {
                onDragOver: onFigDragOver,
                onDragLeave: () => setDropHover(false),
                onDrop: onFigDrop,
              } : {})}
            />
          )}
        </div>
      ) : hasImage ? (
        <div style={styles.figBox}>
          <img src={cell.image} alt={cell.caption ?? ''} style={styles.img} />
        </div>
      ) : (
        <div style={styles.figBox}>
          <div style={styles.pending} data-testid={`report-split-pending-${cell.id}`}>rendering…</div>
        </div>
      )}
    </div>
  )

  return (
    <div
      ref={rootRef}
      data-testid={`report-splitcell-${cell.id}`}
      data-layout={layout}
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
      {hover && (
        <CellChrome
          cellId={cell.id}
          styles={{ chrome: styles.chrome, chromeBtn: styles.chromeBtn }}
          onCopy={doCopy}
          onDuplicate={doDuplicate}
          onDelete={onRemove}
          deleteTestid={`report-splitcell-delete-${cell.id}`}
          deleteTitle="Delete split block"
          leading={
            <span
              data-testid={`report-split-drag-${cell.id}`}
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
          trailing={
            <>
              {canEditFigure && (
                <button
                  data-testid={`report-split-edit-${cell.id}`}
                  style={figEditOpen ? styles.chromeBtnActive : styles.chromeBtn}
                  title={figEditOpen
                    ? 'Done editing the figure'
                    : 'Edit the figure side (annotations, layers, callouts)'}
                  onClick={toggleFigEdit}
                >✎</button>
              )}
              {canEditFigure && (
                <button
                  data-testid={`report-split-refresh-${cell.id}`}
                  style={styles.chromeBtn}
                  title="Refresh the figure side from the live data"
                  onClick={refreshFigure}
                >⟳</button>
              )}
              <div style={{ position: 'relative' }}>
                <button
                  data-testid={`report-split-layout-${cell.id}`}
                  style={styles.chromeBtn}
                  title="Split layout (text vs figure arrangement)"
                  aria-haspopup="menu"
                  aria-expanded={layoutMenu}
                  onClick={() => setLayoutMenu(v => !v)}
                >{LAYOUT_OPTS.find(o => o.value === layout)?.glyph ?? '◧'} ▾</button>
                {layoutMenu && (
                  <div style={styles.layoutMenu} role="menu"
                    data-testid={`report-split-layout-menu-${cell.id}`}>
                    {LAYOUT_OPTS.map(o => (
                      <button key={o.value} role="menuitemradio"
                        aria-checked={layout === o.value}
                        data-testid={`report-split-layout-${cell.id}-${o.value}`}
                        style={{ ...styles.layoutMenuItem, ...(layout === o.value ? styles.layoutMenuItemActive : {}) }}
                        onClick={() => setLayout(o.value)}>
                        <span style={styles.layoutGlyph}>{o.glyph}</span> {o.label}
                        {layout === o.value && <span style={{ marginLeft: 'auto' }}>✓</span>}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </>
          }
        />
      )}

      {/* The two panes, ordered + oriented by split_layout (side-by-side or
          stacked; text first or figure first). */}
      <div style={{ ...styles.panes,
        gridTemplateColumns: stacked ? '1fr' : '1fr 1fr',
        gridTemplateRows: stacked ? 'auto auto' : '1fr' }}>
        {textFirst ? <>{textPane}{figurePane}</> : <>{figurePane}{textPane}</>}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  cell: {
    position: 'relative',
    borderRadius: 6,
    padding: '6px',
    marginBottom: 6,
    border: '1px solid #313244',
    borderTop: '2px solid transparent',
  },
  cellDragging: { opacity: 0.4 },
  cellDropBefore: { borderTop: '2px solid #89b4fa' },
  panes: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, alignItems: 'stretch',
  },
  pane: { minWidth: 0, display: 'flex', flexDirection: 'column' },
  layoutMenu: {
    position: 'absolute', top: '100%', right: 0, marginTop: 4, zIndex: 20,
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6,
    padding: 4, minWidth: 130, boxShadow: '0 6px 18px rgba(0,0,0,0.5)',
    display: 'flex', flexDirection: 'column', gap: 2,
  },
  layoutMenuItem: {
    display: 'flex', alignItems: 'center', gap: 8, width: '100%',
    background: 'none', border: 'none', color: '#cdd6f4', cursor: 'pointer',
    fontSize: 12, padding: '5px 8px', borderRadius: 4, textAlign: 'left',
  },
  layoutMenuItemActive: { background: '#313244' },
  layoutGlyph: { fontSize: 15, width: 18, textAlign: 'center' },
  chrome: {
    position: 'absolute', top: 6, right: 6, zIndex: 4,
    display: 'flex', alignItems: 'center', gap: 2,
    background: 'rgba(24,24,37,0.96)', borderRadius: 8, padding: 3,
    border: '1px solid #313244', boxShadow: '0 3px 10px rgba(0,0,0,0.35)',
  },
  chromeBtn: {
    background: 'none', border: 'none', color: '#cdd6f4', cursor: 'pointer',
    fontSize: 15, lineHeight: 1, borderRadius: 6,
    width: 24, height: 24, display: 'inline-flex',
    alignItems: 'center', justifyContent: 'center', padding: 0,
    transition: 'background 100ms ease, color 100ms ease',
  },
  chromeBtnActive: {
    background: '#89b4fa', border: 'none', color: '#11111b',
    cursor: 'pointer', fontSize: 15, lineHeight: 1, borderRadius: 6,
    width: 24, height: 24, display: 'inline-flex',
    alignItems: 'center', justifyContent: 'center', padding: 0,
  },
  dragHandle: {
    cursor: 'grab', color: '#7f849c', fontSize: 15, userSelect: 'none',
    lineHeight: 1, display: 'inline-flex', alignItems: 'center',
    height: 24, padding: '0 3px',
  },
  rendered: {
    cursor: 'text', minHeight: 40, padding: '4px 4px', borderRadius: 4, flex: 1,
  },
  emptyHint: { color: '#585b70', fontSize: 12, fontStyle: 'italic' },
  textarea: {
    width: '100%', boxSizing: 'border-box', resize: 'none',
    background: '#11111b', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 5,
    padding: '6px 8px', fontSize: 12.5,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    lineHeight: 1.5, outline: 'none', overflow: 'hidden', minHeight: 60,
  },
  figBox: {
    position: 'relative', width: '100%', aspectRatio: '4 / 3',
    background: '#11111b', borderRadius: 6, border: '1px solid #313244',
    overflow: 'hidden',
  },
  img: { display: 'block', width: '100%', height: '100%', objectFit: 'contain' },
  pending: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: '100%', height: '100%', color: '#6c7086', fontSize: 11,
  },
  shield: {
    position: 'absolute', inset: 0, zIndex: 3, background: 'transparent',
  },
  dropzone: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 6, width: '100%', aspectRatio: '4 / 3',
    border: '2px dashed #45475a', borderRadius: 8, color: '#585b70',
    transition: 'border-color 120ms ease, background 120ms ease',
  },
  dropzoneHot: {
    border: '2px dashed #89b4fa', background: 'rgba(137,180,250,0.08)',
    color: '#89b4fa',
  },
  dropzoneIcon: { fontSize: 22, lineHeight: 1 },
  dropzoneText: { fontSize: 11, textAlign: 'center', padding: '0 8px' },
}
