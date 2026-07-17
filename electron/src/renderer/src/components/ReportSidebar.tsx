/**
 * ReportSidebar.tsx — the Report Builder dock (third in-flow flex child in the
 * App body, after PlotControlDock).
 *
 * Composes markdown + embedded figures. Header carries the report title
 * (click-to-edit), a dirty dot, New/Open/Save buttons and a rendered↔raw
 * toggle. The body is a scrollable cell list with a trailing "+ Add text cell"
 * button; the whole body is a drop target (ConsoleBar model) accepting figure /
 * window pills → report_add_figure at the drop position.
 *
 * Left-edge resize uses the SubWindow Pointer-Capture pattern (NOT react-rnd).
 */
import React, { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { FIGURE_DRAG_MIME, WINDOW_DRAG_MIME } from '../kernel/dnd'
import { reportClipboard } from '../kernel/reportClipboard'
import { ReportCell } from './ReportCell'
import { ReportFigureCell } from './ReportFigureCell'

const MIN_W = 300
const MAX_W = 800
const DEFAULT_W = 420

const DROP_MIMES = [FIGURE_DRAG_MIME, WINDOW_DRAG_MIME]

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

export function ReportSidebar() {
  const { state, sendAction } = useSpyDE()
  // A closed report is surfaced by the backend as `open:false` (NOT by dropping
  // state.report to null — `report_close` still emits a report_state so the
  // renderer clears its cells). Treat both "no state yet" and "open:false" as
  // "no report open" so closing returns to the empty New/Open chrome instead of
  // an open-but-empty body (dangling Save/dirty affordances on a closed report).
  const report = state.report && state.report.open ? state.report : null
  const cells = report?.cells ?? []

  const [width, setWidth] = useState(DEFAULT_W)
  const [rawMode, setRawMode] = useState(false)
  const [titleEditing, setTitleEditing] = useState(false)
  const [titleDraft, setTitleDraft] = useState('')
  const [confirmNew, setConfirmNew] = useState(false)
  // Export dropdown open state + a transient success/failure note in the header.
  const [exportMenuOpen, setExportMenuOpen] = useState(false)
  const [exportNote, setExportNote] = useState<{ ok: boolean; text: string } | null>(null)
  // True while ANY export (HTML/Markdown/PDF) is in flight — disables the
  // Export menu items so a second export can't be triggered mid-flight
  // (which would otherwise cross-wire the `spyde:report_exported` match).
  // Always cleared on success, failure, AND timeout (finally-style).
  const [exporting, setExporting] = useState(false)
  const exportNoteTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const exportMenuRef = useRef<HTMLDivElement>(null)
  // Body drop state: whether a compatible pill is over the body, and the cell
  // index it would insert BEFORE (the between-cell indicator line).
  const [dropIndex, setDropIndex] = useState<number | null>(null)
  // Cell reorder (HTML5 DnD within the list).
  const dragCellId = useRef<string | null>(null)
  const [dragCell, setDragCell] = useState<string | null>(null)
  const [reorderBefore, setReorderBefore] = useState<string | null>(null)
  // A deferred vectors-figure drop awaiting the embed choice (viewer vs image).
  const [vxChoice, setVxChoice] = useState<{
    source_window_id: number
    index?: number | null
    at_cell?: string | null
    caption?: string
    count?: number
  } | null>(null)

  const bodyRef = useRef<HTMLDivElement>(null)

  // The backend defers a drop whose source tree carries diffraction vectors
  // and asks how HTML exports should embed it (SpyDEContext re-broadcasts the
  // message as this CustomEvent). Picking re-sends the original drop payload
  // with the choice; dismissing just drops it.
  useEffect(() => {
    const onChoice = (e: Event) => {
      setVxChoice((e as CustomEvent).detail)
    }
    window.addEventListener('spyde:report_vectors_choice', onChoice)
    return () => window.removeEventListener('spyde:report_vectors_choice', onChoice)
  }, [])
  const pickVectorsMode = (mode: 'viewer' | 'image') => {
    if (!vxChoice) return
    sendAction('report_add_figure', {
      source_window_id: vxChoice.source_window_id,
      index: vxChoice.index ?? undefined,
      at_cell: vxChoice.at_cell ?? undefined,
      caption: vxChoice.caption ?? '',
      vectors_mode: mode,
    })
    setVxChoice(null)
  }

  // ── Left-edge resize (Pointer-Capture, per SubWindow) ─────────────────────
  const resizeGesture = useRef<{ px: number; w: number } | null>(null)
  const onResizeDown = (e: React.PointerEvent) => {
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId) } catch { /* */ }
    resizeGesture.current = { px: e.clientX, w: width }
  }
  const onResizeMove = (e: React.PointerEvent) => {
    const g = resizeGesture.current
    if (!g) return
    // Dragging the LEFT edge left widens the dock (its right edge is fixed).
    const next = g.w + (g.px - e.clientX)
    setWidth(Math.min(MAX_W, Math.max(MIN_W, next)))
  }
  const onResizeUp = (e: React.PointerEvent) => {
    if (!resizeGesture.current) return
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId) } catch { /* */ }
    resizeGesture.current = null
  }

  // ── Header actions ────────────────────────────────────────────────────────
  const doNew = () => {
    if (report?.dirty) { setConfirmNew(true); return }
    sendAction('report_new', {})
  }
  const confirmNewNow = () => { setConfirmNew(false); sendAction('report_new', {}) }

  const doOpen = async () => {
    const path = await window.electron.reportOpenDialog()
    if (path) sendAction('report_open', { path })
  }

  const doSave = async () => {
    if (report?.path) {
      sendAction('report_save', {})
    } else {
      const path = await window.electron.reportSaveDialog(
        (report?.title || 'report').replace(/[^\w.-]+/g, '_'),
      )
      if (path) sendAction('report_save', { path })
    }
  }

  // ── Export ────────────────────────────────────────────────────────────────
  const canExport = report != null && cells.length > 0

  const basename = (p: string) =>
    p.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || p
  const titleSlug = (report?.title || 'report').replace(/[^\w.-]+/g, '_')

  // Show a transient note in the header area; auto-clears after ~4s.
  const showNote = (ok: boolean, text: string) => {
    if (exportNoteTimer.current) clearTimeout(exportNoteTimer.current)
    setExportNote({ ok, text })
    exportNoteTimer.current = setTimeout(() => setExportNote(null), 4000)
  }

  // Attach a `spyde:report_exported` listener FIRST, then run `trigger()` (which
  // sends the action). Resolves the matched export path via `match(detail)`, or
  // null on ~15s timeout. Listening before triggering closes the (theoretical)
  // race where a reply arrives before the listener is attached.
  const awaitExport = (
    match: (d: { kind?: string; path?: string; token?: string | null }) => boolean,
    trigger: () => void,
    timeoutMs = 15000,
  ): Promise<string | null> =>
    new Promise((resolve) => {
      let done = false
      const finish = (v: string | null) => {
        if (done) return
        done = true
        window.removeEventListener('spyde:report_exported', onEvt)
        clearTimeout(timer)
        resolve(v)
      }
      const onEvt = (ev: Event) => {
        const d = (ev as CustomEvent).detail as { kind?: string; path?: string; token?: string | null }
        if (match(d)) finish(d.path ?? null)
      }
      window.addEventListener('spyde:report_exported', onEvt)
      const timer = setTimeout(() => finish(null), timeoutMs)
      trigger()
    })

  // Run one export "leg", guarding `exporting` around it (set true before, always
  // cleared after — success, failure, or timeout) so the Export menu can't be
  // used to fire a second overlapping export while one is in flight.
  const runExport = async (body: () => Promise<void>) => {
    setExporting(true)
    try {
      await body()
    } finally {
      setExporting(false)
    }
  }

  const exportHtml = (mode: 'static' | 'interactive') => runExport(async () => {
    setExportMenuOpen(false)
    if (!canExport) return
    const path = await window.electron.reportExportDialog('html', `${titleSlug}.html`)
    if (!path) return
    // A fresh token per export correlates THIS export's reply with THIS
    // trigger — two exports in flight at once (or a stray retry) can't match
    // each other's `report_exported` echo. Fall back to a `path` match too
    // (belt-and-braces) in case the reply predates the token contract.
    const token = crypto.randomUUID()
    const done = await awaitExport(
      (d) => d.token === token || d.path === path,
      () => sendAction('report_export_html', { mode, path, token }),
    )
    if (done) showNote(true, `Exported ✓ ${basename(done)}`)
    else showNote(false, 'Export timed out')
  })

  const exportMarkdownFolder = () => runExport(async () => {
    setExportMenuOpen(false)
    if (!canExport) return
    const path = await window.electron.reportExportDialog('folder')
    if (!path) return
    const token = crypto.randomUUID()
    const done = await awaitExport(
      (d) => d.token === token || d.path === path,
      () => sendAction('report_export_markdown', { path, token }),
    )
    if (done) showNote(true, `Exported ✓ ${basename(done)}`)
    else showNote(false, 'Export timed out')
  })

  const exportPdf = () => runExport(async () => {
    setExportMenuOpen(false)
    if (!canExport) return
    const pdfPath = await window.electron.reportExportDialog('pdf', `${titleSlug}.pdf`)
    if (!pdfPath) return
    // First leg: render a STATIC HTML into a temp file (backend generates the
    // path and emits report_exported with it — no `path` supplied). The token
    // is the only way to correlate the reply since the temp name is unknown
    // up front and a concurrent export could otherwise match the wrong event.
    const tmpToken = crypto.randomUUID()
    const tmpPath = await awaitExport(
      (d) => d.token === tmpToken && d.kind === 'html-static',
      () => sendAction('report_export_html', { mode: 'static', temp: true, token: tmpToken }),
    )
    if (!tmpPath) { showNote(false, 'PDF export timed out'); return }
    // Second leg: render that temp HTML to the chosen PDF path (printToPDF).
    const res = await window.electron.reportExportPdf(tmpPath, pdfPath)
    if (res?.ok) showNote(true, `Exported ✓ ${basename(pdfPath)}`)
    else showNote(false, 'PDF export failed')
  })

  // ── Paste (internal cell clipboard) ───────────────────────────────────────
  // Reactive enablement: the Paste button is enabled only while the clipboard
  // holds a cell. useSyncExternalStore subscribes to the module-scope store.
  const clipboardCell = useSyncExternalStore(
    reportClipboard.subscribe, reportClipboard.get, reportClipboard.get,
  )
  const doPaste = () => {
    const cell = reportClipboard.get()
    if (!cell) return
    sendAction('report_paste_cell', { cell })   // append (no index)
  }

  // Close the export menu on outside click / Escape.
  useEffect(() => {
    if (!exportMenuOpen) return
    const onDown = (e: MouseEvent) => {
      if (!exportMenuRef.current?.contains(e.target as Node)) setExportMenuOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setExportMenuOpen(false) }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [exportMenuOpen])

  // Clear the note timer on unmount (StrictMode-safe).
  useEffect(() => () => {
    if (exportNoteTimer.current) clearTimeout(exportNoteTimer.current)
  }, [])

  const commitTitle = () => {
    setTitleEditing(false)
    const t = titleDraft.trim()
    if (t && t !== (report?.title ?? '')) sendAction('report_set_title', { title: t })
  }

  const addTextCell = () =>
    sendAction('report_add_cell', { cell_type: 'markdown', source: '', html: '' })

  // ── Body drop (figure/window pill → report_add_figure at insertion index) ──
  // Insertion index = number of cells whose vertical midpoint is above the
  // cursor (the between-cell position). A drop directly on a placeholder cell is
  // handled by ReportFigureCell itself (it stops propagation).
  const computeDropIndex = (clientY: number): number => {
    const body = bodyRef.current
    if (!body) return cells.length
    const cellEls = Array.from(
      body.querySelectorAll<HTMLElement>('[data-report-cell="1"]'),
    )
    let idx = 0
    for (const el of cellEls) {
      const r = el.getBoundingClientRect()
      if (clientY > r.top + r.height / 2) idx++
      else break
    }
    return idx
  }
  const onBodyDragOver = (e: React.DragEvent) => {
    if (!DROP_MIMES.some(m => e.dataTransfer.types.includes(m))) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    setDropIndex(computeDropIndex(e.clientY))
  }
  const onBodyDrop = (e: React.DragEvent) => {
    if (!DROP_MIMES.some(m => e.dataTransfer.types.includes(m))) return
    e.preventDefault()
    const idx = computeDropIndex(e.clientY)
    setDropIndex(null)
    const src = sourceWindowIdFromDrop(e.dataTransfer)
    if (src != null) sendAction('report_add_figure', { source_window_id: src, index: idx })
  }
  const onBodyDragLeave = (e: React.DragEvent) => {
    // Only clear when actually leaving the body (not moving between children).
    if (e.currentTarget === e.target) setDropIndex(null)
  }

  // ── Cell reorder wiring (HTML5 DnD, markdown cells) ───────────────────────
  const makeDragProps = (cellId: string, index: number) => ({
    onDragStart: (e: React.DragEvent) => {
      dragCellId.current = cellId
      setDragCell(cellId)
      e.dataTransfer.effectAllowed = 'move'
      // A private marker so the body-drop handler ignores an in-list reorder.
      e.dataTransfer.setData('application/x-spyde-report-cell', cellId)
    },
    onDragOver: (e: React.DragEvent) => {
      if (!dragCellId.current) return
      e.preventDefault()
      e.stopPropagation()   // reorder, not a figure insert
      setReorderBefore(cellId)
    },
    onDrop: (e: React.DragEvent) => {
      if (!dragCellId.current) return
      e.preventDefault()
      e.stopPropagation()
      const moved = dragCellId.current
      dragCellId.current = null
      setDragCell(null)
      setReorderBefore(null)
      if (moved && moved !== cellId) {
        sendAction('report_move_cell', { cell_id: moved, index })
      }
    },
    onDragEnd: () => {
      dragCellId.current = null
      setDragCell(null)
      setReorderBefore(null)
    },
    dragging: dragCell === cellId,
    dropBefore: reorderBefore === cellId,
  })

  if (report == null) {
    return (
      <div style={{ ...styles.dock, width }} data-testid="report-sidebar">
        <ResizeHandle onDown={onResizeDown} onMove={onResizeMove} onUp={onResizeUp} />
        <div style={styles.header}>
          <span style={styles.headerTitle}>Report</span>
          <div style={{ flex: 1 }} />
          <button data-testid="report-new" style={styles.hdrBtn} title="New report" onClick={doNew}>New</button>
          <button data-testid="report-open" style={styles.hdrBtn} title="Open report" onClick={doOpen}>Open</button>
        </div>
        <div style={styles.emptyState} data-testid="report-empty">
          No report open. Click <b>New</b> to start, or <b>Open</b> an existing
          .spyde-report.
        </div>
      </div>
    )
  }

  return (
    <div style={{ ...styles.dock, width }} data-testid="report-sidebar">
      <ResizeHandle onDown={onResizeDown} onMove={onResizeMove} onUp={onResizeUp} />

      {/* Header */}
      <div style={styles.header}>
        {titleEditing ? (
          <input
            data-testid="report-title-input"
            autoFocus
            style={styles.titleInput}
            value={titleDraft}
            onChange={(e) => setTitleDraft(e.target.value)}
            onBlur={commitTitle}
            onKeyDown={(e) => {
              if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
              else if (e.key === 'Escape') setTitleEditing(false)
            }}
          />
        ) : (
          <span
            data-testid="report-title"
            style={styles.headerTitle}
            title="Click to rename"
            onClick={() => { setTitleDraft(report.title); setTitleEditing(true) }}
          >
            {report.title || 'Untitled report'}
          </span>
        )}
        {report.dirty && (
          <span data-testid="report-dirty" style={styles.dirtyDot} title="Unsaved changes" />
        )}
        {report.template && (
          <span style={styles.templateBadge} title="Template">tpl</span>
        )}
        <div style={{ flex: 1 }} />
        <button
          data-testid="report-raw-toggle"
          style={rawMode ? styles.hdrBtnActive : styles.hdrBtn}
          title={rawMode ? 'Show rendered' : 'Show raw markdown'}
          onClick={() => setRawMode(v => !v)}
        >{rawMode ? 'Raw' : 'Rich'}</button>
        <button data-testid="report-new" style={styles.hdrBtn} title="New report" onClick={doNew}>New</button>
        <button data-testid="report-open" style={styles.hdrBtn} title="Open report" onClick={doOpen}>Open</button>
        <button
          data-testid="report-paste"
          style={clipboardCell ? styles.hdrBtn : styles.hdrBtnDisabled}
          title={clipboardCell ? 'Paste copied cell' : 'Nothing copied'}
          disabled={!clipboardCell}
          onClick={doPaste}
        >Paste</button>
        <button data-testid="report-save" style={styles.hdrBtnPrimary} title="Save report" onClick={doSave}>Save</button>

        {/* Export dropdown (dock palette, closes on outside click / Escape). */}
        <div ref={exportMenuRef} style={styles.exportWrap}>
          <button
            data-testid="report-export-toggle"
            style={canExport && !exporting ? styles.hdrBtn : styles.hdrBtnDisabled}
            title={exporting ? 'Export in progress…' : canExport ? 'Export report' : 'Add a cell to export'}
            disabled={!canExport || exporting}
            onClick={() => setExportMenuOpen(v => !v)}
          >{exporting ? 'Exporting…' : 'Export ▾'}</button>
          {exportMenuOpen && canExport && (
            <div style={styles.exportMenu} data-testid="report-export-menu" role="menu">
              <ExportItem testid="export-html-interactive" label="Interactive HTML" disabled={exporting}
                onClick={() => exportHtml('interactive')} />
              <ExportItem testid="export-html-static" label="Static HTML" disabled={exporting}
                onClick={() => exportHtml('static')} />
              <ExportItem testid="export-pdf" label="PDF" disabled={exporting} onClick={exportPdf} />
              <ExportItem testid="export-md-folder" label="Markdown folder" disabled={exporting}
                onClick={exportMarkdownFolder} />
            </div>
          )}
        </div>
      </div>

      {/* Transient export success/failure note. */}
      {exportNote && (
        <div
          data-testid="report-export-note"
          style={{ ...styles.exportNote, ...(exportNote.ok ? styles.exportNoteOk : styles.exportNoteErr) }}
        >{exportNote.text}</div>
      )}

      {/* Inline confirm for New over a dirty report. */}
      {confirmNew && (
        <div style={styles.confirmBar} data-testid="report-confirm-new">
          <span style={styles.confirmText}>Discard unsaved changes?</span>
          <div style={{ flex: 1 }} />
          <button style={styles.confirmDiscard} onClick={confirmNewNow} data-testid="report-confirm-discard">Discard</button>
          <button style={styles.hdrBtn} onClick={() => setConfirmNew(false)} data-testid="report-confirm-cancel">Cancel</button>
        </div>
      )}

      {/* Deferred vectors-figure drop: embed choice. */}
      {vxChoice && (
        <div style={styles.vxChoiceBar} data-testid="report-vectors-choice">
          <span style={styles.confirmText}>
            This figure has diffraction vectors
            {vxChoice.count ? ` (${vxChoice.count.toLocaleString()})` : ''}.
            Embed as:
          </span>
          <div style={styles.vxChoiceBtns}>
            <button
              style={styles.hdrBtnActive}
              onClick={() => pickVectorsMode('viewer')}
              data-testid="report-vectors-viewer"
              title="Interactive explorer in HTML exports — drag a virtual detector in the page"
            >Interactive viewer</button>
            <button
              style={styles.hdrBtn}
              onClick={() => pickVectorsMode('image')}
              data-testid="report-vectors-image"
            >Just the image</button>
            <div style={{ flex: 1 }} />
            <button
              style={styles.hdrBtn}
              onClick={() => setVxChoice(null)}
              data-testid="report-vectors-cancel"
            >Cancel</button>
          </div>
        </div>
      )}

      {/* Body: scrollable cell list + trailing add button. */}
      <div
        ref={bodyRef}
        data-testid="report-body"
        style={styles.body}
        onDragOver={onBodyDragOver}
        onDrop={onBodyDrop}
        onDragLeave={onBodyDragLeave}
      >
        {cells.length === 0 && (
          <div style={styles.dropHint} data-testid="report-drop-hint">
            Drag a figure window here, or add a text cell below.
          </div>
        )}

        {cells.map((cell, i) => (
          <div key={cell.id} data-report-cell="1" style={{ position: 'relative' }}>
            {dropIndex === i && <div style={styles.insertLine} data-testid={`report-insert-${i}`} />}
            {cell.cell_type === 'figure'
              ? <ReportFigureCell
                  cell={cell}
                  index={i}
                  onRemove={() => sendAction('report_remove_cell', { cell_id: cell.id })}
                />
              : <ReportCell
                  cell={cell}
                  index={i}
                  rawMode={rawMode}
                  onUpdate={(source, html) => sendAction('report_update_cell', { cell_id: cell.id, source, html })}
                  onRemove={() => sendAction('report_remove_cell', { cell_id: cell.id })}
                  dragProps={makeDragProps(cell.id, i)}
                />}
          </div>
        ))}
        {/* Trailing insert indicator (drop AFTER the last cell). */}
        {dropIndex === cells.length && cells.length > 0 && (
          <div style={styles.insertLine} data-testid={`report-insert-${cells.length}`} />
        )}

        <button
          data-testid="report-add-text"
          style={styles.addBtn}
          onClick={addTextCell}
        >+ Add text cell</button>
      </div>
    </div>
  )
}

// One Export dropdown row (hover-highlight, dock-palette style).
function ExportItem({ testid, label, onClick, disabled }: {
  testid: string; label: string; onClick: () => void; disabled?: boolean
}) {
  return (
    <button
      data-testid={testid}
      role="menuitem"
      disabled={disabled}
      style={disabled ? { ...styles.exportItem, color: '#585b70', cursor: 'default' } : styles.exportItem}
      onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.background = '#313244' }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
      onClick={disabled ? undefined : onClick}
    >{label}</button>
  )
}

function ResizeHandle({ onDown, onMove, onUp }: {
  onDown: (e: React.PointerEvent) => void
  onMove: (e: React.PointerEvent) => void
  onUp: (e: React.PointerEvent) => void
}) {
  const [hover, setHover] = useState(false)
  return (
    <div
      data-testid="report-resize-handle"
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{ ...styles.resizeHandle, background: hover ? '#89b4fa' : 'transparent' }}
    />
  )
}

const styles: Record<string, React.CSSProperties> = {
  dock: {
    position: 'relative',
    flexShrink: 0,
    height: '100%',
    background: '#181825',
    borderLeft: '1px solid #313244',
    display: 'flex', flexDirection: 'column',
    color: '#cdd6f4',
  },
  resizeHandle: {
    position: 'absolute', left: -3, top: 0, bottom: 0, width: 6,
    cursor: 'ew-resize', zIndex: 5,
    transition: 'background 120ms ease',
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '8px 10px', borderBottom: '1px solid #313244',
    flexShrink: 0,
  },
  headerTitle: {
    fontSize: 13, fontWeight: 600, color: '#cdd6f4',
    cursor: 'text', overflow: 'hidden', textOverflow: 'ellipsis',
    whiteSpace: 'nowrap', maxWidth: 160,
  },
  titleInput: {
    background: '#11111b', border: '1px solid #89b4fa', borderRadius: 5,
    color: '#cdd6f4', fontSize: 13, fontWeight: 600, padding: '2px 6px',
    outline: 'none', maxWidth: 180,
  },
  dirtyDot: {
    width: 7, height: 7, borderRadius: '50%', background: '#fab387',
    flexShrink: 0, marginLeft: 1,
  },
  templateBadge: {
    fontSize: 9, fontWeight: 700, color: '#a6adc8',
    background: '#313244', borderRadius: 4, padding: '1px 4px',
  },
  hdrBtn: {
    background: '#1e1e2e', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 5, padding: '3px 8px', fontSize: 11, cursor: 'pointer',
  },
  hdrBtnActive: {
    background: '#89b4fa', color: '#11111b', border: '1px solid #89b4fa',
    borderRadius: 5, padding: '3px 8px', fontSize: 11, cursor: 'pointer', fontWeight: 600,
  },
  hdrBtnPrimary: {
    background: '#fab387', color: '#11111b', border: '1px solid #fab387',
    borderRadius: 5, padding: '3px 10px', fontSize: 11, cursor: 'pointer', fontWeight: 700,
  },
  hdrBtnDisabled: {
    background: '#1e1e2e', color: '#585b70', border: '1px solid #313244',
    borderRadius: 5, padding: '3px 8px', fontSize: 11, cursor: 'default',
  },
  exportWrap: { position: 'relative', display: 'inline-flex' },
  exportMenu: {
    position: 'absolute', top: '100%', right: 0, marginTop: 4, zIndex: 20,
    minWidth: 150, background: '#1e1e2e', border: '1px solid #45475a',
    borderRadius: 6, padding: 4, display: 'flex', flexDirection: 'column', gap: 2,
    boxShadow: '0 6px 22px rgba(0,0,0,0.5)',
  },
  exportItem: {
    background: 'none', border: 'none', color: '#cdd6f4', cursor: 'pointer',
    textAlign: 'left', padding: '5px 8px', fontSize: 11.5, borderRadius: 4,
  },
  exportNote: {
    padding: '4px 10px', fontSize: 11, fontWeight: 600,
    borderBottom: '1px solid #313244',
  },
  exportNoteOk: { color: '#a6e3a1', background: 'rgba(166,227,161,0.08)' },
  exportNoteErr: { color: '#f38ba8', background: 'rgba(243,139,168,0.08)' },
  confirmBar: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '6px 10px', background: 'rgba(250,179,135,0.08)',
    borderBottom: '1px solid #313244',
  },
  confirmText: { fontSize: 11.5, color: '#fab387' },
  vxChoiceBar: {
    display: 'flex', flexDirection: 'column', gap: 6,
    padding: '6px 10px', background: 'rgba(250,179,135,0.08)',
    borderBottom: '1px solid #313244',
  },
  vxChoiceBtns: { display: 'flex', alignItems: 'center', gap: 6 },
  confirmDiscard: {
    background: '#f38ba8', color: '#11111b', border: 'none',
    borderRadius: 5, padding: '3px 10px', fontSize: 11, cursor: 'pointer', fontWeight: 600,
  },
  body: {
    flex: 1, minHeight: 0, overflowY: 'auto',
    padding: '10px 10px 24px',
  },
  emptyState: {
    padding: 20, fontSize: 12.5, color: '#7f849c', lineHeight: 1.6,
  },
  dropHint: {
    padding: '20px 12px', fontSize: 12.5, color: '#585b70',
    textAlign: 'center', border: '2px dashed #313244', borderRadius: 8,
    marginBottom: 10,
  },
  insertLine: {
    height: 2, background: '#89b4fa', borderRadius: 1, margin: '2px 0',
  },
  addBtn: {
    display: 'block', width: '100%', marginTop: 8,
    background: '#1e1e2e', color: '#a6adc8', border: '1px dashed #45475a',
    borderRadius: 6, padding: '7px', fontSize: 12, cursor: 'pointer',
  },
}
