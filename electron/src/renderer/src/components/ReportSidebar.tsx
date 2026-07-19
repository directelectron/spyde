/**
 * ReportSidebar.tsx — the Report Builder dock (third in-flow flex child in the
 * App body, after PlotControlDock).
 *
 * Composes markdown + embedded figures + split blocks (text beside a figure).
 * The COMPACT top bar (Wave B redesign) carries only: the document title
 * (click-to-edit), a dirty dot, a type badge (Report / Presentation), a single
 * "File ▾" menu (New Report / New Presentation / New from guide ▸ / Open / Save /
 * Save As Template / Export ▸) and a Present ▶ button shown ONLY for a
 * presentation. Everything else (Aa text-size, Rich/Raw, Capture, Paste) was
 * removed to de-clutter — Ctrl+V still pastes an image.
 *
 * The body is a scrollable cell list (text / figure / image / SPLIT) with
 * trailing "+ Add text cell" / "+ Add split block" / "+ Add image" buttons; the
 * whole body is a drop target (ConsoleBar model) accepting figure / window
 * pills → report_add_figure at the drop position.
 *
 * Left-edge resize uses the SubWindow Pointer-Capture pattern (NOT react-rnd).
 */
import React, { useEffect, useRef, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { FIGURE_DRAG_MIME, WINDOW_DRAG_MIME } from '../kernel/dnd'
import { reportFromGuide } from '../kernel/reportFromGuide'
import { ReportCell } from './ReportCell'
import { ReportFigureCell } from './ReportFigureCell'
import { ReportImageCell } from './ReportImageCell'
import { ReportSplitCell } from './ReportSplitCell'
import { GUIDES } from '@guides/index'

const MIN_W = 300
const MAX_W = 800
const DEFAULT_W = 420

const DROP_MIMES = [FIGURE_DRAG_MIME, WINDOW_DRAG_MIME]

/** The figure payload of a pill drop: the source window id plus — when the
 *  FIGURE_DRAG_MIME payload carries them — the dragged window's shown figure
 *  id and its view tag (view:'3d' while the 3-D IPF explorer was up), which
 *  report_add_figure branches on to snapshot the 3-D scene. */
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

// The image file extensions a PHOTO cell may carry (must mirror the backend's
// IMAGE_EXTS). Anything else the browser hands us is normalised to png (the
// backend defaults unknown exts to png too).
const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'gif', 'webp'] as const

/** Map an image file's MIME / name to one of IMAGE_EXTS. */
function imageExtOf(file: File): string {
  const fromType = (file.type.split('/')[1] || '').toLowerCase()
  if ((IMAGE_EXTS as readonly string[]).includes(fromType)) return fromType
  const fromName = (file.name.split('.').pop() || '').toLowerCase()
  if ((IMAGE_EXTS as readonly string[]).includes(fromName)) return fromName
  return 'png'
}

/** Read a File/Blob as a data URL. Rejects on read error. */
function readFileAsDataURL(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => resolve(String(fr.result || ''))
    fr.onerror = () => reject(fr.error)
    fr.readAsDataURL(file)
  })
}

/** True when a DataTransfer carries at least one image FILE (drop-a-photo path,
 *  distinct from a figure/window pill drop). */
function hasImageFiles(dt: DataTransfer): boolean {
  if (dt.files && dt.files.length) {
    for (const f of Array.from(dt.files)) {
      if (f.type.startsWith('image/')) return true
    }
  }
  // During dragover the file list isn't readable yet — fall back to the items
  // kind/type (Files with an image type).
  if (dt.items && dt.items.length) {
    for (const it of Array.from(dt.items)) {
      if (it.kind === 'file' && it.type.startsWith('image/')) return true
    }
  }
  return false
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
  // The document TYPE ('report' | 'presentation'), absent on an older backend →
  // 'report'. Drives the type badge, the Present button, and the Slides export.
  const docType = (report?.type === 'presentation') ? 'presentation' : 'report'
  const isPresentation = docType === 'presentation'

  const [width, setWidth] = useState(DEFAULT_W)
  const [titleEditing, setTitleEditing] = useState(false)
  const [titleDraft, setTitleDraft] = useState('')
  // Which pending New (report | presentation) the dirty-confirm bar is guarding
  // (null → not confirming). New over a dirty document routes through it.
  const [confirmNew, setConfirmNew] = useState<'report' | 'presentation' | null>(null)
  // The compact "File ▾" menu open state, and its nested Export / New-from-guide
  // submenu flags (only one submenu open at a time).
  const [menuOpen, setMenuOpen] = useState(false)
  const [guideSubOpen, setGuideSubOpen] = useState(false)
  const [exportSubOpen, setExportSubOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  // A transient export success/failure note shown under the header.
  const [exportNote, setExportNote] = useState<{ ok: boolean; text: string } | null>(null)
  // True while ANY export (HTML/Markdown/PDF) is in flight — disables the
  // Export menu items so a second export can't be triggered mid-flight
  // (which would otherwise cross-wire the `spyde:report_exported` match).
  // Always cleared on success, failure, AND timeout (finally-style).
  const [exporting, setExporting] = useState(false)
  const exportNoteTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
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
    slide_break?: boolean | null
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
      ...(vxChoice.slide_break != null ? { slide_break: vxChoice.slide_break } : {}),
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
  // New = a TYPE choice (report | presentation). Over a dirty document it routes
  // through the confirm bar, remembering WHICH type was requested.
  const doNew = (type: 'report' | 'presentation') => {
    setMenuOpen(false)
    if (report?.dirty) { setConfirmNew(type); return }
    sendAction('report_new', { type })
  }
  const confirmNewNow = () => {
    const type = confirmNew ?? 'report'
    setConfirmNew(null)
    sendAction('report_new', { type })
  }

  const doOpen = async () => {
    setMenuOpen(false)
    const path = await window.electron.reportOpenDialog()
    if (path) sendAction('report_open', { path })
  }

  const doSaveAsTemplate = async () => {
    setMenuOpen(false)
    const path = await window.electron.reportSaveDialog(
      (report?.title || 'template').replace(/[^\w.-]+/g, '_'),
    )
    if (path) sendAction('report_save_as_template', { path })
  }

  const doSave = async () => {
    setMenuOpen(false)
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
    closeMenus()
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
    closeMenus()
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
    closeMenus()
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

  const exportSlides = () => runExport(async () => {
    closeMenus()
    if (!canExport) return
    const path = await window.electron.reportExportDialog('html', `${titleSlug}-slides.html`)
    if (!path) return
    const token = crypto.randomUUID()
    const done = await awaitExport(
      (d) => d.token === token || d.path === path,
      () => sendAction('report_export_html', { mode: 'slides', path, token }),
    )
    if (done) showNote(true, `Exported ✓ ${basename(done)}`)
    else showNote(false, 'Export timed out')
  })

  // ── Present mode (Phase 6) ────────────────────────────────────────────────
  // Open the full-screen slide deck. PresentGate (inside the provider) listens
  // for this event and mounts PresentMode; the report is the source of slides.
  // Only offered for a PRESENTATION (a scrolling report has no slides to run).
  const doPresent = () => {
    window.dispatchEvent(new CustomEvent('spyde:report_present', { detail: { resume: false } }))
  }

  // Close the File menu (and its submenus) — shared by every menu action.
  const closeMenus = () => {
    setMenuOpen(false)
    setGuideSubOpen(false)
    setExportSubOpen(false)
  }

  // Seed a fresh PRESENTATION from a guide (each step → a slide). Picking a guide
  // dispatches report_new {type:'presentation'} + a report_add_cell per step.
  const newFromGuide = (guideId: string) => {
    closeMenus()
    const g = GUIDES.find(x => x.id === guideId)
    if (g) reportFromGuide(g, sendAction)
  }

  // Close the File menu on outside click / Escape.
  useEffect(() => {
    if (!menuOpen) return
    const onDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) closeMenus()
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeMenus() }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [menuOpen])

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

  // A split block: ONE atomic cell — a text side + an EMPTY figure side (a drop
  // zone the user fills by dropping a figure window onto it). Works in both a
  // report and a presentation.
  const addSplitCell = () =>
    sendAction('report_add_split_cell', {})

  // ── Add an image (photo) cell — shared by drop / paste / browse ────────────
  const fileInputRef = useRef<HTMLInputElement>(null)
  const addImageFile = async (file: Blob, ext: string, index?: number) => {
    try {
      const dataUrl = await readFileAsDataURL(file)
      if (!dataUrl) return
      sendAction('report_add_image_cell', {
        image_b64: dataUrl, image_ext: ext,
        ...(index !== undefined ? { index } : {}),
      })
    } catch { /* unreadable file — silently ignore */ }
  }
  // Browse: the hidden <input type=file> picked a file.
  const onBrowsePicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) await addImageFile(file, imageExtOf(file))
    // Reset so re-picking the SAME file fires change again.
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // Paste from clipboard: scoped to the sidebar (a focused paste). Reads the
  // first image/* clipboard item and adds it — the killer flow (paste a
  // screenshot). Only fires when the report body has focus / hover so a paste
  // into a text cell's editor is NOT hijacked.
  const dockRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (report == null) return
    const onPaste = (e: ClipboardEvent) => {
      // Don't steal a paste aimed at a text input / textarea (caption/markdown
      // editing) — only handle a paste landing on the report chrome itself.
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' ||
          target.isContentEditable)) {
        return
      }
      // Only act when the paste is within the report dock.
      if (!dockRef.current?.contains(target as Node) &&
          document.activeElement && !dockRef.current?.contains(document.activeElement)) {
        // Fall through only if the sidebar is the active region; otherwise ignore.
        if (!dockRef.current?.matches(':hover')) return
      }
      const items = e.clipboardData?.items
      if (!items) return
      for (const it of Array.from(items)) {
        if (it.kind === 'file' && it.type.startsWith('image/')) {
          const blob = it.getAsFile()
          if (blob) {
            e.preventDefault()
            const ext = (it.type.split('/')[1] || 'png').toLowerCase()
            void addImageFile(blob,
              (IMAGE_EXTS as readonly string[]).includes(ext) ? ext : 'png')
          }
          return
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report == null])

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
    // Accept a figure/window pill OR an image FILE being dragged in from the OS.
    const isPill = DROP_MIMES.some(m => e.dataTransfer.types.includes(m))
    const isImageFile = e.dataTransfer.types.includes('Files') &&
      hasImageFiles(e.dataTransfer)
    if (!isPill && !isImageFile) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    setDropIndex(computeDropIndex(e.clientY))
  }
  const onBodyDrop = (e: React.DragEvent) => {
    // An image FILE drop → add a photo cell (branch FIRST so it never collides
    // with the pill path). Falls through to the pill path otherwise.
    if (hasImageFiles(e.dataTransfer)) {
      e.preventDefault()
      const idx = computeDropIndex(e.clientY)
      setDropIndex(null)
      const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
      // Insert each dropped image in order at the drop point.
      files.forEach((f, k) => { void addImageFile(f, imageExtOf(f), idx + k) })
      return
    }
    if (!DROP_MIMES.some(m => e.dataTransfer.types.includes(m))) return
    e.preventDefault()
    const idx = computeDropIndex(e.clientY)
    setDropIndex(null)
    const src = figurePayloadFromDrop(e.dataTransfer)
    if (src != null) {
      sendAction('report_add_figure', {
        source_window_id: src.windowId, index: idx,
        ...(src.view !== undefined ? { view: src.view } : {}),
        ...(src.figId !== undefined ? { fig_id: src.figId } : {}),
      })
    }
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

  // The compact File menu — the single collapse point for New / Open / Save /
  // Export. Shared by the empty-state and open-report chrome so both offer the
  // same New Report / New Presentation / New from guide / Open entries.
  const fileMenu = (
    <div ref={menuRef} style={styles.menuWrap}>
      <button
        data-testid="report-menu-toggle"
        style={styles.hdrBtn}
        title="File menu"
        aria-haspopup="menu"
        aria-expanded={menuOpen}
        onClick={() => { setMenuOpen(v => !v); setGuideSubOpen(false); setExportSubOpen(false) }}
      >File ▾</button>
      {menuOpen && (
        <div style={styles.menu} data-testid="report-menu" role="menu">
          <MenuItem testid="menu-new-report" label="New Report"
            onClick={() => doNew('report')} />
          <MenuItem testid="menu-new-presentation" label="New Presentation"
            onClick={() => doNew('presentation')} />
          {/* New from guide ▸ — seeds a PRESENTATION, one slide per step. Click
              expands the nested list INLINE (no hover submenu — robust + never
              clips off-screen). */}
          <MenuItem testid="menu-new-from-guide"
            label={guideSubOpen ? 'New from guide ▾' : 'New from guide ▸'}
            onClick={() => { setGuideSubOpen(v => !v); setExportSubOpen(false) }} />
          {guideSubOpen && (
            <div style={styles.submenuInline} data-testid="menu-guide-submenu" role="menu">
              {GUIDES.map(g => (
                <MenuItem key={g.id} testid={`from-guide-${g.id}`} label={g.title}
                  onClick={() => newFromGuide(g.id)} />
              ))}
            </div>
          )}
          <div style={styles.menuSep} />
          <MenuItem testid="report-open" label="Open…" onClick={doOpen} />
          {report != null && (
            <>
              <MenuItem testid="report-save" label="Save" onClick={doSave} />
              <MenuItem testid="report-save-template" label="Save As Template"
                onClick={doSaveAsTemplate} />
              <div style={styles.menuSep} />
              {/* Export ▸ — expands INLINE on click. Slides deck is
                  presentation-only. */}
              <MenuItem testid="report-export-toggle"
                disabled={!canExport || exporting}
                label={exporting ? 'Exporting… ▸' : (exportSubOpen ? 'Export ▾' : 'Export ▸')}
                onClick={() => { if (canExport && !exporting) { setExportSubOpen(v => !v); setGuideSubOpen(false) } }} />
              {exportSubOpen && canExport && (
                <div style={styles.submenuInline} data-testid="report-export-menu" role="menu">
                  <MenuItem testid="export-html-interactive" label="Interactive HTML"
                    disabled={exporting} onClick={() => exportHtml('interactive')} />
                  <MenuItem testid="export-html-static" label="Static HTML"
                    disabled={exporting} onClick={() => exportHtml('static')} />
                  {isPresentation && (
                    <MenuItem testid="export-slides" label="Slides deck (HTML)"
                      disabled={exporting} onClick={exportSlides} />
                  )}
                  <MenuItem testid="export-pdf" label="PDF"
                    disabled={exporting} onClick={exportPdf} />
                  <MenuItem testid="export-md-folder" label="Markdown folder"
                    disabled={exporting} onClick={exportMarkdownFolder} />
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )

  if (report == null) {
    return (
      <div style={{ ...styles.dock, width }} data-testid="report-sidebar">
        <ResizeHandle onDown={onResizeDown} onMove={onResizeMove} onUp={onResizeUp} />
        <div style={styles.header}>
          <span style={styles.headerTitle}>Report</span>
          <div style={{ flex: 1 }} />
          {fileMenu}
        </div>
        <div style={styles.emptyState} data-testid="report-empty">
          No document open. Open the <b>File ▾</b> menu to start a
          <b> New Report</b>, a <b>New Presentation</b>, seed a deck
          {' '}<b>from a guide</b>, or <b>Open</b> an existing .spyde-report.
        </div>
      </div>
    )
  }

  return (
    <div ref={dockRef} style={{ ...styles.dock, width }} data-testid="report-sidebar">
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
        {/* Type badge — Report / Presentation (replaces the old "tpl"-only slot). */}
        <span
          data-testid="report-type-badge"
          data-type={docType}
          style={isPresentation ? styles.typeBadgePresentation : styles.typeBadgeReport}
          title={isPresentation ? 'Presentation (slide deck)' : 'Report (scrolling article)'}
        >{isPresentation ? 'Presentation' : 'Report'}</span>
        <div style={{ flex: 1 }} />
        {/* Present ▶ — presentations only (a scrolling report has no slides). */}
        {isPresentation && (
          <button
            data-testid="report-present"
            style={canExport ? styles.hdrBtnPrimary : styles.hdrBtnDisabled}
            title={canExport ? 'Present as slides' : 'Add a slide to present'}
            disabled={!canExport}
            onClick={doPresent}
          >Present ▶</button>
        )}
        {fileMenu}
      </div>

      {/* Transient export success/failure note. */}
      {exportNote && (
        <div
          data-testid="report-export-note"
          style={{ ...styles.exportNote, ...(exportNote.ok ? styles.exportNoteOk : styles.exportNoteErr) }}
        >{exportNote.text}</div>
      )}

      {/* Inline confirm for New over a dirty document. */}
      {confirmNew && (
        <div style={styles.confirmBar} data-testid="report-confirm-new">
          <span style={styles.confirmText}>
            Discard unsaved changes and start a new {confirmNew}?
          </span>
          <div style={{ flex: 1 }} />
          <button style={styles.confirmDiscard} onClick={confirmNewNow} data-testid="report-confirm-discard">Discard</button>
          <button style={styles.hdrBtn} onClick={() => setConfirmNew(null)} data-testid="report-confirm-cancel">Cancel</button>
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

      {/* Body: scrollable cell list + trailing add buttons. */}
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
            Drag a figure window here, or add a text / split cell below.
          </div>
        )}

        {cells.map((cell, i) => {
          return (
          <div key={cell.id} data-report-cell="1" style={{ position: 'relative' }}>
            {dropIndex === i && <div style={styles.insertLine} data-testid={`report-insert-${i}`} />}
            {cell.cell_type === 'figure'
              ? <ReportFigureCell
                  cell={cell}
                  index={i}
                  onRemove={() => sendAction('report_remove_cell', { cell_id: cell.id })}
                  dragProps={makeDragProps(cell.id, i)}
                  reorderActive={dragCell != null}
                />
              : cell.cell_type === 'image'
              ? <ReportImageCell
                  cell={cell}
                  index={i}
                  onRemove={() => sendAction('report_remove_cell', { cell_id: cell.id })}
                  dragProps={makeDragProps(cell.id, i)}
                />
              : cell.cell_type === 'split'
              ? <ReportSplitCell
                  cell={cell}
                  index={i}
                  onRemove={() => sendAction('report_remove_cell', { cell_id: cell.id })}
                  dragProps={makeDragProps(cell.id, i)}
                  reorderActive={dragCell != null}
                />
              : <ReportCell
                  cell={cell}
                  index={i}
                  onUpdate={(source, html) => sendAction('report_update_cell', { cell_id: cell.id, source, html })}
                  onRemove={() => sendAction('report_remove_cell', { cell_id: cell.id })}
                  dragProps={makeDragProps(cell.id, i)}
                />}
          </div>
          )
        })}
        {/* Trailing insert indicator (drop AFTER the last cell). */}
        {dropIndex === cells.length && cells.length > 0 && (
          <div style={styles.insertLine} data-testid={`report-insert-${cells.length}`} />
        )}

        <div style={styles.addRow}>
          <button
            data-testid="report-add-text"
            style={styles.addBtn}
            onClick={addTextCell}
          >+ Add text cell</button>
          <button
            data-testid="report-add-split"
            style={styles.addBtn}
            title="Add a split block — text on one side, a figure/photo on the other"
            onClick={addSplitCell}
          >+ Add split block</button>
          <button
            data-testid="report-add-image"
            style={styles.addBtn}
            title="Add a photo (or drop / paste one anywhere in this panel)"
            onClick={() => fileInputRef.current?.click()}
          >+ Add image</button>
        </div>
        <input
          ref={fileInputRef}
          data-testid="report-image-input"
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={onBrowsePicked}
        />
      </div>
    </div>
  )
}

// One File-menu / submenu row (hover-highlight, dock-palette style).
function MenuItem({ testid, label, onClick, disabled }: {
  testid: string; label: string; onClick: () => void; disabled?: boolean
}) {
  return (
    <button
      data-testid={testid}
      role="menuitem"
      disabled={disabled}
      style={disabled ? { ...styles.menuItem, color: '#585b70', cursor: 'default' } : styles.menuItem}
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
  typeBadgeReport: {
    fontSize: 9, fontWeight: 700, letterSpacing: 0.3, color: '#a6adc8',
    background: '#313244', border: '1px solid #45475a',
    borderRadius: 4, padding: '1px 6px', whiteSpace: 'nowrap',
  },
  typeBadgePresentation: {
    fontSize: 9, fontWeight: 700, letterSpacing: 0.3, color: '#11111b',
    background: '#89b4fa', border: '1px solid #89b4fa',
    borderRadius: 4, padding: '1px 6px', whiteSpace: 'nowrap',
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
  // Compact "File ▾" menu (single collapse point for New/Open/Save/Export).
  menuWrap: { position: 'relative', display: 'inline-flex' },
  menu: {
    position: 'absolute', top: '100%', right: 0, marginTop: 4, zIndex: 20,
    minWidth: 172, background: '#1e1e2e', border: '1px solid #45475a',
    borderRadius: 6, padding: 4, display: 'flex', flexDirection: 'column', gap: 1,
    boxShadow: '0 6px 22px rgba(0,0,0,0.5)',
  },
  menuItem: {
    background: 'none', border: 'none', color: '#cdd6f4', cursor: 'pointer',
    textAlign: 'left', padding: '5px 8px', fontSize: 11.5, borderRadius: 4,
    width: '100%', whiteSpace: 'nowrap',
  },
  menuSep: { height: 1, background: '#313244', margin: '3px 2px' },
  // A nested submenu expanded INLINE below its parent row (indented, subtle
  // left rule) — no absolute positioning, so it never clips off-screen.
  submenuInline: {
    display: 'flex', flexDirection: 'column', gap: 1,
    marginLeft: 8, paddingLeft: 4, borderLeft: '2px solid #313244',
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
  addRow: {
    display: 'flex', gap: 8, marginTop: 8,
  },
  addBtn: {
    flex: 1, minWidth: 0,
    background: '#1e1e2e', color: '#a6adc8', border: '1px dashed #45475a',
    borderRadius: 6, padding: '7px', fontSize: 12, cursor: 'pointer',
  },
}
