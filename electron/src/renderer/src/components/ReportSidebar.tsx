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
import { SlideNotesEditor } from './SlideNotesEditor'
import { GUIDES } from '@guides/index'
import type { ReportCell as ReportCellType } from '../kernel/protocol'

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

/** One slide's worth of cells, carrying each cell's ORIGINAL flat index (so the
 *  slide-native list still drives the index-keyed drop/reorder machinery
 *  unchanged) plus the slide's start cell (index 0 of the group — where the
 *  per-slide attributes live). The renderer mirror of `ReportDoc.slides()`. */
interface SlideGroup {
  /** 0-based slide number (for the "Slide N" label). */
  n: number
  /** The slide's cells, each paired with its index in the flat `cells` array. */
  items: Array<{ cell: ReportCellType; index: number }>
  /** The slide's FIRST cell — where slide_kind / slide_style / notes are carried
   *  and to which the per-slide verbs are addressed. */
  first: ReportCellType
}

/** Group the flat cell list into slides by `slide_break`, mirroring
 *  `ReportDoc.slides()` (a break STARTS a new slide; the first cell always begins
 *  slide 0). Each cell keeps its ORIGINAL flat index. */
function groupSlides(cells: ReportCellType[]): SlideGroup[] {
  const groups: SlideGroup[] = []
  cells.forEach((cell, index) => {
    if (cell.slide_break && groups.length) {
      groups.push({ n: groups.length, items: [{ cell, index }], first: cell })
    } else if (!groups.length) {
      groups.push({ n: 0, items: [{ cell, index }], first: cell })
    } else {
      groups[groups.length - 1].items.push({ cell, index })
    }
  })
  return groups
}

// The three background presets a slide can carry (slide_style). '' == the
// standard dark stage ("Default"); the labels + swatch colours are shown in the
// per-slide header's Background picker.
const SLIDE_STYLES: Array<{ value: '' | 'plain' | 'accent'; label: string; swatch: string }> = [
  { value: '', label: 'Default', swatch: '#1e1e2e' },
  { value: 'plain', label: 'Plain', swatch: '#0e0e16' },
  { value: 'accent', label: 'Accent', swatch: '#89b4fa' },
]

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
  // Presentation slide-native chrome state:
  //  • which slides' Speaker-notes editors are expanded (keyed by the slide's
  //    first-cell id — the per-slide attribute owner),
  //  • which slide's Background picker popover is open (first-cell id or null),
  //  • whether the "+ Add slide" starter menu is open.
  const [openNotes, setOpenNotes] = useState<Set<string>>(new Set())
  const [bgPickerFor, setBgPickerFor] = useState<string | null>(null)
  const [addSlideMenu, setAddSlideMenu] = useState(false)
  const addSlideRef = useRef<HTMLDivElement>(null)
  // Whole-slide reorder DnD (drag a slide's grip onto another slide group).
  const dragSlideN = useRef<number | null>(null)
  const [dragSlide, setDragSlide] = useState<number | null>(null)
  const [slideDropOn, setSlideDropOn] = useState<number | null>(null)
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

  // Close the "+ Add slide" starter menu on outside click / Escape.
  useEffect(() => {
    if (!addSlideMenu) return
    const onDown = (e: MouseEvent) => {
      if (!addSlideRef.current?.contains(e.target as Node)) setAddSlideMenu(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setAddSlideMenu(false) }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [addSlideMenu])

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

  // ── Presentation: "+ Add slide" starter layouts ──────────────────────────────
  // Each starter appends a cell (at the end) with slide_break=true so it opens a
  // NEW slide. A leading break on cell 0 is a harmless no-op (the first cell is
  // always slide 0), so the FIRST slide added to an empty deck still reads as
  // Slide 1. Three starters: a blank text slide, a split (text + figure) slide,
  // and a title slide (a text cell pre-marked slide_kind='title').
  const addTextSlide = () => {
    setAddSlideMenu(false)
    sendAction('report_add_cell', { cell_type: 'markdown', source: '', html: '', slide_break: true })
  }
  const addSplitSlide = () => {
    setAddSlideMenu(false)
    sendAction('report_add_split_cell', { slide_break: true })
  }
  const addTitleSlide = () => {
    setAddSlideMenu(false)
    sendAction('report_add_cell', {
      cell_type: 'markdown',
      source: '# Title\n\nSubtitle',
      html: '<h1>Title</h1><p>Subtitle</p>',
      slide_break: true,
      slide_kind: 'title',
    })
  }

  // ── Per-slide header verbs (addressed to the slide's FIRST cell) ──────────────
  const toggleSlideTitle = (firstCellId: string) =>
    sendAction('report_set_slide_kind', { cell_id: firstCellId })
  const setSlideStyle = (firstCellId: string, style: '' | 'plain' | 'accent') => {
    sendAction('report_set_slide_style', { cell_id: firstCellId, slide_style: style })
    setBgPickerFor(null)
  }
  const setSlideNotes = (firstCellId: string, notes: string) =>
    sendAction('report_set_slide_notes', { cell_id: firstCellId, notes })
  const toggleNotesEditor = (firstCellId: string) =>
    setOpenNotes(prev => {
      const next = new Set(prev)
      if (next.has(firstCellId)) next.delete(firstCellId)
      else next.add(firstCellId)
      return next
    })
  // Reorder a WHOLE slide (drag the slide's grip). from/to are slide positions.
  const moveSlide = (from: number, to: number) => {
    if (from === to) return
    sendAction('report_move_slide', { from, to })
  }

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

  // ── Whole-slide reorder wiring (HTML5 DnD, presentation only) ──────────────
  // ONLY the slide's GRIP is draggable (never the whole group), so a slide drag
  // never collides with a CELL reorder or with text selection/editing inside the
  // slide's cells. The slide GROUP is the drop TARGET: dropping a dragged grip
  // onto another group moves the WHOLE slide (report_move_slide {from, to} — the
  // block reorder). Slide numbers (`n`) ARE the from/to positions. The drop
  // handlers no-op unless a SLIDE (not a cell) is the thing being dragged
  // (`dragSlideN.current != null`), so a cell drop onto a group falls through to
  // the body/cell handlers untouched.
  const makeSlideGripProps = (n: number) => ({
    draggable: true,
    onDragStart: (e: React.DragEvent) => {
      dragSlideN.current = n
      setDragSlide(n)
      e.dataTransfer.effectAllowed = 'move'
      e.dataTransfer.setData('application/x-spyde-report-slide', String(n))
      e.stopPropagation()
    },
    onDragEnd: () => {
      dragSlideN.current = null
      setDragSlide(null)
      setSlideDropOn(null)
    },
  })
  const makeSlideDropProps = (n: number) => ({
    onDragOver: (e: React.DragEvent) => {
      if (dragSlideN.current == null) return   // a cell drag → let it fall through
      e.preventDefault()
      e.stopPropagation()
      setSlideDropOn(n)
    },
    onDrop: (e: React.DragEvent) => {
      if (dragSlideN.current == null) return   // a cell drag → let it fall through
      e.preventDefault()
      e.stopPropagation()
      const from = dragSlideN.current
      dragSlideN.current = null
      setDragSlide(null)
      setSlideDropOn(null)
      if (from !== n) moveSlide(from, n)
    },
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
          <div style={styles.emptyLead}>Start a new document</div>
          <div style={styles.cardGrid}>
            <NewDocCard
              testid="report-new-report-card"
              icon={<ReportGlyph />}
              title="Report"
              desc="A scrolling article — text, figures, and combined panels."
              onClick={() => doNew('report')}
            />
            <NewDocCard
              testid="report-new-presentation-card"
              icon={<PresentationGlyph />}
              title="Presentation"
              desc="A slide deck — present live figures full-screen."
              onClick={() => doNew('presentation')}
            />
          </div>

          {/* Seed a presentation from a built-in guide (one slide per step). */}
          {GUIDES.length > 0 && (
            <div style={styles.emptySection}>
              <div style={styles.emptySectionLabel}>…or seed a presentation from a guide</div>
              <div style={styles.guideChips}>
                {GUIDES.map(g => (
                  <button
                    key={g.id}
                    data-testid={`report-empty-from-guide-${g.id}`}
                    style={styles.guideChip}
                    title={g.summary}
                    onClick={() => newFromGuide(g.id)}
                    onMouseEnter={(e) => { e.currentTarget.style.background = '#313244' }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
                  >{g.title}</button>
                ))}
              </div>
            </div>
          )}

          <div style={styles.emptyOpenRow}>
            <button
              data-testid="report-empty-open"
              style={styles.emptyOpenBtn}
              onClick={doOpen}
              onMouseEnter={(e) => { e.currentTarget.style.background = '#313244' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
            >Open an existing .spyde-report…</button>
          </div>
        </div>
      </div>
    )
  }

  // Render ONE cell (by its flat index `i`) — the between-cell insert indicator +
  // the right cell component by type. Shared by the flat (report) list and the
  // slide-grouped (presentation) list so both render cells identically.
  const renderCell = (cell: ReportCellType, i: number) => (
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

  // The slide-grouped body (presentation only): each slide is a bordered, labeled
  // group with a per-slide header (Slide N + Title-slide toggle + Background
  // picker + drag grip) and a collapsible Speaker-notes area below.
  const slides = isPresentation ? groupSlides(cells) : []

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
            {isPresentation
              ? 'Empty deck — add your first slide below, or drag a figure window here.'
              : 'Drag a figure window here, or add a text / split cell below.'}
          </div>
        )}

        {/* PRESENTATION: a sequence of labeled SLIDE groups, each with a
            per-slide header (Slide N + Title toggle + Background) and a
            collapsible Speaker-notes area below. REPORT: a flat cell list. */}
        {isPresentation
          ? slides.map((slide) => {
              const firstId = slide.first.id
              const meta = slideMetaOf(slide.first)
              const isTitle = meta.kind === 'title'
              const notes = String(slide.first.notes ?? '')
              const hasNotes = notes.trim().length > 0
              const notesOpen = openNotes.has(firstId)
              return (
                <div
                  key={firstId}
                  data-testid={`report-slide-${slide.n}`}
                  data-slide-kind={isTitle ? 'title' : 'content'}
                  data-slide-style={meta.style || 'default'}
                  style={{
                    ...styles.slideGroup,
                    ...(dragSlide === slide.n ? styles.slideGroupDragging : {}),
                    ...(slideDropOn === slide.n && dragSlide !== slide.n
                      ? styles.slideGroupDropOn : {}),
                  }}
                  {...makeSlideDropProps(slide.n)}
                >
                  {/* Per-slide header: grip + Slide N + Title toggle + Background. */}
                  <div style={styles.slideHeader} data-testid={`report-slide-header-${slide.n}`}>
                    <span
                      style={styles.slideGrip}
                      title="Drag to reorder this slide"
                      data-testid={`report-slide-grip-${slide.n}`}
                      {...makeSlideGripProps(slide.n)}
                    >⠿</span>
                    <span style={styles.slideLabel}>Slide {slide.n + 1}</span>
                    <div style={{ flex: 1 }} />
                    {/* Title-slide toggle — a clear labeled pill (not a cryptic T). */}
                    <button
                      data-testid={`report-slide-title-toggle-${slide.n}`}
                      data-active={isTitle ? '1' : '0'}
                      style={isTitle ? styles.slidePillActive : styles.slidePill}
                      title="Make this a big centered TITLE / SECTION slide"
                      onClick={() => toggleSlideTitle(firstId)}
                    >
                      <span style={styles.slideCheck}>{isTitle ? '☑' : '☐'}</span> Title slide
                    </button>
                    {/* Background picker — a small labeled dropdown (Default / Plain /
                        Accent) with colour swatches, NOT the cryptic ◐/○/● cycle. */}
                    <div style={styles.bgPickerWrap}>
                      <button
                        data-testid={`report-slide-bg-toggle-${slide.n}`}
                        style={styles.slidePill}
                        title="Slide background"
                        aria-haspopup="menu"
                        aria-expanded={bgPickerFor === firstId}
                        onClick={() => setBgPickerFor(v => (v === firstId ? null : firstId))}
                      >
                        <span style={{ ...styles.bgSwatch, background: swatchOf(meta.style) }} />
                        {labelOf(meta.style)} ▾
                      </button>
                      {bgPickerFor === firstId && (
                        <div
                          style={styles.bgMenu}
                          data-testid={`report-slide-bg-menu-${slide.n}`}
                          role="menu"
                        >
                          {SLIDE_STYLES.map(s => (
                            <button
                              key={s.value || 'default'}
                              data-testid={`report-slide-bg-${slide.n}-${s.value || 'default'}`}
                              role="menuitemradio"
                              aria-checked={meta.style === s.value}
                              style={styles.bgMenuItem}
                              onMouseEnter={(e) => { e.currentTarget.style.background = '#313244' }}
                              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
                              onClick={() => setSlideStyle(firstId, s.value)}
                            >
                              <span style={{ ...styles.bgSwatch, background: s.swatch }} />
                              {s.label}
                              {meta.style === s.value && <span style={styles.bgCheck}>✓</span>}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* The slide's cell(s). */}
                  <div style={styles.slideCells}>
                    {slide.items.map(({ cell, index }) => renderCell(cell, index))}
                  </div>

                  {/* Speaker notes — BELOW the slide, collapsible, presenter-only. */}
                  <div style={styles.notesRow}>
                    <button
                      data-testid={`report-slide-notes-toggle-${slide.n}`}
                      data-has-notes={hasNotes ? '1' : '0'}
                      style={notesOpen || hasNotes ? styles.notesToggleActive : styles.notesToggle}
                      title="Speaker notes for this slide (presenter view only)"
                      onClick={() => toggleNotesEditor(firstId)}
                    >
                      📝 Speaker notes
                      {hasNotes && <span style={styles.notesDot} title="This slide has notes" />}
                      <span style={styles.notesChevron}>{notesOpen ? '▾' : '▸'}</span>
                    </button>
                  </div>
                  {notesOpen && (
                    <SlideNotesEditor
                      cellId={firstId}
                      notes={notes}
                      onCommit={(v) => setSlideNotes(firstId, v)}
                      onClose={() => toggleNotesEditor(firstId)}
                    />
                  )}
                </div>
              )
            })
          : cells.map((cell, i) => renderCell(cell, i))}

        {/* Trailing insert indicator (drop AFTER the last cell). */}
        {dropIndex === cells.length && cells.length > 0 && (
          <div style={styles.insertLine} data-testid={`report-insert-${cells.length}`} />
        )}

        {/* Add row: slide-native for a presentation, cell-native for a report. */}
        {isPresentation ? (
          <div style={styles.addRow}>
            <div ref={addSlideRef} style={styles.addSlideWrap}>
              <button
                data-testid="report-add-slide"
                style={styles.addSlideBtn}
                title="Add a new slide"
                aria-haspopup="menu"
                aria-expanded={addSlideMenu}
                onClick={() => setAddSlideMenu(v => !v)}
              >+ Add slide ▾</button>
              {addSlideMenu && (
                <div style={styles.addSlideMenu} data-testid="report-add-slide-menu" role="menu">
                  <MenuItem testid="add-slide-text" label="Text slide" onClick={addTextSlide} />
                  <MenuItem testid="add-slide-split" label="Split slide (text + figure)"
                    onClick={addSplitSlide} />
                  <MenuItem testid="add-slide-title" label="Title slide" onClick={addTitleSlide} />
                </div>
              )}
            </div>
          </div>
        ) : (
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
        )}
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

// The per-slide presentation attributes read off the slide's FIRST cell (the
// renderer mirror of model.slide_meta — kept local to avoid a PresentMode import
// cycle). kind '' (content) | 'title'; style '' (default) | 'plain' | 'accent'.
function slideMetaOf(first: ReportCellType | undefined):
  { kind: '' | 'title'; style: '' | 'plain' | 'accent' } {
  const k = (first?.slide_kind ?? '').trim().toLowerCase()
  const s = (first?.slide_style ?? '').trim().toLowerCase()
  return {
    kind: k === 'title' ? 'title' : '',
    style: s === 'plain' || s === 'accent' ? (s as 'plain' | 'accent') : '',
  }
}

/** The human label for a slide background style. */
function labelOf(style: '' | 'plain' | 'accent'): string {
  return (SLIDE_STYLES.find(s => s.value === style) ?? SLIDE_STYLES[0]).label
}

/** The swatch colour for a slide background style. */
function swatchOf(style: '' | 'plain' | 'accent'): string {
  return (SLIDE_STYLES.find(s => s.value === style) ?? SLIDE_STYLES[0]).swatch
}

// A big pick-a-document-type card shown in the empty state (New Report / New
// Presentation). Icon + title + one-line description; accent border on hover.
function NewDocCard({ testid, icon, title, desc, onClick }: {
  testid: string; icon: React.ReactNode; title: string; desc: string
  onClick: () => void
}) {
  const [hover, setHover] = useState(false)
  return (
    <button
      data-testid={testid}
      style={hover ? { ...styles.docCard, ...styles.docCardHover } : styles.docCard}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div style={hover ? { ...styles.docCardIcon, ...styles.docCardIconHover } : styles.docCardIcon}>
        {icon}
      </div>
      <div style={styles.docCardTitle}>{title}</div>
      <div style={styles.docCardDesc}>{desc}</div>
    </button>
  )
}

// Inline SVG glyphs for the two document-type cards (currentColor → theme-aware).
const cardSvg = {
  width: 26, height: 26, viewBox: '0 0 24 24', fill: 'none',
  stroke: 'currentColor', strokeWidth: 1.7,
  strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const,
}
// A page with text lines — a scrolling report.
function ReportGlyph() {
  return (
    <svg {...cardSvg} aria-hidden>
      <rect x="4" y="3" width="16" height="18" rx="2" />
      <path d="M8 8h8M8 12h8M8 16h5" />
    </svg>
  )
}
// A framed slide with a play triangle — a presentation deck.
function PresentationGlyph() {
  return (
    <svg {...cardSvg} aria-hidden>
      <rect x="3" y="4" width="18" height="12" rx="1.5" />
      <path d="M12 16v3M9 21h6" />
      <path d="M10.5 8.5l3.5 2-3.5 2z" fill="currentColor" stroke="none" />
    </svg>
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
    padding: 16, fontSize: 12.5, color: '#7f849c', lineHeight: 1.6,
    overflowY: 'auto',
  },
  emptyLead: {
    fontSize: 11, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase',
    color: '#6c7086', margin: '2px 2px 10px',
  },
  cardGrid: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10,
  },
  docCard: {
    display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
    gap: 6, textAlign: 'left', cursor: 'pointer',
    background: '#181825', border: '1px solid #313244', borderRadius: 10,
    padding: '14px 12px', color: '#cdd6f4',
    transition: 'border-color 120ms ease, background 120ms ease, transform 120ms ease',
  },
  docCardHover: {
    background: '#1e1e2e', borderColor: '#89b4fa', transform: 'translateY(-1px)',
  },
  docCardIcon: {
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    width: 40, height: 40, borderRadius: 9,
    background: 'rgba(137,180,250,0.10)', color: '#89b4fa',
    transition: 'background 120ms ease',
  },
  docCardIconHover: { background: 'rgba(137,180,250,0.20)' },
  docCardTitle: { fontSize: 14, fontWeight: 700, color: '#cdd6f4', marginTop: 2 },
  docCardDesc: { fontSize: 11, color: '#7f849c', lineHeight: 1.4 },
  emptySection: { marginTop: 18 },
  emptySectionLabel: {
    fontSize: 11, fontWeight: 700, letterSpacing: 0.6, textTransform: 'uppercase',
    color: '#6c7086', margin: '0 2px 8px',
  },
  guideChips: { display: 'flex', flexWrap: 'wrap', gap: 6 },
  guideChip: {
    background: 'transparent', border: '1px solid #313244', borderRadius: 999,
    color: '#bac2de', cursor: 'pointer', fontSize: 11.5, padding: '4px 11px',
    transition: 'background 100ms ease',
  },
  emptyOpenRow: { marginTop: 20, borderTop: '1px solid #26263a', paddingTop: 12 },
  emptyOpenBtn: {
    width: '100%', textAlign: 'left', background: 'transparent',
    border: '1px solid #313244', borderRadius: 8, color: '#bac2de',
    cursor: 'pointer', fontSize: 12, padding: '8px 11px',
    transition: 'background 100ms ease',
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
  // ── Slide-native (presentation) chrome ───────────────────────────────────────
  slideGroup: {
    position: 'relative',
    border: '1px solid #313244', borderRadius: 10,
    background: '#1e1e2e', padding: '2px 6px 6px',
    marginBottom: 12,
  },
  slideGroupDragging: { opacity: 0.4 },
  slideGroupDropOn: { borderColor: '#89b4fa', boxShadow: '0 0 0 1px #89b4fa inset' },
  slideHeader: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '5px 2px 6px', borderBottom: '1px solid #313244',
    marginBottom: 6,
  },
  slideGrip: {
    cursor: 'grab', color: '#585b70', fontSize: 13, userSelect: 'none',
    lineHeight: 1,
  },
  slideLabel: {
    fontSize: 11, fontWeight: 700, letterSpacing: 0.4, color: '#89b4fa',
    textTransform: 'uppercase',
  },
  slidePill: {
    display: 'inline-flex', alignItems: 'center', gap: 4,
    background: '#181825', color: '#a6adc8', border: '1px solid #313244',
    borderRadius: 6, padding: '2px 7px', fontSize: 10.5, cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  slidePillActive: {
    display: 'inline-flex', alignItems: 'center', gap: 4,
    background: 'rgba(137,180,250,0.16)', color: '#89b4fa',
    border: '1px solid #89b4fa',
    borderRadius: 6, padding: '2px 7px', fontSize: 10.5, cursor: 'pointer',
    fontWeight: 600, whiteSpace: 'nowrap',
  },
  slideCheck: { fontSize: 11, lineHeight: 1 },
  bgPickerWrap: { position: 'relative', display: 'inline-flex' },
  bgSwatch: {
    width: 10, height: 10, borderRadius: 3, border: '1px solid #45475a',
    display: 'inline-block', flexShrink: 0,
  },
  bgMenu: {
    position: 'absolute', top: '100%', right: 0, marginTop: 4, zIndex: 20,
    minWidth: 128, background: '#1e1e2e', border: '1px solid #45475a',
    borderRadius: 6, padding: 4, display: 'flex', flexDirection: 'column', gap: 1,
    boxShadow: '0 6px 22px rgba(0,0,0,0.5)',
  },
  bgMenuItem: {
    display: 'flex', alignItems: 'center', gap: 7,
    background: 'transparent', border: 'none', color: '#cdd6f4', cursor: 'pointer',
    textAlign: 'left', padding: '5px 8px', fontSize: 11.5, borderRadius: 4,
    width: '100%', whiteSpace: 'nowrap',
  },
  bgCheck: { marginLeft: 'auto', color: '#89b4fa', fontSize: 11 },
  slideCells: {},
  notesRow: { marginTop: 6 },
  notesToggle: {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    background: 'transparent', color: '#6c7086',
    border: '1px dashed #313244', borderRadius: 6,
    padding: '3px 9px', fontSize: 11, cursor: 'pointer',
  },
  notesToggleActive: {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    background: 'rgba(137,180,250,0.06)', color: '#89b4fa',
    border: '1px solid rgba(137,180,250,0.35)', borderRadius: 6,
    padding: '3px 9px', fontSize: 11, cursor: 'pointer',
  },
  notesDot: {
    width: 6, height: 6, borderRadius: '50%', background: '#89b4fa',
    flexShrink: 0,
  },
  notesChevron: { fontSize: 9, color: '#6c7086' },
  addSlideWrap: { position: 'relative', flex: 1, display: 'flex' },
  addSlideBtn: {
    flex: 1, minWidth: 0,
    background: 'rgba(137,180,250,0.10)', color: '#89b4fa',
    border: '1px dashed #89b4fa', borderRadius: 6, padding: '8px',
    fontSize: 12.5, fontWeight: 600, cursor: 'pointer',
  },
  addSlideMenu: {
    position: 'absolute', bottom: '100%', left: 0, right: 0, marginBottom: 4,
    zIndex: 20, background: '#1e1e2e', border: '1px solid #45475a',
    borderRadius: 6, padding: 4, display: 'flex', flexDirection: 'column', gap: 1,
    boxShadow: '0 -6px 22px rgba(0,0,0,0.5)',
  },
}
