/**
 * ReportFigureCell.tsx — a figure cell in the Report sidebar.
 *
 * States:
 *   • placeholder (template drop zone) — dashed box with caption text; drop a
 *     compatible figure/window pill onto it to FILL it (report_add_figure
 *     {source_window_id, at_cell:id}).
 *   • data_offline — the SignalRef couldn't rebind: show the baked PNG (cell.png)
 *     + a small "data offline" badge.
 *   • live — the report figure iframe for fig_id, reusing the exact
 *     iframeRefs/replayState mounting pattern from WindowContent.
 *
 * Phase 2 adds two interactions on a LIVE (non-placeholder) figure cell:
 *
 *   1. COMPOSE drop zones — while a figure/window pill is dragged over the cell,
 *      a 5-zone overlay appears (center + 4 edges). An edge drop immediately
 *      tiles the source in on that side (repfig_compose {mode:'tile-<dir>'}). A
 *      center drop queries the backend for compatible modes (repfig_query_compose
 *      → the spyde:repfig_compose_options CustomEvent) and, when non-tile options
 *      exist, opens a small anchored popover (Overlay / Callout / Tile right).
 *
 *   2. EDIT toolbar — an "Edit" toggle in the hover chrome opens a compact dock
 *      panel below the figure, driven by cell.figure (the pixel-free FigureSpec):
 *      per-panel layer list (cmap / alpha / visibility / remove), a per-panel
 *      refresh (⟳ → repfig_refresh_panel, re-snapshots ONLY that panel) and
 *      remove, and an annotations list + add palette (Text / Circle / Rect /
 *      Arrow).
 *
 * Below the figure: an editable caption (click-to-edit → report_set_caption) +
 * hover chrome (Edit toggle, Refresh-ALL-panels-from-live → report_refresh_figure,
 * delete → report_remove_cell).
 */
import React, { useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { reportClipboard, type SerializedFigureCell } from '../kernel/reportClipboard'
import type { ReportCell, RepfigPanel, RepfigLayer, RepfigSpec } from '../kernel/protocol'
import { FIGURE_DRAG_MIME, WINDOW_DRAG_MIME } from '../kernel/dnd'
import { COLORMAPS } from '../kernel/colormaps'
import { useKeyedDebounce } from './wizardHooks'
import { CellChrome } from './CellChrome'

// The compose modes the backend can return (subset of these per drop).
type ComposeMode =
  | 'overlay' | 'callout'
  | 'tile-up' | 'tile-down' | 'tile-left' | 'tile-right'

// The five drop zones on a figure cell (or, on a multi-panel grid, within the
// hovered PANEL's cell rect).
type Zone = 'center' | 'up' | 'down' | 'left' | 'right'
const ZONE_TILE: Record<Exclude<Zone, 'center'>, ComposeMode> = {
  up: 'tile-up', down: 'tile-down', left: 'tile-left', right: 'tile-right',
}

// The currently-hovered zone PLUS which panel it's relative to (for a grid
// figure) and the panel's on-screen rect (fraction of the shield box, 0..1) so
// ComposeZones can position itself over just that cell. `panelId` is null for a
// single-panel figure (whole-box zones, today's behaviour) and `panelRect` is
// the full box (0,0,1,1) in that case.
interface HoverZone {
  zone: Zone
  panelId: string | null
  panelLabel: string | null
  panelRect: { left: number; top: number; width: number; height: number }
}

const FULL_RECT = { left: 0, top: 0, width: 1, height: 1 }

// The figure cell's CSS box has no native pixel width/height to key an
// aspect-ratio off — the `figure` message (host:"report") carries no `aspect`
// field the way an MDI navigator figure's does (that one is the real-space
// scan aspect, meaningless here), and anyplotlib's report-grid `subplots()`
// call doesn't report its own figsize back either. So the box's CSS
// `aspect-ratio` is DERIVED from the panel grid shape: each panel cell is
// assumed ~4:3 (a common default for an image plot), scaled by cols/rows —
// matches a single panel to the box's long-standing 16/10 default (close to
// 4:3 widened a bit for the caption/chrome) and degrades sensibly for a wide
// row or tall column of panels. This is a SANE DEFAULT, not a measured value.
const PANEL_ASPECT = 4 / 3
function figureAspectRatio(figure: RepfigSpec | undefined | null): number {
  const layout = figure?.layout
  if (!layout || layout.kind !== 'grid') return 16 / 10
  const rows = Math.max(1, Number(layout.rows) || 1)
  const cols = Math.max(1, Number(layout.cols) || 1)
  return (PANEL_ASPECT * cols) / rows
}

// Map a cursor fraction (fx, fy) WITHIN a cell rect (0..1 local to that rect)
// to a Zone: a ~30%-wide edge strip on each side, center otherwise. Shared by
// the single-panel (whole box) and grid (per-panel cell) paths.
function zoneFromLocalFraction(fx: number, fy: number): Zone {
  const edge = 0.3
  const dl = fx, dr = 1 - fx, dt = fy, db = 1 - fy
  const m = Math.min(dl, dr, dt, db)
  if (m > edge) return 'center'
  if (m === dl) return 'left'
  if (m === dr) return 'right'
  if (m === dt) return 'up'
  return 'down'
}

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
const isComposeDrag = (dt: DataTransfer) =>
  DROP_MIMES.some(m => dt.types.includes(m))

// A pending compose prompt (center drop) awaiting the backend's options reply.
interface ComposePrompt {
  sourceWindowId: number
  options: ComposeMode[]
  sameShape: boolean
  navSignalPair: boolean
  /** The grid panel this compose is relative to (null on a single-panel figure). */
  targetPanelId: string | null
}

interface Props {
  cell: ReportCell
  onRemove: () => void
  /** Own index in the cell list (Duplicate → insert at index+1). */
  index: number
}

export function ReportFigureCell({ cell, onRemove, index }: Props) {
  const { state, iframeRefs, replayState, sendAction, dragKind, requestFigurePng } = useSpyDE()
  const [captionEditing, setCaptionEditing] = useState(false)
  const [captionDraft, setCaptionDraft] = useState(cell.caption ?? '')
  const [hover, setHover] = useState(false)
  const [dropHover, setDropHover] = useState(false)     // placeholder fill hover
  const [editOpen, setEditOpen] = useState(false)
  // The SELECTED spec panel id (null = figure-level), mirrored from the backend's
  // report_panel_selected → spyde:report_panel_selected CustomEvent (the backend
  // is the source of truth; a click on the live figure, a dock chip, or a widget
  // drag all funnel through it). Drives WHICH section the edit dock shows.
  const [selectedPanel, setSelectedPanel] = useState<string | null>(null)
  // Mirror editOpen into a ref so the unmount cleanup reads the current value
  // without re-arming the effect (and switching edit mode off) on every toggle.
  const editOpenRef = React.useRef(editOpen)
  React.useEffect(() => { editOpenRef.current = editOpen }, [editOpen])
  // Which compose zone the cursor is over (non-placeholder live cell only), plus
  // which panel it targets on a multi-panel grid.
  const [hoverZone, setHoverZone] = useState<HoverZone | null>(null)
  // A center-drop compose prompt (popover) awaiting / showing options.
  const [prompt, setPrompt] = useState<ComposePrompt | null>(null)

  React.useEffect(() => {
    if (!captionEditing) setCaptionDraft(cell.caption ?? '')
  }, [cell.caption, captionEditing])

  // Toggle backend edit mode alongside the local editOpen flag: in edit mode the
  // backend rebuilds the figure with draggable annotation widgets (drag → persist);
  // out of it the annotations are static markers again. Best-effort cleanup on
  // unmount / report close switches edit mode OFF so the cell isn't left rebuilding
  // in interactive mode with no editor open.
  const toggleEdit = () => {
    setEditOpen(v => {
      const next = !v
      sendAction('repfig_set_edit_mode', { cell_id: cell.id, editing: next })
      return next
    })
  }
  React.useEffect(() => {
    return () => {
      if (editOpenRef.current) {
        sendAction('repfig_set_edit_mode', { cell_id: cell.id, editing: false })
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cell.id])

  // Mirror the backend's selection for THIS cell while the editor is open. The
  // backend re-emits report_panel_selected on a live-figure click, a chip click,
  // or a widget drag; we filter by cell_id. When the editor closes, selection is
  // reset (the dock unmounts anyway).
  React.useEffect(() => {
    if (!editOpen) { setSelectedPanel(null); return }
    const onSelected = (ev: Event) => {
      const d = (ev as CustomEvent).detail as { cell_id?: string; panel_id?: string | null }
      if (d?.cell_id !== cell.id) return
      setSelectedPanel(d.panel_id ?? null)
    }
    window.addEventListener('spyde:report_panel_selected', onSelected)
    return () => window.removeEventListener('spyde:report_panel_selected', onSelected)
  }, [editOpen, cell.id])

  // Clear a lingering hovered zone once the drag ends (the shield unmounts) so a
  // fresh drag doesn't flash a stale highlight.
  React.useEffect(() => { if (dragKind == null) setHoverZone(null) }, [dragKind])

  const fig = state.reportFigures.get(cell.id)
  // CSS-only responsive sizing: figBox is width:100% of the cell (which tracks
  // the sidebar width via ordinary block layout), height held by aspect-ratio
  // derived from the panel grid shape — no JS resize loop on this side. The
  // iframe itself relayouts to its box on resize (anyplotlib-side, not here).
  const figBoxStyle: React.CSSProperties = {
    ...styles.figBox, aspectRatio: String(figureAspectRatio(cell.figure)),
  }

  const commitCaption = () => {
    setCaptionEditing(false)
    if (captionDraft !== (cell.caption ?? '')) {
      sendAction('report_set_caption', { cell_id: cell.id, caption: captionDraft })
    }
  }

  // ── Placeholder fill drop (unchanged Phase-1 behaviour) ───────────────────
  const onPlaceholderDragOver = (e: React.DragEvent) => {
    if (!isComposeDrag(e.dataTransfer)) return
    e.preventDefault()
    e.stopPropagation()   // don't also trigger the sidebar-body insertion logic
    e.dataTransfer.dropEffect = 'copy'
    setDropHover(true)
  }
  const onPlaceholderDrop = (e: React.DragEvent) => {
    if (!isComposeDrag(e.dataTransfer)) return
    e.preventDefault()
    e.stopPropagation()
    setDropHover(false)
    const src = sourceWindowIdFromDrop(e.dataTransfer)
    if (src != null) sendAction('report_add_figure', { source_window_id: src, at_cell: cell.id })
  }

  // ── Compose drop zones (live figure cell) ─────────────────────────────────
  // Panels laid out in a grid (cell.figure.layout.kind === 'grid' with >1
  // panel). Single-panel / non-grid figures keep the whole-box 5-zone behaviour.
  const gridPanels = cell.figure?.layout?.kind === 'grid' ? (cell.figure.panels ?? []) : []
  const gridRows = Math.max(1, Number(cell.figure?.layout?.rows) || 1)
  const gridCols = Math.max(1, Number(cell.figure?.layout?.cols) || 1)
  const isGridFigure = gridPanels.length > 1

  // Nearest occupied grid panel to a (row, col) cell (by grid-cell center
  // distance) — used when the cursor is over a HOLE in a sparse grid.
  const nearestPanelAt = (row: number, col: number): RepfigPanel | null => {
    if (!gridPanels.length) return null
    let best: RepfigPanel | null = null
    let bestDist = Infinity
    for (const p of gridPanels) {
      const [pr, pc] = p.grid_pos
      const d = (pr - row) ** 2 + (pc - col) ** 2
      if (d < bestDist) { bestDist = d; best = p }
    }
    return best
  }

  // Map the cursor position within the shield box to a HoverZone. On a grid
  // figure this first resolves WHICH panel cell the cursor is over (uniform
  // rows/cols — report grids have no ratios), then computes the zone from the
  // LOCAL fraction within that cell; on a single-panel figure it's the whole
  // box (today's behaviour, panelId null).
  const hoverZoneAt = (e: React.DragEvent): HoverZone => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const fx = (e.clientX - r.left) / Math.max(1, r.width)
    const fy = (e.clientY - r.top) / Math.max(1, r.height)
    if (!isGridFigure) {
      return { zone: zoneFromLocalFraction(fx, fy), panelId: null, panelLabel: null, panelRect: FULL_RECT }
    }
    const col = Math.min(gridCols - 1, Math.max(0, Math.floor(fx * gridCols)))
    const row = Math.min(gridRows - 1, Math.max(0, Math.floor(fy * gridRows)))
    let panel = gridPanels.find(p => p.grid_pos[0] === row && p.grid_pos[1] === col) ?? null
    let cellRow = row, cellCol = col
    if (!panel) {
      // Hole — target the nearest occupied panel instead, but keep the
      // highlighted rect on the HOVERED (empty) cell so the overlay tracks the
      // cursor.
      panel = nearestPanelAt(row, col)
    }
    const panelRect = {
      left: cellCol / gridCols, top: cellRow / gridRows,
      width: 1 / gridCols, height: 1 / gridRows,
    }
    const localFx = fx * gridCols - cellCol
    const localFy = fy * gridRows - cellRow
    const idx = gridPanels.indexOf(panel as RepfigPanel)
    return {
      zone: zoneFromLocalFraction(localFx, localFy),
      panelId: panel ? panel.id : null,
      panelLabel: panel ? panelLabel(idx) : null,
      panelRect,
    }
  }
  const onComposeDragOver = (e: React.DragEvent) => {
    if (!isComposeDrag(e.dataTransfer)) return
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
    setHoverZone(hoverZoneAt(e))
  }
  const onComposeDragLeave = (e: React.DragEvent) => {
    // Only clear when actually leaving the box (not crossing a child overlay).
    if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
      setHoverZone(null)
    }
  }
  const onComposeDrop = (e: React.DragEvent) => {
    if (!isComposeDrag(e.dataTransfer)) return
    e.preventDefault()
    e.stopPropagation()
    const hz = hoverZoneAt(e)
    setHoverZone(null)
    const src = sourceWindowIdFromDrop(e.dataTransfer)
    if (src == null) return
    if (hz.zone !== 'center') {
      // Edge → tile immediately on that side, relative to the targeted panel
      // (undefined on a single-panel figure → backend legacy default).
      sendAction('repfig_compose', {
        cell_id: cell.id, mode: ZONE_TILE[hz.zone], source_window_id: src,
        ...(hz.panelId != null ? { target_panel_id: hz.panelId } : {}),
      })
      return
    }
    // Center → query which modes are compatible, then decide.
    beginCenterCompose(src, hz.panelId)
  }

  // Center-drop: fire repfig_query_compose and wait for the matching
  // spyde:repfig_compose_options CustomEvent for THIS cell (~2 s timeout → fall
  // back to tile-right). If only tiles come back, tile-right directly; if richer
  // options exist, open the popover. `targetPanelId` (grid figure only) is
  // threaded through to both the query and the follow-up compose action.
  const beginCenterCompose = (src: number, targetPanelId: string | null) => {
    let done = false
    const onOptions = (ev: Event) => {
      const d = (ev as CustomEvent).detail as {
        cell_id?: string; source_window_id?: number
        options?: string[]; detail?: { same_shape?: boolean; nav_signal_pair?: boolean }
      }
      if (d?.cell_id !== cell.id || d?.source_window_id !== src) return
      if (done) return
      done = true
      window.removeEventListener('spyde:repfig_compose_options', onOptions)
      clearTimeout(timer)
      const options = (d.options ?? []) as ComposeMode[]
      const same = !!d.detail?.same_shape
      const navPair = !!d.detail?.nav_signal_pair
      const hasRich = options.includes('overlay') || options.includes('callout')
      if (!hasRich) {
        // Only tiles → tile-right directly (no ambiguity to resolve).
        sendAction('repfig_compose', {
          cell_id: cell.id, mode: 'tile-right', source_window_id: src,
          ...(targetPanelId != null ? { target_panel_id: targetPanelId } : {}),
        })
        return
      }
      setPrompt({ sourceWindowId: src, options, sameShape: same, navSignalPair: navPair,
                 targetPanelId })
    }
    window.addEventListener('spyde:repfig_compose_options', onOptions)
    const timer = setTimeout(() => {
      if (done) return
      done = true
      window.removeEventListener('spyde:repfig_compose_options', onOptions)
      // No reply → safe default.
      sendAction('repfig_compose', {
        cell_id: cell.id, mode: 'tile-right', source_window_id: src,
        ...(targetPanelId != null ? { target_panel_id: targetPanelId } : {}),
      })
    }, 2000)
    sendAction('repfig_query_compose', {
      cell_id: cell.id, source_window_id: src,
      ...(targetPanelId != null ? { target_panel_id: targetPanelId } : {}),
    })
  }

  const runCompose = (mode: ComposeMode) => {
    if (!prompt) return
    sendAction('repfig_compose', {
      cell_id: cell.id, mode, source_window_id: prompt.sourceWindowId,
      ...(prompt.targetPanelId != null ? { target_panel_id: prompt.targetPanelId } : {}),
    })
    setPrompt(null)
  }

  const isLive = !cell.placeholder && !cell.data_offline && !!fig

  // ── Copy / Duplicate ──────────────────────────────────────────────────────
  // Build the serialized figure cell. For a LIVE cell harvest a fresh PNG from
  // the iframe (so the OFFLINE fallback baked into the paste matches what's on
  // screen); fall back to the cell's baked `png` when the figure can't answer or
  // the cell is already offline.
  const serialize = async (): Promise<SerializedFigureCell> => {
    let png: string | null = cell.png ?? null
    if (fig?.figId) {
      const live = await requestFigurePng(fig.figId)
      if (live) png = live
    }
    return {
      cell_type: 'figure',
      caption: cell.caption ?? '',
      figure: cell.figure,
      png,
    }
  }
  const doCopy = async () => {
    const ser = await serialize()
    reportClipboard.set(ser)
    // Best-effort: also mirror the PNG to the OS clipboard so a paste into an
    // external app (Word / Slack) works. Ignore failure — the internal
    // clipboard is the source of truth for in-report paste.
    if (ser.png) {
      try { await window.electron.clipboardWritePng(ser.png) } catch { /* ignore */ }
    }
  }
  const doDuplicate = async () => {
    // Duplicate = copy → paste at own index+1 immediately; the internal
    // clipboard is NOT touched (a plain Copy would be lost otherwise).
    const ser = await serialize()
    sendAction('report_paste_cell', { cell: ser, index: index + 1 })
  }

  return (
    <div
      data-testid={`report-figcell-${cell.id}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={styles.cell}
    >
      {/* Hover chrome: Edit toggle + Copy + Duplicate + Refresh-from-live +
          delete (not on a placeholder). */}
      {hover && !cell.placeholder && (
        <CellChrome
          cellId={cell.id}
          styles={{ chrome: styles.chrome, chromeBtn: styles.chromeBtn }}
          onCopy={doCopy}
          onDuplicate={doDuplicate}
          onDelete={onRemove}
          deleteTestid={`report-figcell-delete-${cell.id}`}
          deleteTitle="Delete figure"
          leading={
            <button
              data-testid={`report-figcell-edit-toggle-${cell.id}`}
              style={editOpen ? styles.chromeBtnActive : styles.chromeBtn}
              title="Edit figure (layers, annotations)"
              onClick={toggleEdit}
            >✎</button>
          }
          trailing={
            <button
              data-testid={`report-figcell-refresh-${cell.id}`}
              style={styles.chromeBtn}
              title="Refresh all panels from live plots"
              onClick={() => sendAction('report_refresh_figure', { cell_id: cell.id })}
            >⟳</button>
          }
        />
      )}

      {cell.placeholder ? (
        // Template placeholder — dashed drop zone.
        <div
          data-testid={`report-figcell-placeholder-${cell.id}`}
          onDragOver={onPlaceholderDragOver}
          onDragLeave={() => setDropHover(false)}
          onDrop={onPlaceholderDrop}
          style={{ ...styles.placeholder, ...(dropHover ? styles.placeholderHot : {}) }}
        >
          <div style={styles.placeholderIcon}>▤</div>
          <div style={styles.placeholderText}>
            {cell.caption || 'Drop a figure here'}
          </div>
        </div>
      ) : cell.data_offline ? (
        // Rebind failed — show the baked snapshot + a "data offline" badge.
        <div style={figBoxStyle}>
          {cell.png
            ? <img src={cell.png} alt={cell.caption ?? ''} style={styles.offlineImg} />
            : <div style={styles.offlineMissing}>snapshot unavailable</div>}
          <span style={styles.offlineBadge} data-testid={`report-figcell-offline-${cell.id}`}>
            data offline
          </span>
        </div>
      ) : fig ? (
        // Live report figure — a seamless (no-flash) iframe swap host + the
        // compose drop-zone shield stacked on top.
        <div style={figBoxStyle}>
          <SeamlessFigureFrame
            figId={fig.figId}
            filePath={fig.filePath}
            title={fig.title}
            iframeRefs={iframeRefs}
            replayState={replayState}
          />
          {/* Drag shield: the figure iframe is out-of-process and swallows DnD, so
              while a window/figure pill is in flight we mount a transparent shield
              over it to catch dragover/drop (same reason SubWindow shields during
              gestures). Mounted ONLY during the drag → no interference otherwise.
              The zone overlay renders inside it once a zone is hovered. Sits ABOVE
              the frames (composeShield z 3) so pointer capture still works. */}
          {dragKind === 'window' && (
            <div
              data-testid={`figcell-compose-shield-${cell.id}`}
              style={styles.composeShield}
              onDragOver={onComposeDragOver}
              onDragLeave={onComposeDragLeave}
              onDrop={onComposeDrop}
            >
              {hoverZone != null && (
                <ComposeZones active={hoverZone.zone} cellId={cell.id}
                  panelRect={hoverZone.panelRect} panelLabel={hoverZone.panelLabel} />
              )}
            </div>
          )}
        </div>
      ) : (
        // Figure cell whose iframe hasn't arrived yet — show the baked PNG if any.
        <div style={figBoxStyle}>
          {cell.png
            ? <img src={cell.png} alt={cell.caption ?? ''} style={styles.offlineImg} />
            : <div style={styles.pending} data-testid={`report-figcell-pending-${cell.id}`}>rendering…</div>}
        </div>
      )}

      {/* Center-drop compose prompt (Overlay / Callout / Tile right). Only one
          cell shows a prompt at a time, so the bare spec testids are unambiguous. */}
      {prompt && (
        <div style={styles.promptWrap} data-testid="figcell-compose-prompt"
          data-cell={cell.id} role="dialog">
          <div style={styles.promptTitle}>Combine figure…</div>
          <div style={styles.promptRow}>
            {prompt.sameShape && (
              <button data-testid="compose-overlay" style={styles.promptBtn}
                title="Overlay the source as a translucent layer"
                onClick={() => runCompose('overlay')}>Overlay</button>
            )}
            {prompt.navSignalPair && (
              <button data-testid="compose-callout" style={styles.promptBtn}
                title="Add the source as a callout inset"
                onClick={() => runCompose('callout')}>Callout</button>
            )}
            <button data-testid="compose-tile" style={styles.promptBtn}
              title="Tile the source to the right"
              onClick={() => runCompose('tile-right')}>Tile right</button>
            <button style={styles.promptCancel} title="Cancel"
              onClick={() => setPrompt(null)}>Cancel</button>
          </div>
        </div>
      )}

      {/* Caption line (not on a placeholder). */}
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

      {/* Edit toolbar (layers + annotations) — only on a live figure cell. */}
      {editOpen && isLive && cell.figure && (
        <FigureEditPanel cell={cell} selectedPanel={selectedPanel} onClose={() => {
          setEditOpen(false)
          sendAction('repfig_set_edit_mode', { cell_id: cell.id, editing: false })
        }} />
      )}
    </div>
  )
}

// ── Seamless figure iframe swap (no blank flash on rebuild) ────────────────────

/**
 * SeamlessFigureFrame — hosts the report cell's figure iframe and swaps it
 * WITHOUT the blank flash a naive `src` change causes.
 *
 * A report figure rebuild (compose edit, refresh, layout change) mints a BRAND
 * NEW anyplotlib figId for the same cell. Swapping one iframe's `src` blanks the
 * frame while the new document + ESM load + first paint (~100s of ms) → a jarring
 * flash. Instead we keep the OLD iframe mounted and visible while the NEW one
 * loads stacked underneath (absolute inset 0, opacity 0), and only PROMOTE the
 * new one (opacity 1, unmount the old) once it has actually PAINTED.
 *
 * The "painted" signal: there is no explicit ready-postMessage from the figure
 * iframe, so we reuse the SAME handshake the rest of the app relies on — the
 * iframe `load` event, then `replayState(figId)` (which pushes the pixel/selector
 * state the frame needs to draw), then two rAFs to let that state paint (the same
 * 2-rAF settle anyplotlib's own export tests wait on). Only then do we promote.
 *
 * Cross-wiring safety: SpyDEContext keys iframeRefs / latestStates / the PNG
 * harvest by figId, and the two frames have DISTINCT figIds, so mounting both
 * briefly never crosses their state. Each frame binds its OWN figId into
 * iframeRefs and clears it on unmount.
 *
 * Also owns the ResizeObserver → resizeFigure(figId, w, h) so the figure
 * relayouts when the CSS-responsive cell box changes size (mirrors WindowContent).
 */
function SeamlessFigureFrame({ figId, filePath, title, iframeRefs, replayState }: {
  figId: string
  filePath: string | null
  title: string
  iframeRefs: React.MutableRefObject<Map<string, HTMLIFrameElement>>
  replayState: (figId: string) => void
}) {
  // The figId currently PROMOTED (opacity 1). Starts as the first figId; updated
  // only once a newer frame has painted.
  const [shownFigId, setShownFigId] = React.useState(figId)
  // The incoming figId while it loads underneath (null when nothing pending).
  const [pendingFigId, setPendingFigId] = React.useState<string | null>(null)
  const boxRef = React.useRef<HTMLDivElement | null>(null)
  const shownRef = React.useRef(shownFigId)
  shownRef.current = shownFigId
  // figId → its OWN filePath. Each figId's `src` MUST stay pinned to the path it
  // was minted with — a frame that stays mounted (the OLD one during a swap) must
  // NOT have its src rewritten to the new figId's path (that would reload it and
  // defeat the seamless swap). The current prop path always belongs to `figId`.
  const pathByFigId = React.useRef<Map<string, string | null>>(new Map())
  pathByFigId.current.set(figId, filePath)

  // When the prop figId changes to something we're neither showing nor already
  // loading, start loading it underneath.
  React.useEffect(() => {
    if (figId !== shownFigId && figId !== pendingFigId) {
      setPendingFigId(figId)
    }
    // If the prop reverts to the shown one, drop any stale pending frame.
    if (figId === shownFigId && pendingFigId != null) {
      setPendingFigId(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [figId])

  // Promote the pending frame once it has painted: load → replayState → 2 rAFs.
  const onFrameLoad = (loadedFigId: string, el: HTMLIFrameElement) => {
    replayState(loadedFigId)
    window.electron.resizeFigure(loadedFigId,
      Math.max(80, el.clientWidth), Math.max(80, el.clientHeight))
    if (loadedFigId === shownRef.current) return   // the currently-shown frame
    // Let the replayed state paint, THEN promote (drop the old frame).
    requestAnimationFrame(() => requestAnimationFrame(() => {
      // Guard: only promote if this is still the frame we're waiting for (a newer
      // rebuild may have superseded it).
      setPendingFigId(prev => (prev === loadedFigId ? null : prev))
      setShownFigId(prev => (prev === loadedFigId ? prev : loadedFigId))
    }))
  }

  // Keep the SHOWN figure sized to the (CSS-responsive) box — the report cell has
  // no explicit pixel size; its aspect-ratio box resizes with the sidebar. Mirrors
  // WindowContent's rAF-debounced ResizeObserver so a resize triggers exactly one
  // relayout per frame.
  React.useEffect(() => {
    const fit = () => {
      const el = iframeRefs.current.get(shownRef.current)
      if (el && el.clientWidth && el.clientHeight) {
        window.electron.resizeFigure(shownRef.current,
          Math.max(80, el.clientWidth), Math.max(80, el.clientHeight))
      }
    }
    let raf = requestAnimationFrame(fit)
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf); raf = requestAnimationFrame(fit)
    })
    if (boxRef.current) ro.observe(boxRef.current)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [shownFigId, iframeRefs])

  // Render the shown frame (opacity 1), and — while a newer one loads — the
  // pending frame stacked on top but INVISIBLE (opacity 0) so its paint doesn't
  // flash before it's ready. Each frame keeps its OWN pinned src (pathByFigId) so
  // the old frame is never reloaded during the overlap. Both bind their own figId
  // into iframeRefs.
  const renderFrame = (fid: string, visible: boolean) => (
    <iframe
      key={fid}
      ref={el => {
        if (el) iframeRefs.current.set(fid, el)
        else { iframeRefs.current.delete(fid); pathByFigId.current.delete(fid) }
      }}
      src={pathByFigId.current.get(fid) ?? undefined}
      onLoad={(e) => onFrameLoad(fid, e.currentTarget)}
      style={{ ...styles.frame, opacity: visible ? 1 : 0 }}
      title={title}
      data-testid={`figure-${fid}`}
    />
  )

  return (
    <div ref={boxRef} style={styles.frameHost}>
      {renderFrame(shownFigId, true)}
      {pendingFigId != null && pendingFigId !== shownFigId &&
        renderFrame(pendingFigId, false)}
    </div>
  )
}

// ── 5-zone drop overlay ───────────────────────────────────────────────────────

// `panelRect` positions the zones overlay INSIDE the hovered grid cell (percent
// of the shield box); defaults to the full box on a single-panel figure.
// `panelLabel` (e.g. "Panel B"), when present, is appended to the tile labels
// so a multi-panel drop reads as "Tile → of Panel B".
function ComposeZones({ active, cellId, panelRect, panelLabel: targetLabel }: {
  active: Zone
  cellId: string
  panelRect?: { left: number; top: number; width: number; height: number }
  panelLabel?: string | null
}) {
  const zStyle = (z: Zone): React.CSSProperties => ({
    ...styles.zone,
    ...(active === z ? styles.zoneHot : {}),
  })
  const rootStyle: React.CSSProperties = panelRect
    ? {
        ...styles.zonesRoot,
        inset: 'auto',
        left: `${panelRect.left * 100}%`, top: `${panelRect.top * 100}%`,
        width: `${panelRect.width * 100}%`, height: `${panelRect.height * 100}%`,
        right: 'auto', bottom: 'auto',
      }
    : styles.zonesRoot
  const ofSuffix = targetLabel ? ` of ${targetLabel}` : ''
  // Only the cell under the cursor shows zones, so the bare spec testids
  // (figcell-zone-<zone>) are unambiguous; data-cell disambiguates in the DOM.
  return (
    <div style={rootStyle} data-testid="figcell-zones" data-cell={cellId}>
      {/* Edges first (thin strips), center last so it sits between them. */}
      <div data-testid="figcell-zone-up" style={{ ...zStyle('up'), ...styles.zoneUp }}>
        <span style={styles.zoneLabel}>{`Tile ↑${ofSuffix}`}</span>
      </div>
      <div data-testid="figcell-zone-down" style={{ ...zStyle('down'), ...styles.zoneDown }}>
        <span style={styles.zoneLabel}>{`Tile ↓${ofSuffix}`}</span>
      </div>
      <div data-testid="figcell-zone-left" style={{ ...zStyle('left'), ...styles.zoneLeft }}>
        <span style={styles.zoneLabel}>{`Tile ←${ofSuffix}`}</span>
      </div>
      <div data-testid="figcell-zone-right" style={{ ...zStyle('right'), ...styles.zoneRight }}>
        <span style={styles.zoneLabel}>{`Tile →${ofSuffix}`}</span>
      </div>
      <div data-testid="figcell-zone-center" style={{ ...zStyle('center'), ...styles.zoneCenter }}>
        <span style={styles.zoneLabel}>{targetLabel ? `Combine${ofSuffix}` : 'Overlay / Combine'}</span>
      </div>
    </div>
  )
}

// ── Edit panel (layers + annotations) ─────────────────────────────────────────

// A1, B2… panel labels from grid position: row-major letter per panel index.
const PANEL_LETTERS = 'ABCDEFGHIJKLMNOP'
function panelLabel(index: number): string {
  return `Panel ${PANEL_LETTERS[index] ?? String(index + 1)}`
}

// A short human label for an annotation entry.
const ANNOT_LABEL: Record<string, string> = {
  text: 'Text', circle: 'Circle', ellipse: 'Ellipse', rect: 'Rect',
  arrow: 'Arrow', line: 'Line',
}

// The accent used as the default annotation color everywhere it's created
// (PanelEdit.addAnnotation / FigureLevelEdit.addFigAnnotation) — the swatch
// falls back to this when an existing annotation carries no color at all.
const ANNOT_COLOR_DEFAULT = '#ff9800'

// A compact native color input, styled to sit inline in an annotation row
// without disturbing its layout (fixed small square, no browser chrome
// beyond the swatch itself). `value` may be missing/non-string on an
// annotation predating this control — falls back to the accent default.
function ColorSwatch({ value, onChange, testid, title }: {
  value: unknown
  onChange: (color: string) => void
  testid: string
  title: string
}) {
  const color = typeof value === 'string' && value ? value : ANNOT_COLOR_DEFAULT
  return (
    <input
      type="color"
      data-testid={testid}
      title={title}
      value={color}
      onChange={(e) => onChange(e.target.value)}
      style={styles.colorSwatch}
    />
  )
}

function FigureEditPanel({ cell, selectedPanel, onClose }: {
  cell: ReportCell
  selectedPanel: string | null
  onClose: () => void
}) {
  const { sendAction } = useSpyDE()
  const panels = cell.figure?.panels ?? []
  const multiPanel = panels.length > 1

  // Debounced per-(panel,layer) alpha sender so a dragged slider doesn't flood
  // repfig_set_layer (mirrors PlotControlDock's LayersSection pattern).
  const debounceSet = useKeyedDebounce(150)
  const setLayer = (panelId: string, layerId: string,
                    payload: Record<string, unknown>, debounce = false) => {
    const send = () => sendAction('repfig_set_layer',
      { cell_id: cell.id, panel_id: panelId, layer_id: layerId, ...payload })
    if (!debounce) { send(); return }
    debounceSet(`${panelId}:${layerId}`, send)
  }

  // The selection SOURCE OF TRUTH is the backend; clicking a chip just tells it
  // to select (it echoes report_panel_selected → the prop updates). `null` = the
  // Figure chip (figure-level).
  const selectPanel = (panelId: string | null) =>
    sendAction('repfig_select_panel', { cell_id: cell.id, panel_id: panelId })

  // The panel whose section to show (resolve the selected id to a panel; unknown
  // / null → figure-level).
  const activePanel = selectedPanel != null
    ? panels.find(p => p.id === selectedPanel) ?? null
    : null

  return (
    <div style={styles.editPanel} data-testid={`figcell-edit-${cell.id}`}>
      <div style={styles.editHeader}>
        <span style={styles.editTitle}>Edit figure</span>
        <div style={{ flex: 1 }} />
        <button style={styles.editClose} title="Close editor" onClick={onClose}>×</button>
      </div>

      {/* Selection chips: one per panel (A, B, C…) + a Figure chip. */}
      <div style={styles.chipRow} data-testid={`figcell-chips-${cell.id}`}>
        {panels.map((panel, i) => (
          <button
            key={panel.id}
            data-testid={`figcell-chip-${panel.id}`}
            style={activePanel?.id === panel.id ? styles.chipActive : styles.chip}
            title={`Select ${panelLabel(i)}`}
            onClick={() => selectPanel(panel.id)}
          >{PANEL_LETTERS[i] ?? String(i + 1)}</button>
        ))}
        <button
          data-testid={`figcell-chip-figure-${cell.id}`}
          style={activePanel == null ? styles.chipActive : styles.chip}
          title="Figure-level controls (layout, caption, figure annotations)"
          onClick={() => selectPanel(null)}
        >Figure</button>
      </div>

      {activePanel != null ? (
        <PanelEdit
          key={activePanel.id}
          cellId={cell.id}
          panel={activePanel}
          index={panels.indexOf(activePanel)}
          canRemovePanel={multiPanel}
          onSetLayer={setLayer}
          sendAction={sendAction}
        />
      ) : (
        <FigureLevelEdit cell={cell} sendAction={sendAction} />
      )}
    </div>
  )
}

// ── Figure-level section (shown when the "Figure" chip is active) ───────────────

// The panels that occupy a GRID cell — mirrors the backend's `_grid_panels`:
// every panel NOT referenced as a callout inset on any panel's `insets`.
function gridPanelsOf(panels: RepfigPanel[]): RepfigPanel[] {
  const insetIds = new Set<string>()
  for (const p of panels) {
    for (const ins of (p.insets ?? [])) {
      const pid = (ins as Record<string, unknown>).panel
      if (typeof pid === 'string') insetIds.add(pid)
    }
  }
  return panels.filter(p => !insetIds.has(p.id))
}

// The three layout presets, deduplicated for the CURRENT grid-panel count N:
// row (1×N), column (N×1), grid (2 cols × ceil(N/2) rows). Two presets that
// produce the same (rows, cols) shape for this N collapse to one entry (e.g.
// N=2: row=1×2, column=2×1, grid=2×1 — grid is dropped as a duplicate of
// column).
const LAYOUT_PRESET_LABEL: Record<string, string> = { row: 'Row', column: 'Column', grid: 'Grid' }
function presetShape(preset: 'row' | 'column' | 'grid', n: number): { rows: number; cols: number } {
  if (preset === 'row') return { rows: 1, cols: n }
  if (preset === 'column') return { rows: n, cols: 1 }
  const cols = 2
  return { rows: Math.ceil(n / cols), cols }
}
function distinctPresets(n: number): Array<{ preset: 'row' | 'column' | 'grid'; rows: number; cols: number }> {
  const seen = new Set<string>()
  const out: Array<{ preset: 'row' | 'column' | 'grid'; rows: number; cols: number }> = []
  for (const preset of ['row', 'column', 'grid'] as const) {
    const shape = presetShape(preset, n)
    const key = `${shape.rows}x${shape.cols}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push({ preset, ...shape })
  }
  return out
}

// A tiny inline schematic (~28×20px) of a preset's grid shape: pure CSS boxes,
// one per cell, filled row-major (N may be less than rows*cols for the "grid"
// preset when N is odd — the last cell of the schematic is left empty to
// mirror the real row-major fill).
function LayoutPresetIcon({ rows, cols, n }: { rows: number; cols: number; n: number }) {
  const cells = rows * cols
  return (
    <div style={{ ...styles.presetIcon, gridTemplateRows: `repeat(${rows}, 1fr)`, gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {Array.from({ length: cells }, (_, i) => (
        <div key={i} style={i < n ? styles.presetCellFilled : styles.presetCellEmpty} />
      ))}
    </div>
  )
}

function FigureLevelEdit({ cell, sendAction }: {
  cell: ReportCell
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}) {
  const figure = cell.figure
  const panels = figure?.panels ?? []
  const layout = figure?.layout
  const isGrid = layout?.kind === 'grid'
  const rows = Math.max(1, Number(layout?.rows) || 1)
  const cols = Math.max(1, Number(layout?.cols) || 1)
  const gridSummary = isGrid && panels.length > 1 ? `${rows} × ${cols} grid` : 'single panel'

  const gridPanelCount = gridPanelsOf(panels).length
  const presets = gridPanelCount >= 2 ? distinctPresets(gridPanelCount) : []
  const applyPreset = (preset: 'row' | 'column' | 'grid') =>
    sendAction('repfig_apply_layout_preset', { cell_id: cell.id, preset })

  // Debounced layout sender so a dragged slider doesn't flood repfig_set_layout.
  const debounceSet = useKeyedDebounce(150)
  const setLayout = (payload: Record<string, unknown>) =>
    debounceSet('layout', () => sendAction('repfig_set_layout', { cell_id: cell.id, ...payload }))

  const hspace = Number(layout?.hspace ?? 0.2)
  const wspace = Number(layout?.wspace ?? 0.2)
  const [draftH, setDraftH] = React.useState(hspace)
  const [draftW, setDraftW] = React.useState(wspace)
  React.useEffect(() => { setDraftH(hspace) }, [hspace])
  React.useEffect(() => { setDraftW(wspace) }, [wspace])

  const annotations = figure?.annotations ?? []

  // Figure-fraction defaults (0..1, centered) — same accent as panel annotations.
  const ANNOT_COLOR = ANNOT_COLOR_DEFAULT
  const addFigAnnotation = (kind: 'text' | 'circle' | 'rect' | 'arrow') => {
    let annotation: Record<string, unknown>
    if (kind === 'text') {
      annotation = { kind: 'text', x: 0.5, y: 0.5, text: 'Label', color: ANNOT_COLOR, fontsize: 14 }
    } else if (kind === 'circle') {
      annotation = { kind: 'circle', x: 0.5, y: 0.5, r: 0.08, color: ANNOT_COLOR }
    } else if (kind === 'rect') {
      annotation = { kind: 'rect', x: 0.5, y: 0.5, w: 0.2, h: 0.15, color: ANNOT_COLOR }
    } else {
      annotation = { kind: 'arrow', x: 0.35, y: 0.35, u: 0.15, v: 0.15, color: ANNOT_COLOR }
    }
    sendAction('repfig_add_fig_annotation', { cell_id: cell.id, annotation })
  }

  return (
    <div style={styles.panelBlock} data-testid={`figcell-figure-edit-${cell.id}`}>
      <div style={styles.subLabel}>Layout</div>
      <div style={styles.figGridSummary} data-testid={`figcell-grid-summary-${cell.id}`}>
        {gridSummary}
      </div>
      {presets.length > 0 && (
        <div style={styles.presetRow} data-testid={`figcell-layout-presets-${cell.id}`}>
          {presets.map(({ preset, rows: pr, cols: pc }) => (
            <button
              key={preset}
              data-testid={`figcell-layout-preset-${preset}-${cell.id}`}
              style={styles.presetBtn}
              title={`${LAYOUT_PRESET_LABEL[preset]} layout (${pr} × ${pc})`}
              onClick={() => applyPreset(preset)}
            >
              <LayoutPresetIcon rows={pr} cols={pc} n={gridPanelCount} />
              <span style={styles.presetLabel}>{LAYOUT_PRESET_LABEL[preset]}</span>
            </button>
          ))}
        </div>
      )}
      {isGrid && panels.length > 1 && (
        <>
          <div style={styles.layerTop}>
            <span style={styles.hint}>row gap</span>
            <input
              data-testid={`figcell-hspace-${cell.id}`}
              type="range" min={0} max={1} step={0.05}
              value={draftH}
              onChange={(e) => { const v = Number(e.target.value); setDraftH(v); setLayout({ hspace: v }) }}
              style={{ flex: 1 }}
            />
            <span style={{ ...styles.hint, minWidth: 26, textAlign: 'right' }}>{draftH.toFixed(2)}</span>
          </div>
          <div style={styles.layerTop}>
            <span style={styles.hint}>col gap</span>
            <input
              data-testid={`figcell-wspace-${cell.id}`}
              type="range" min={0} max={1} step={0.05}
              value={draftW}
              onChange={(e) => { const v = Number(e.target.value); setDraftW(v); setLayout({ wspace: v }) }}
              style={{ flex: 1 }}
            />
            <span style={{ ...styles.hint, minWidth: 26, textAlign: 'right' }}>{draftW.toFixed(2)}</span>
          </div>
        </>
      )}

      <div style={styles.subLabel}>Caption</div>
      <div style={styles.figCaptionHint}>
        Edit the caption below the figure (click it to type).
      </div>

      {/* Figure-level annotations (figure fractions, draggable in edit mode). */}
      <div style={styles.subLabel}>Figure annotations</div>
      {annotations.length === 0 && (
        <div style={styles.annEmpty}>None yet — add one below.</div>
      )}
      {annotations.map((ann, ai) => (
        <FigureAnnotationRow
          key={ann.id ?? ai}
          cellId={cell.id}
          index={ai}
          annotation={ann as Record<string, unknown> & { kind: string }}
          sendAction={sendAction}
        />
      ))}
      <div style={styles.annPalette}>
        {(['text', 'circle', 'rect', 'arrow'] as const).map(k => (
          <button
            key={k}
            data-testid={`figcell-add-fig-${k}-${cell.id}`}
            style={styles.annAddBtn}
            title={`Add figure ${ANNOT_LABEL[k]}`}
            onClick={() => addFigAnnotation(k)}
          >+ {ANNOT_LABEL[k]}</button>
        ))}
      </div>
    </div>
  )
}

function PanelEdit({ cellId, panel, index, canRemovePanel, onSetLayer, sendAction }: {
  cellId: string
  panel: RepfigPanel
  index: number
  canRemovePanel: boolean
  onSetLayer: (panelId: string, layerId: string, payload: Record<string, unknown>, debounce?: boolean) => void
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}) {
  const annotations = panel.annotations ?? []

  // A robust default annotation position + size in the panel's DATA coordinates.
  // Prefer the snapshot axes (x_axis/y_axis float arrays carried on the spec);
  // fall back to pixel-index midpoint (0..N-1) via the layer clim-agnostic size.
  const annotationDefaults = () => {
    const xs = panel.axes?.x_axis
    const ys = panel.axes?.y_axis
    let x0 = 0, x1 = 100, y0 = 0, y1 = 100
    if (xs && xs.length) { x0 = xs[0]; x1 = xs[xs.length - 1] }
    if (ys && ys.length) { y0 = ys[0]; y1 = ys[ys.length - 1] }
    const cx = (x0 + x1) / 2
    const cy = (y0 + y1) / 2
    const w = Math.abs(x1 - x0) || 100
    const h = Math.abs(y1 - y0) || 100
    const rx = w * 0.15    // ~15% of the image
    const ry = h * 0.15
    return { cx, cy, rx, ry, w, h }
  }

  // Build the annotation dict in the EXACT anyplotlib-marker kwarg shape the
  // backend's figure_builder._apply_annotations consumes (it pops offsets/texts/
  // widths/heights/U/V and forwards the rest as add_* kwargs). Getting these
  // names wrong means the annotation is appended to the spec but never DRAWS
  // (the builder pops a None offsets and `continue`s). Offsets are (N,2) [x,y]
  // arrays in DATA coordinates; a single marker is a 1-length list.
  const ANNOT_COLOR = ANNOT_COLOR_DEFAULT
  const addAnnotation = (kind: 'text' | 'circle' | 'rect' | 'arrow') => {
    const d = annotationDefaults()
    let annotation: Record<string, unknown>
    if (kind === 'text') {
      // add_texts(offsets, texts, color=, fontsize=)
      annotation = { kind: 'text', offsets: [[d.cx, d.cy]], texts: ['Label'],
        color: ANNOT_COLOR, fontsize: 12 }
    } else if (kind === 'circle') {
      // add_circles(offsets, radius=, edgecolors=, facecolors=)
      annotation = { kind: 'circle', offsets: [[d.cx, d.cy]], radius: Math.min(d.rx, d.ry),
        edgecolors: ANNOT_COLOR, facecolors: null, linewidths: 1.5, alpha: 1.0 }
    } else if (kind === 'rect') {
      // add_rectangles(offsets, widths, heights, edgecolors=, facecolors=) —
      // offset is the rectangle CENTER (matplotlib collection convention).
      annotation = { kind: 'rect', offsets: [[d.cx, d.cy]], widths: [d.rx * 2],
        heights: [d.ry * 2], edgecolors: ANNOT_COLOR, facecolors: null,
        linewidths: 1.5, alpha: 1.0 }
    } else {
      // add_arrows(offsets, U, V, edgecolors=) — tail at (cx-rx, cy-ry), pointing
      // toward the center.
      annotation = { kind: 'arrow', offsets: [[d.cx - d.rx, d.cy - d.ry]],
        U: [d.rx], V: [d.ry], edgecolors: ANNOT_COLOR, linewidths: 1.6 }
    }
    sendAction('repfig_add_annotation', { cell_id: cellId, panel_id: panel.id, annotation })
  }

  return (
    <div style={styles.panelBlock} data-testid={`figcell-panel-${panel.id}`}>
      <div style={styles.panelHeader}>
        <span style={styles.panelLabel}>{panelLabel(index)}</span>
        <div style={{ flex: 1 }} />
        <button
          data-testid={`figcell-panel-refresh-${panel.id}`}
          style={styles.smallRefresh}
          title="Refresh this panel from the live plot"
          onClick={() => sendAction('repfig_refresh_panel', { cell_id: cellId, panel_id: panel.id })}
        >⟳</button>
        {canRemovePanel && (
          <button
            data-testid={`figcell-panel-remove-${panel.id}`}
            style={styles.smallRemove}
            title="Remove this panel"
            onClick={() => sendAction('repfig_remove_panel', { cell_id: cellId, panel_id: panel.id })}
          >remove panel</button>
        )}
      </div>

      {/* Layers */}
      <div style={styles.subLabel}>Layers</div>
      {(panel.layers ?? []).map((layer, li) => (
        <LayerEdit
          key={layer.id}
          cellId={cellId}
          panelId={panel.id}
          layer={layer}
          isBase={li === 0}
          onSet={onSetLayer}
          sendAction={sendAction}
        />
      ))}

      {/* Annotations */}
      <div style={styles.subLabel}>Annotations</div>
      {annotations.length === 0 && (
        <div style={styles.annEmpty}>None yet — add one below.</div>
      )}
      {annotations.map((ann, ai) => (
        <AnnotationRow
          key={ai}
          cellId={cellId}
          panelId={panel.id}
          index={ai}
          annotation={ann}
          sendAction={sendAction}
        />
      ))}
      <div style={styles.annPalette}>
        {(['text', 'circle', 'rect', 'arrow'] as const).map(k => (
          <button
            key={k}
            data-testid={`figcell-add-${k}-${panel.id}`}
            style={styles.annAddBtn}
            title={`Add ${ANNOT_LABEL[k]}`}
            onClick={() => addAnnotation(k)}
          >+ {ANNOT_LABEL[k]}</button>
        ))}
      </div>
    </div>
  )
}

function LayerEdit({ cellId, panelId, layer, isBase, onSet, sendAction }: {
  cellId: string
  panelId: string
  layer: RepfigLayer
  isBase: boolean
  onSet: (panelId: string, layerId: string, payload: Record<string, unknown>, debounce?: boolean) => void
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}) {
  const [draftAlpha, setDraftAlpha] = React.useState(layer.alpha)
  React.useEffect(() => { setDraftAlpha(layer.alpha) }, [layer.alpha])
  const title = layer.source?.title || (isBase ? 'Base' : 'Layer')

  return (
    <div style={styles.layerRow} data-testid={`figcell-layer-${panelId}-${layer.id}`}>
      <div style={styles.layerTop}>
        <span style={styles.layerTitle} title={title}>{title}</span>
        <button
          data-testid={`figcell-layer-visible-${layer.id}`}
          title={layer.visible ? 'Hide layer' : 'Show layer'}
          onClick={() => onSet(panelId, layer.id, { visible: !layer.visible })}
          style={layer.visible ? styles.eyeOn : styles.eyeOff}
        >{layer.visible ? '◉' : '○'}</button>
        <button
          data-testid={`figcell-layer-remove-${layer.id}`}
          title="Remove layer"
          onClick={() => sendAction('repfig_remove_layer',
            { cell_id: cellId, panel_id: panelId, layer_id: layer.id })}
          style={styles.removeBtn}
        >×</button>
      </div>
      <div style={styles.layerControls}>
        <select
          data-testid={`figcell-layer-cmap-${layer.id}`}
          style={{ ...styles.select, flex: 1 }}
          value={COLORMAPS.includes(layer.cmap) ? layer.cmap : COLORMAPS[0]}
          onChange={(e) => onSet(panelId, layer.id, { cmap: e.target.value })}
        >
          {/* Include the layer's cmap even if it's outside the standard set (e.g.
              an overlay cmap like "cool"/"spring") so it round-trips. */}
          {!COLORMAPS.includes(layer.cmap) && (
            <option value={layer.cmap}>{layer.cmap}</option>
          )}
          {COLORMAPS.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
      <div style={styles.layerTop}>
        <span style={styles.hint}>alpha</span>
        <input
          data-testid={`figcell-layer-alpha-${layer.id}`}
          type="range" min={0} max={1} step={0.05}
          value={draftAlpha}
          onChange={(e) => {
            const v = Number(e.target.value)
            setDraftAlpha(v)
            onSet(panelId, layer.id, { alpha: v }, true)
          }}
          style={{ flex: 1 }}
        />
        <span style={{ ...styles.hint, minWidth: 26, textAlign: 'right' }}>
          {draftAlpha.toFixed(2)}
        </span>
      </div>
    </div>
  )
}

function AnnotationRow({ cellId, panelId, index, annotation, sendAction }: {
  cellId: string
  panelId: string
  index: number
  annotation: Record<string, unknown> & { kind: string }
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}) {
  const kind = String(annotation.kind ?? '')
  const isText = kind === 'text'
  // A text annotation stores its string(s) in `texts` (the anyplotlib add_texts
  // arg); read/write the first entry. Fall back to a legacy `text` scalar.
  const textOf = (a: Record<string, unknown>): string => {
    const ts = a.texts
    if (Array.isArray(ts) && ts.length) return String(ts[0] ?? '')
    return String(a.text ?? '')
  }
  const [editing, setEditing] = React.useState(false)
  const [draft, setDraft] = React.useState(textOf(annotation))
  const current = textOf(annotation)
  React.useEffect(() => { if (!editing) setDraft(current) }, [current, editing])

  const commitText = () => {
    setEditing(false)
    if (draft !== current) {
      // Preserve the array shape the builder expects; drop any legacy `text`.
      const { text: _drop, ...rest } = annotation
      sendAction('repfig_update_annotation', {
        cell_id: cellId, panel_id: panelId, index,
        annotation: { ...rest, texts: [draft] },
      })
    }
  }

  // Color: `text` kind stores it in `color`; circle/rect/arrow store it in
  // `edgecolors` (a plain string here, not the add_* array form — the builder
  // forwards it straight through as an mpl kwarg). Debounced (~250ms) like the
  // layer alpha slider so dragging the native picker doesn't flood the backend.
  const colorField = isText ? 'color' : 'edgecolors'
  const debounceColor = useKeyedDebounce(250)
  const setColor = (color: string) => {
    debounceColor(`${panelId}:${index}:color`, () => {
      sendAction('repfig_update_annotation', {
        cell_id: cellId, panel_id: panelId, index,
        annotation: { ...annotation, [colorField]: color },
      })
    })
  }

  const label = ANNOT_LABEL[kind] ?? kind

  return (
    <div style={styles.annRow} data-testid={`figcell-annotation-${panelId}-${index}`}>
      <span style={styles.annKind}>{label}</span>
      <ColorSwatch
        value={annotation[colorField]}
        onChange={setColor}
        testid={`figcell-annotation-color-${panelId}-${index}`}
        title="Annotation color"
      />
      {isText ? (
        editing ? (
          <input
            data-testid={`figcell-annotation-text-input-${panelId}-${index}`}
            autoFocus
            style={styles.annInput}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitText}
            onKeyDown={(e) => {
              if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
              else if (e.key === 'Escape') { setDraft(String(annotation.text ?? '')); setEditing(false) }
            }}
          />
        ) : (
          <span
            data-testid={`figcell-annotation-text-${panelId}-${index}`}
            style={styles.annText}
            title="Click to edit text"
            onClick={() => { setDraft(current); setEditing(true) }}
          >{current || <span style={styles.captionPlaceholder}>(empty)</span>}</span>
        )
      ) : (
        <span style={styles.annText} />
      )}
      <div style={{ flex: 1 }} />
      <button
        data-testid={`figcell-annotation-remove-${panelId}-${index}`}
        style={styles.removeBtn}
        title="Delete annotation"
        onClick={() => sendAction('repfig_remove_annotation',
          { cell_id: cellId, panel_id: panelId, index })}
      >×</button>
    </div>
  )
}

// A FIGURE-LEVEL annotation row (sibling of AnnotationRow): figure-fraction
// markers store a text kind's string in the `text` scalar (anyplotlib
// figure-marker schema, not panel `texts`), and edits route through the
// repfig_*_fig_annotation actions.
function FigureAnnotationRow({ cellId, index, annotation, sendAction }: {
  cellId: string
  index: number
  annotation: Record<string, unknown> & { kind: string }
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}) {
  const kind = String(annotation.kind ?? '')
  const isText = kind === 'text'
  const current = String(annotation.text ?? '')
  const [editing, setEditing] = React.useState(false)
  const [draft, setDraft] = React.useState(current)
  React.useEffect(() => { if (!editing) setDraft(current) }, [current, editing])

  const commitText = () => {
    setEditing(false)
    if (draft !== current) {
      sendAction('repfig_update_fig_annotation', {
        cell_id: cellId, index,
        annotation: { ...annotation, text: draft },
      })
    }
  }

  // Figure-level annotations use `color` uniformly across every kind (unlike a
  // panel annotation's shape kinds, which use `edgecolors`). Debounced like the
  // panel swatch so dragging the native picker doesn't flood the backend.
  const debounceColor = useKeyedDebounce(250)
  const setColor = (color: string) => {
    debounceColor(`fig:${index}:color`, () => {
      sendAction('repfig_update_fig_annotation', {
        cell_id: cellId, index,
        annotation: { ...annotation, color },
      })
    })
  }

  const label = ANNOT_LABEL[kind] ?? kind

  return (
    <div style={styles.annRow} data-testid={`figcell-fig-annotation-${index}`}>
      <span style={styles.annKind}>{label}</span>
      <ColorSwatch
        value={annotation.color}
        onChange={setColor}
        testid={`figcell-fig-annotation-color-${index}`}
        title="Annotation color"
      />
      {isText ? (
        editing ? (
          <input
            data-testid={`figcell-fig-annotation-text-input-${index}`}
            autoFocus
            style={styles.annInput}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitText}
            onKeyDown={(e) => {
              if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
              else if (e.key === 'Escape') { setDraft(current); setEditing(false) }
            }}
          />
        ) : (
          <span
            data-testid={`figcell-fig-annotation-text-${index}`}
            style={styles.annText}
            title="Click to edit text"
            onClick={() => { setDraft(current); setEditing(true) }}
          >{current || <span style={styles.captionPlaceholder}>(empty)</span>}</span>
        )
      ) : (
        <span style={styles.annText} />
      )}
      <div style={{ flex: 1 }} />
      <button
        data-testid={`figcell-fig-annotation-remove-${index}`}
        style={styles.removeBtn}
        title="Delete figure annotation"
        onClick={() => sendAction('repfig_remove_fig_annotation', { cell_id: cellId, index })}
      >×</button>
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
    position: 'absolute', top: 4, right: 6, zIndex: 4,
    display: 'flex', alignItems: 'center', gap: 4,
    background: 'rgba(24,24,37,0.92)', borderRadius: 5, padding: '1px 3px',
  },
  chromeBtn: {
    background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
    fontSize: 13, padding: '0 3px', lineHeight: 1,
  },
  chromeBtnActive: {
    background: '#89b4fa', border: 'none', color: '#11111b', cursor: 'pointer',
    fontSize: 13, padding: '0 3px', lineHeight: 1, borderRadius: 4,
  },
  figBox: {
    // aspectRatio is set per-instance (figBoxStyle, derived from the figure's
    // panel grid shape via figureAspectRatio) — width:100% here is what makes
    // the box track the sidebar's CSS width; no JS resize loop involved.
    position: 'relative', width: '100%',
    background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    overflow: 'hidden',
  },
  // Fills the figBox; hosts the (up to two, briefly-overlapping) figure iframes
  // during a seamless swap. Frames are absolute inset 0, so this box is what the
  // ResizeObserver watches for the CSS-responsive relayout.
  frameHost: {
    position: 'absolute', inset: 0,
  },
  frame: {
    position: 'absolute', inset: 0, width: '100%', height: '100%',
    border: 'none',
    // opacity is set per-frame: 0 while a pending frame loads underneath, 1 once
    // promoted. The promoted frame has ALREADY painted (we waited its load + 2
    // rAFs), so the swap is a hard, in-the-same-commit opacity flip with the old
    // frame removed simultaneously — no fade-in gap, no blank flash.
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
  // ── Compose zones ──────────────────────────────────────────────────────────
  composeShield: {
    position: 'absolute', inset: 0, zIndex: 3,
  },
  zonesRoot: {
    position: 'absolute', inset: 0, pointerEvents: 'none',
  },
  zone: {
    position: 'absolute', display: 'flex', alignItems: 'center',
    justifyContent: 'center',
    border: '1.5px dashed rgba(137,180,250,0.35)',
    background: 'rgba(24,24,37,0.10)',
    boxSizing: 'border-box',
    transition: 'background 70ms, border-color 70ms',
  },
  zoneHot: {
    borderColor: '#89b4fa',
    background: 'rgba(137,180,250,0.22)',
  },
  zoneUp: { left: '28%', right: '28%', top: 0, height: '28%' },
  zoneDown: { left: '28%', right: '28%', bottom: 0, height: '28%' },
  zoneLeft: { top: '28%', bottom: '28%', left: 0, width: '28%' },
  zoneRight: { top: '28%', bottom: '28%', right: 0, width: '28%' },
  zoneCenter: { left: '28%', right: '28%', top: '28%', bottom: '28%' },
  zoneLabel: {
    fontSize: 9.5, fontWeight: 600, color: '#bac2de',
    textShadow: '0 1px 2px rgba(0,0,0,0.8)', textAlign: 'center',
    padding: '0 2px', lineHeight: 1.15,
  },
  // ── Compose prompt popover ───────────────────────────────────────────────
  promptWrap: {
    position: 'absolute', left: '50%', top: '46%', transform: 'translate(-50%, -50%)',
    zIndex: 6, minWidth: 180,
    background: 'rgba(24,24,37,0.97)', border: '1px solid #89b4fa',
    borderRadius: 8, padding: '8px 10px',
    boxShadow: '0 6px 22px rgba(0,0,0,0.55)',
  },
  promptTitle: { fontSize: 11.5, fontWeight: 600, color: '#cdd6f4', marginBottom: 6 },
  promptRow: { display: 'flex', flexWrap: 'wrap', gap: 5 },
  promptBtn: {
    background: '#1e1e2e', color: '#cdd6f4', border: '1px solid #45475a',
    borderRadius: 5, padding: '3px 9px', fontSize: 11, cursor: 'pointer',
  },
  promptCancel: {
    background: 'none', color: '#7f849c', border: '1px solid #313244',
    borderRadius: 5, padding: '3px 8px', fontSize: 11, cursor: 'pointer',
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
  // ── Edit panel ────────────────────────────────────────────────────────────
  editPanel: {
    marginTop: 4,
    background: '#181825', border: '1px solid #313244', borderRadius: 6,
    padding: '6px 8px',
  },
  editHeader: {
    display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4,
  },
  editTitle: { fontSize: 11, fontWeight: 600, color: '#cdd6f4' },
  editClose: {
    background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
    fontSize: 15, lineHeight: 1, padding: '0 2px',
  },
  chipRow: {
    display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 4,
    paddingBottom: 4, borderBottom: '1px solid #1e1e2e',
  },
  chip: {
    background: '#1e1e2e', color: '#a6adc8', border: '1px solid #313244',
    borderRadius: 5, padding: '2px 8px', fontSize: 10.5, cursor: 'pointer',
    lineHeight: 1.2,
  },
  chipActive: {
    background: '#89b4fa', color: '#11111b', border: '1px solid #89b4fa',
    borderRadius: 5, padding: '2px 8px', fontSize: 10.5, cursor: 'pointer',
    fontWeight: 600, lineHeight: 1.2,
  },
  figGridSummary: { fontSize: 10.5, color: '#cdd6f4', padding: '1px 0 3px' },
  // ── Layout presets ────────────────────────────────────────────────────────
  presetRow: { display: 'flex', gap: 6, padding: '2px 0 4px' },
  presetBtn: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 5,
    padding: '4px 6px', cursor: 'pointer',
  },
  presetIcon: {
    display: 'grid', gap: 1.5, width: 28, height: 20,
  },
  presetCellFilled: {
    background: '#89b4fa', borderRadius: 1,
  },
  presetCellEmpty: {
    background: '#313244', borderRadius: 1,
  },
  presetLabel: { fontSize: 8.5, color: '#a6adc8' },
  figCaptionHint: { fontSize: 9.5, color: '#6c7086', fontStyle: 'italic', padding: '1px 0' },
  panelBlock: {
    borderTop: '1px solid #1e1e2e', paddingTop: 5, marginTop: 4,
  },
  panelHeader: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 },
  panelLabel: { fontSize: 10.5, fontWeight: 600, color: '#a6adc8' },
  smallRemove: {
    background: 'none', border: '1px solid #45475a', color: '#f38ba8',
    borderRadius: 4, padding: '1px 6px', fontSize: 9.5, cursor: 'pointer',
  },
  smallRefresh: {
    background: 'none', border: '1px solid #45475a', color: '#a6adc8',
    borderRadius: 4, padding: '1px 6px', fontSize: 9.5, cursor: 'pointer',
    lineHeight: 1,
  },
  subLabel: { fontSize: 9.5, color: '#6c7086', margin: '5px 0 2px', fontWeight: 600 },
  layerRow: {
    display: 'flex', flexDirection: 'column', gap: 3,
    padding: '4px 0', borderBottom: '1px solid #1e1e2e',
  },
  layerTop: { display: 'flex', gap: 6, alignItems: 'center' },
  layerControls: { display: 'flex', gap: 6 },
  layerTitle: {
    flex: 1, fontSize: 10.5, color: '#cdd6f4', minWidth: 0,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  select: {
    background: '#1e1e2e', color: '#cdd6f4',
    border: '1px solid #313244', borderRadius: 4, padding: '2px 4px', fontSize: 11,
  },
  hint: { fontSize: 9.5, color: '#6c7086' },
  eyeOn: {
    background: 'none', border: 'none', color: '#89b4fa', cursor: 'pointer',
    fontSize: 12, padding: '0 4px', lineHeight: 1,
  },
  eyeOff: {
    background: 'none', border: 'none', color: '#585b70', cursor: 'pointer',
    fontSize: 12, padding: '0 4px', lineHeight: 1,
  },
  removeBtn: {
    background: 'none', border: 'none', color: '#f38ba8', cursor: 'pointer',
    fontSize: 13, padding: '0 4px', lineHeight: 1, fontWeight: 700,
  },
  annEmpty: { fontSize: 9.5, color: '#585b70', fontStyle: 'italic', padding: '1px 0' },
  annRow: {
    display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0',
  },
  annKind: {
    fontSize: 9, fontWeight: 700, color: '#a6adc8',
    background: '#313244', borderRadius: 4, padding: '1px 5px',
  },
  // Compact native color input — fixed small square, no browser-default label/
  // padding, sits inline in the row without disturbing its height.
  colorSwatch: {
    width: 16, height: 16, padding: 0, border: '1px solid #45475a',
    borderRadius: 3, background: 'none', cursor: 'pointer', flexShrink: 0,
  },
  annText: {
    fontSize: 10.5, color: '#cdd6f4', cursor: 'text', minWidth: 0,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
    maxWidth: 150,
  },
  annInput: {
    background: '#11111b', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 4, padding: '1px 5px', fontSize: 10.5, outline: 'none',
    minWidth: 0, flex: '0 1 150px',
  },
  annPalette: { display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 },
  annAddBtn: {
    background: '#1e1e2e', color: '#a6adc8', border: '1px solid #313244',
    borderRadius: 5, padding: '2px 7px', fontSize: 10, cursor: 'pointer',
  },
}
