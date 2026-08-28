/**
 * SpyDEContext.tsx — React context for the SpyDE Python backend state.
 *
 * Listens for PLOTAPP: messages, maintains the window/figure registry,
 * toolbar configs, and status, and re-renders the MDI when things change.
 */
import React, {
  createContext, useCallback, useContext, useEffect, useReducer, useRef, useState,
} from 'react'
import { useFigureBridge, shellReducer, shellInitialState, LOG_MAX } from '@de/shell-renderer'
import type {
  ShellState, ShellAction, LogEntry, SubItem, EnvPhase, EnvSetupState,
} from '@de/shell-renderer'
import { asPlotAppMessage } from './protocol'
import type { ReportDocState, ReportCell } from './protocol'
import { WINDOW_DRAG_MIME, FIGURE_DRAG_MIME, stashWindowDrag } from './dnd'
import { dlog, dragDumpToConsole } from './dragDiag'
import { EnvSetupOverlay } from '../components/EnvSetupOverlay'

// Re-export the report doc types so components import them from the kernel
// context (the single import surface the rest of the renderer already uses).
export type {
  ReportDocState, ReportCell, RepfigSpec, RepfigPanel, RepfigLayer,
} from './protocol'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ParamSpec {
  name?: string
  type?: string                 // 'enum' | 'number' | 'int' | 'float' | 'bool' | 'file' | ...
  default?: unknown
  options?: string[]
  min?: number                  // when min & max given, a numeric param renders a slider
  max?: number
  step?: number
  extensions?: string[]         // for type 'file' (e.g. ['.cif'])
  tab?: string                  // optional caret tab this param belongs to
  // Show this row only when another param currently equals `value`.
  display_condition?: { parameter: string; value: unknown }
}
export interface SubAction {
  name: string
  icon: string
  label?: string
  toggle: boolean
  parameters: Record<string, ParamSpec>
}
export interface ToolbarAction {
  name: string
  icon: string
  side: 'left' | 'right' | 'top' | 'bottom'
  toggle: boolean
  parameters: Record<string, ParamSpec>
  subfunctions?: SubAction[]
}

export interface SpyDEWindow {
  windowId: number
  title: string
  isNavigator: boolean
  figures: SpyDEFigure[]        // may be multiple iframes in one SubWindow
  toolbarActions: ToolbarAction[]
  visible: boolean
  aspect?: number               // image width/height — sizes the window so the
                                // image fills it (no aspect-letterbox / misaligned selector)
}

export interface SpyDEFigure {
  figId: string
  windowId: number
  filePath: string | null       // null until HTML is written to disk
  title: string
  isNavigator: boolean
  view?: string                 // "3d" for the IPF 3-D explorer figure (2D/3D toggle)
  viewLabel?: string            // chip text for the unified view selector (εxx, VDF, IPF X…)
  viewKind?: string             // "2d" | "3d" — representation kind of this named view
  strainComponents?: string[]   // ["exx","eyy","exy","omega"] → the strain component toggle
}

export type MetadataDict = Record<string, Record<string, string>>
/** Static per-field config text (description / storage key / units), keyed
 *  {group: {prop: …}} — see `MetadataMessage.info`. */
export interface MetadataField {
  description?: string
  key?: string
  units?: string
  derived?: string
}
export type MetadataInfo = Record<string, Record<string, MetadataField>>
/** Dask block layout of the displayed node — the dock's chunk viewer draws it.
 *  `chunks[d]` is truncated to the first 128 entries; `counts[d]` is the real
 *  number of blocks along that dimension. */
export interface ChunkInfo {
  shape: number[]
  chunks: number[][]
  counts: number[]
  names: string[]
  nav_ndim: number
  dtype: string
  itemsize: number
  nbytes: number
  chunk_bytes: number
  n_chunks: number
  /** A chunk does NOT hold whole signal frames — the navigator-killer. */
  signal_split: boolean
}
export interface Composition { elements: string[]; percentages: Record<string, number> }
export interface Histogram {
  counts: number[]
  edges: number[]
  vmin: number
  vmax: number
  threshold?: number | null   // dotted marker line (Find-Vectors detector threshold)
  dataMin?: number            // full data extent; the bins may cover less of it
  dataMax?: number
  clipped?: boolean           // bins are robust quantiles — end bins are overflow
}

// LogEntry and LOG_MAX are the shell's (@de/shell-renderer/shellState) —
// re-exported so the components that import them from here keep working.
export type { LogEntry } from '@de/shell-renderer'
export { LOG_MAX } from '@de/shell-renderer'

/** A console_result/console_vars value description (shape × dtype badge). */
export interface ConsoleVarKind {
  name: string
  kind?: string
  shape?: number[] | null
  dtype?: string | null
  lazy?: boolean
}
/** One row of the console's live variable table — the result-chip strip reads
 *  the "assign"/"out" entries; "signal" entries resolve a dropped windowId to
 *  its console variable name (drag-in from a SubWindow's console-ref grip). */
export interface ConsoleVarEntry extends ConsoleVarKind {
  source: 'signal' | 'assign' | 'out'
  window_ids?: number[] | null
}
/** The console bar's last-executed-cell readout (the echo strip). */
export interface ConsoleResult {
  execId: number
  ok: boolean
  valueRepr: string
  stdout: string
  error: string
  traceback: string
  durationMs: number
  result: ConsoleVarKind | null
}

/**
 * The console live-preview reply (eye-toggled thumbnail/sparkline/scalar) — the
 * camelCase mirror of `ConsolePreviewResultMessage`. `previewId` gates
 * newest-wins in ConsoleBar; `kind` selects the ConsolePreviewSlot render.
 */
export interface ConsolePreviewResult {
  previewId: number
  kind: 'image' | 'sparkline' | 'scalar' | 'unavailable'
  w: number
  h: number
  dataB64: string
  points: (number | null)[] | null
  text: string
  shape: number[] | null
  dtype: string | null
  reason: string
  elapsedMs: number
}

export interface SelectorInfo {
  windowId: number
  mode: 'crosshair' | 'integrate'
  title?: string
  /** Per-selector key (a navigator can carry several selectors). */
  selectorId?: number
  /** Widget colour — the dock row's dot. */
  color?: string
  /** How many navigation positions a POINT selector sums (1 = plain
   *  crosshair). Only sent for a 1-D (movie/time) navigator; its absence is
   *  what hides the control on a 2-D one, where "n frames" has no direction. */
  sumFrames?: number
  /** Length of that navigation axis, so the dock can cap the ladder. */
  navSize?: number
  /** Seconds per navigation position (0 when the axis isn't time), so a
   *  summed window can be labelled with the rate it works out to. */
  navScale?: number
  /** Raw camera frames integrated into ONE navigation position, when the
   *  source streams finer than it was loaded at (a CSB event stream). Its
   *  presence is what lets the width ladder go BELOW one position to a single
   *  raw frame; absent for an ordinary movie, where nothing lies underneath. */
  rawPerPlane?: number
}
/** The named navigators a navigator window offers (its top chip strip). */
export interface NavigatorOptions { names: string[]; current?: string | null }
export type { SubItem } from '@de/shell-renderer'
export interface TreeNode { name: string; signal_id: number; children: TreeNode[] }
export interface AxisRow {
  index: number
  name: string
  size: number
  scale: number | null
  offset: number | null
  units: string
  navigate: boolean
}

// Extends the shell's chrome slice (status / logs / env setup / backend
// death / computing overlays / toolbar action state) — see
// @de/shell-renderer's shellState.ts. Only SpyDE's own fields are listed here.
interface State extends ShellState {
  windows: Map<number, SpyDEWindow>
  figures: Map<string, SpyDEFigure>
  // Report figure cells' iframes — same SpyDEFigure shape + same figId-keyed
  // iframeRefs/replayState binary-replay machinery, but keyed by CELL id and
  // kept OUT of the MDI `windows`/`figures` state so they never open an MDI
  // subwindow. `host:"report"` figure messages route here.
  reportFigures: Map<string, SpyDEFigure>
  // The authoritative report document (mirrored from `report_state`), or null
  // before any report is opened/created.
  report: ReportDocState | null
  metadata: Map<number, MetadataDict>
  // {group: {prop: raw}} of writable cells for that window's metadata dict —
  // presence gates editability, the value is the unit-free raw value the
  // inline editor pre-fills with. Sent as a sibling field on the same
  // `metadata` message (see protocol.ts).
  metadataEditable: Map<number, Record<string, Record<string, string>>>
  // Static field descriptions for the dock's detail popover. One config-derived
  // table for the whole app (not per window), so the last one received wins.
  metadataInfo: MetadataInfo
  // windowId → dask block layout, absent for eager data.
  chunking: Map<number, ChunkInfo>
  histograms: Map<number, Histogram>
  selectors: Map<number, SelectorInfo>
  signalTrees: Map<number, TreeNode>
  signalTreeActive: Map<number, number>   // windowId → active node signal_id
  navigatorOptions: Map<number, NavigatorOptions>   // navigator windowId → named navigators
  axes: Map<number, AxisRow[]>
  // windowId → the Axes table "+" origin-pick is live on that window's plot.
  // BACKEND-OWNED (the `offset_pick` message): it owns the crosshair widget, so
  // it owns the toggle. A renderer-local boolean drifted — switching windows
  // reset it while the crosshair stayed on the plot.
  offsetPick: Map<number, boolean>
  composition: Map<number, Composition>     // windowId → sample elements + percentages
  dashboardUrl: string | null
  activeWindowId: number | null
  navShapePrompt: NavShapePrompt | null   // pending scan-shape/step-size dialog
  signalTypes: Map<number, { current: string; options: string[] }>   // windowId → signal-type info
  playback: { playing: boolean; speed: number; loop: boolean }   // movie playback clock (session-wide)
  consoleResult: ConsoleResult | null       // last-executed cell (the ConsoleBar echo strip)
  consoleVars: ConsoleVarEntry[]            // live variable table (chips + signal-ref resolution)
  consoleCompletions: { completeId: number; matches: string[] } | null
  consolePreview: ConsolePreviewResult | null   // last live-preview reply (the eye-toggled slot)
}

// First-run environment setup progress: the shell's, re-exported.
export type { EnvPhase, EnvSetupState } from '@de/shell-renderer'

// Backend `nav_shape_prompt`: confirm the scan grid + step size before opening a
// navigated dataset (4D-STEM / stack). Mirrors NavShapeDialog's prop type.
export interface NavShapePrompt {
  nav_shape: number[]
  n_patterns: number
  signal_shape: number[]
  scale: number
  units: string
  filename: string
}

type Action =
  | ShellAction
  | { type: 'READY'; dashboardUrl?: string }
  | { type: 'FIGURE'; windowId: number; figId: string; fileUrl: string | null; title: string; isNavigator: boolean; aspect?: number; view?: string; viewLabel?: string; viewKind?: string; strainComponents?: string[] }
  | { type: 'WINDOW_TITLE'; windowId: number; title: string }
  | { type: 'TOOLBAR_CONFIG'; windowId: number; plotId: number; actions: ToolbarAction[] }
  | { type: 'WINDOW_VISIBILITY'; windowId: number; visible: boolean }
  | { type: 'WINDOW_CLOSED'; windowId: number }
  | { type: 'SET_ACTIVE'; windowId: number }
  | { type: 'METADATA'; windowIds: number[]; metadata: MetadataDict
      editable?: Record<string, Record<string, string>>; info?: MetadataInfo
      chunking?: ChunkInfo | null }
  | { type: 'COMPOSITION'; windowIds: number[]; composition: Composition }
  | { type: 'AXES'; windowIds: number[]; axes: AxisRow[] }
  | { type: 'OFFSET_PICK'; windowId: number; on: boolean }
  | { type: 'HISTOGRAM'; windowId: number; histogram: Histogram }
  | { type: 'NAV_SHAPE_PROMPT'; prompt: NavShapePrompt | null }
  | { type: 'SIGNAL_TYPE'; windowIds: number[]; current: string; options: string[] }
  | { type: 'SELECTOR_INFO'; info: SelectorInfo }
  | { type: 'SELECTOR_REMOVED'; selectorId: number }
  | { type: 'SIGNAL_TREE'; windowId: number; tree: TreeNode; activeSignalId?: number }
  | { type: 'NAVIGATOR_OPTIONS'; windowId: number; names: string[]; current?: string | null }
  | { type: 'PLAYBACK'; playing: boolean; speed: number; loop: boolean }
  | { type: 'CONSOLE_RESULT'; result: ConsoleResult }
  | { type: 'CONSOLE_VARS'; vars: ConsoleVarEntry[] }
  | { type: 'CONSOLE_COMPLETIONS'; completeId: number; matches: string[] }
  | { type: 'CONSOLE_PREVIEW_RESULT'; preview: ConsolePreviewResult }
  | { type: 'REPORT_STATE'; report: ReportDocState }
  | { type: 'REPORT_FIGURE'; cellId: string; figure: SpyDEFigure }

function spydeReducer(state: State, action: Action): State {
  switch (action.type) {
    case 'READY':
      return {
        ...state,
        ready: true,
        dashboardUrl: action.dashboardUrl ?? null,
        status: 'Ready',
      }

    case 'FIGURE': {
      // The main process already wrote the HTML to disk and gave us a file:// URL.
      const figure: SpyDEFigure = {
        figId: action.figId,
        windowId: action.windowId,
        filePath: action.fileUrl,
        title: action.title,
        isNavigator: action.isNavigator,
        view: action.view,
        viewLabel: action.viewLabel,
        viewKind: action.viewKind,
        strainComponents: action.strainComponents,
      }

      const newFigures = new Map(state.figures)
      newFigures.set(action.figId, figure)

      // Attach figure to its window (create window record if needed)
      const newWindows = new Map(state.windows)
      if (!newWindows.has(action.windowId)) {
        newWindows.set(action.windowId, {
          windowId: action.windowId,
          title: action.title,
          isNavigator: action.isNavigator,
          figures: [],
          toolbarActions: [],
          visible: true,
        })
      }
      const win = { ...newWindows.get(action.windowId)! }
      // Replace by id; AND a new secondary view replaces the window's previous
      // figure of that same `view` (the IPF X/Y/Z selector re-emits the 3-D
      // explorer and the density heatmap with fresh fig ids — view="3d"/"density");
      // AND a new named view (view_label) replaces the prior figure with that same
      // label (re-running strain/IPF emits fresh fig ids — don't stack chips).
      win.figures = [
        ...win.figures.filter(f => f.figId !== action.figId
          && !(action.view != null && f.view === action.view)
          && !(action.viewLabel != null && f.viewLabel === action.viewLabel
               && f.figId !== action.figId)),
        figure,
      ]
      // A secondary view figure (e.g. the IPF 3-D explorer, view="3d") or a
      // named chip view (view_label — strain εyy/εxy/ω, committed-tree views)
      // must NOT rename the window or flip its navigator flag — those belong
      // to the primary figure. (Without the view_label guard a committed
      // Strain window ended up titled "ω": the last-emitted chip view won.)
      if (!action.view && !action.viewLabel) {
        win.title = action.title
        win.isNavigator = action.isNavigator
        if (action.aspect && action.aspect > 0) win.aspect = action.aspect
      }
      newWindows.set(action.windowId, win)

      // Default the active (sidebar-controlled) window to the first signal panel.
      const activeWindowId = state.activeWindowId ??
        (action.isNavigator ? null : action.windowId)

      return { ...state, windows: newWindows, figures: newFigures, activeWindowId }
    }

    case 'WINDOW_TITLE': {
      const win = state.windows.get(action.windowId)
      if (!win || win.title === action.title) return state
      const newWindows = new Map(state.windows)
      newWindows.set(action.windowId, { ...win, title: action.title })
      return { ...state, windows: newWindows }
    }

    case 'TOOLBAR_CONFIG': {
      // toolbar_config can arrive BEFORE the figure message that creates the
      // window (PlotState emits it at construction). Upsert so it's never
      // dropped — the figure later fills in title/figures on the same record.
      const newWindows = new Map(state.windows)
      const existing = newWindows.get(action.windowId)
      newWindows.set(action.windowId, {
        windowId: action.windowId,
        title: existing?.title ?? 'Plot',
        isNavigator: existing?.isNavigator ?? false,
        figures: existing?.figures ?? [],
        toolbarActions: action.actions,
        visible: existing?.visible ?? true,
      })
      return { ...state, windows: newWindows }
    }

    case 'WINDOW_VISIBILITY': {
      const newWindows = new Map(state.windows)
      const win = newWindows.get(action.windowId)
      if (win) {
        newWindows.set(action.windowId, { ...win, visible: action.visible })
      }
      return { ...state, windows: newWindows }
    }

    case 'WINDOW_CLOSED': {
      const newWindows = new Map(state.windows)
      newWindows.delete(action.windowId)
      // Drop ALL per-window state so e.g. the Navigator Selector toggle and the
      // histogram/metadata/axes for a closed window don't linger in the dock.
      const drop = <V,>(m: Map<number, V>) => {
        if (!m.has(action.windowId)) return m
        const n = new Map(m); n.delete(action.windowId); return n
      }
      const activeWindowId = state.activeWindowId === action.windowId
        ? (newWindows.size ? [...newWindows.keys()][0] : null)
        : state.activeWindowId
      // Selectors are keyed by selector_id (not window id) — prune every row
      // whose OWNING window closed.
      const selectors = new Map(
        [...state.selectors].filter(([, s]) => s.windowId !== action.windowId),
      )
      // A closed window can never legitimately still be "computing" — drop it
      // so a stale overlay can't linger if the stop message raced the close.
      const computingWindows = state.computingWindows.has(action.windowId)
        ? new Set([...state.computingWindows].filter(id => id !== action.windowId))
        : state.computingWindows
      return {
        ...state,
        windows: newWindows,
        selectors,
        histograms: drop(state.histograms),
        metadata: drop(state.metadata),
        metadataEditable: drop(state.metadataEditable),
        chunking: drop(state.chunking),
        axes: drop(state.axes),
        offsetPick: drop(state.offsetPick),
        composition: drop(state.composition),
        signalTrees: drop(state.signalTrees),
        signalTreeActive: drop(state.signalTreeActive),
        navigatorOptions: drop(state.navigatorOptions),
        activeActions: drop(state.activeActions),
        subItems: drop(state.subItems),
        computingWindows,
        activeWindowId,
      }
    }

    case 'SET_ACTIVE':
      return { ...state, activeWindowId: action.windowId }

    case 'METADATA': {
      const metadata = new Map(state.metadata)
      const metadataEditable = new Map(state.metadataEditable)
      const chunking = new Map(state.chunking)
      for (const wid of action.windowIds) {
        metadata.set(wid, action.metadata)
        metadataEditable.set(wid, action.editable ?? {})
        // Eager data sends null — drop any stale layout from a lazy ancestor.
        if (action.chunking) chunking.set(wid, action.chunking)
        else chunking.delete(wid)
      }
      return { ...state, metadata, metadataEditable, chunking,
               metadataInfo: action.info ?? state.metadataInfo }
    }

    case 'COMPOSITION': {
      const composition = new Map(state.composition)
      for (const wid of action.windowIds) composition.set(wid, action.composition)
      return { ...state, composition }
    }

    case 'AXES': {
      const axes = new Map(state.axes)
      for (const wid of action.windowIds) axes.set(wid, action.axes)
      return { ...state, axes }
    }

    case 'OFFSET_PICK': {
      if ((state.offsetPick.get(action.windowId) ?? false) === action.on) return state
      const offsetPick = new Map(state.offsetPick)
      offsetPick.set(action.windowId, action.on)
      return { ...state, offsetPick }
    }

    case 'HISTOGRAM': {
      const histograms = new Map(state.histograms)
      histograms.set(action.windowId, action.histogram)
      return { ...state, histograms }
    }

    case 'SELECTOR_INFO': {
      // Keyed by selector_id when present (one row PER SELECTOR); merged so a
      // mode-only re-emit (set_selector_mode) keeps the title/colour from the
      // creation-time message.
      const selectors = new Map(state.selectors)
      const key = action.info.selectorId ?? action.info.windowId
      const prev = selectors.get(key)
      selectors.set(key, { ...prev, ...action.info })
      return { ...state, selectors }
    }

    case 'SELECTOR_REMOVED': {
      // A signal window closed and its driving selector was deregistered
      // backend-side — drop its dock row directly. WINDOW_CLOSED alone can't
      // do this: selectors are keyed by the NAVIGATOR's window_id (one
      // navigator can drive several signal windows / selectors), so closing
      // one signal window doesn't match any selector's windowId there.
      if (!state.selectors.has(action.selectorId)) return state
      const selectors = new Map(state.selectors)
      selectors.delete(action.selectorId)
      return { ...state, selectors }
    }

    case 'NAVIGATOR_OPTIONS': {
      const navigatorOptions = new Map(state.navigatorOptions)
      navigatorOptions.set(action.windowId, { names: action.names, current: action.current })
      return { ...state, navigatorOptions }
    }

    case 'PLAYBACK':
      return {
        ...state,
        playback: { playing: action.playing, speed: action.speed, loop: action.loop },
      }

    case 'CONSOLE_RESULT':
      return { ...state, consoleResult: action.result }

    case 'CONSOLE_VARS':
      return { ...state, consoleVars: action.vars }

    case 'CONSOLE_COMPLETIONS':
      return {
        ...state,
        consoleCompletions: { completeId: action.completeId, matches: action.matches },
      }

    case 'CONSOLE_PREVIEW_RESULT':
      return { ...state, consolePreview: action.preview }

    case 'REPORT_STATE': {
      // Prune `reportFigures` to the cells the authoritative document still
      // has — a cell can be removed (report_remove_cell / undo / New / a
      // different report opened) without ever going through REPORT_FIGURE
      // again, so nothing else would otherwise evict its stale entry. A
      // closed report (open:false) clears every report figure outright. This
      // only ever removes figIds that were stamped into `reportFigures` by a
      // REPORT_FIGURE action (i.e. belonged to a report cell) — MDI `figures`
      // is a completely separate map and is untouched here.
      if (!action.report.open) {
        return { ...state, report: action.report, reportFigures: new Map() }
      }
      const liveIds = new Set(action.report.cells.map(c => c.id))
      let reportFigures = state.reportFigures
      for (const cellId of reportFigures.keys()) {
        if (!liveIds.has(cellId)) {
          if (reportFigures === state.reportFigures) reportFigures = new Map(reportFigures)
          reportFigures.delete(cellId)
        }
      }
      return { ...state, report: action.report, reportFigures }
    }

    case 'REPORT_FIGURE': {
      // A report figure cell's iframe (host:"report"), keyed by CELL id. A
      // re-render of the same cell (Refresh from live / rebind) replaces the
      // entry with the fresh figId — the ReportFigureCell mounts the new one.
      const reportFigures = new Map(state.reportFigures)
      reportFigures.set(action.cellId, action.figure)
      return { ...state, reportFigures }
    }

    case 'SIGNAL_TREE': {
      const signalTrees = new Map(state.signalTrees)
      signalTrees.set(action.windowId, action.tree)
      const signalTreeActive = new Map(state.signalTreeActive)
      const prevActive = signalTreeActive.get(action.windowId)
      const nodeChanged = action.activeSignalId != null && action.activeSignalId !== prevActive
      if (action.activeSignalId != null) signalTreeActive.set(action.windowId, action.activeSignalId)
      // A genuine tree-node switch (not just a re-emit of the same node)
      // leaves any in-flight compute overlay behind — it belonged to the PRIOR
      // node's plot. Clear it as a staleness backstop; the backend's own
      // matching stop message (see lifecycle.window_computing) is the primary
      // mechanism and should already have cleared it in the normal case.
      const computingWindows = nodeChanged && state.computingWindows.has(action.windowId)
        ? new Set([...state.computingWindows].filter(id => id !== action.windowId))
        : state.computingWindows
      return { ...state, signalTrees, signalTreeActive, computingWindows }
    }

    case 'NAV_SHAPE_PROMPT':
      return { ...state, navShapePrompt: action.prompt }

    case 'SIGNAL_TYPE': {
      const signalTypes = new Map(state.signalTypes)
      for (const wid of action.windowIds)
        signalTypes.set(wid, { current: action.current, options: action.options })
      return { ...state, signalTypes }
    }

    default:
      // Not one of SpyDE's — hand it to the shell, which owns the chrome slice
      // (status / logs / env setup / backend death / computing overlays /
      // toolbar action state) and returns `state` untouched for anything it
      // does not own either.
      return shellReducer(state, action as ShellAction)
  }
}

// ── Context ───────────────────────────────────────────────────────────────────

interface SpyDEContextValue {
  state: State
  iframeRefs: React.MutableRefObject<Map<string, HTMLIFrameElement>>
  // Latest awi_state per figure (key → value). Replayed when an iframe loads so
  // data/selectors pushed before the iframe was listening aren't lost (the
  // "black image" race).
  latestStates: React.MutableRefObject<Map<string, Map<string, unknown>>>
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  setActiveWindow: (windowId: number) => void
  /** Re-send a figure's stashed state. `target` names WHICH iframe — needed
   *  when a figure is mounted twice (sidebar cell + presented slide), where the
   *  shared figId→element map holds only the last-registered one. */
  replayState: (figId: string, target?: HTMLIFrameElement) => void
  // Harvest a rendered PNG from a figure's iframe (the anyplotlib export
  // protocol). Resolves null on timeout/error. Used by the report save flow +
  // any future PNG-export path.
  requestFigurePng: (figId: string, timeoutMs?: number) => Promise<string | null>
  clearNavShapePrompt: () => void
  // Load Stack dialog (renderer-only UI state, opened from the File menu).
  stackDialogOpen: boolean
  openStackDialog: () => void
  closeStackDialog: () => void
  // Check for Updates / GPU Status dialogs (renderer-only UI state, opened
  // from the Help menu — both the native menu and MenuBar.tsx's HTML one).
  updateDialogOpen: boolean
  openUpdateDialog: () => void
  closeUpdateDialog: () => void
  gpuStatusDialogOpen: boolean
  openGpuStatusDialog: () => void
  closeGpuStatusDialog: () => void
  gpuHelpDialogOpen: boolean
  openGpuHelpDialog: () => void
  closeGpuHelpDialog: () => void
  reportDialogOpen: boolean
  openReportDialog: () => void
  closeReportDialog: () => void
  // MDIArea registers its tile-all-windows function here so StatusBar's
  // "Tile" button can trigger it without threading window-layout state (which
  // lives in MDIArea's local refs) through the shared context.
  tileWindowsRef: React.MutableRefObject<(() => void) | null>
  // What kind of thing is currently being dragged (set from window-level
  // dragstart/dragend/drop capture listeners by inspecting the drag's MIME
  // types). 'window' = a window/figure pill (carries a source window); null =
  // nothing / an unrelated drag. Drives the MDI overlay-drop shield: only when
  // a window pill is in flight do OTHER SubWindows mount their transparent
  // drag-shield + "Overlay images" zone over the figure iframe.
  dragKind: 'window' | null
}

const SpyDEContext = createContext<SpyDEContextValue | null>(null)

export function SpyDEProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(spydeReducer, {
    windows: new Map(),
    figures: new Map(),
    reportFigures: new Map(),
    report: null,
    metadata: new Map(),
    metadataEditable: new Map(),
    metadataInfo: {},
    chunking: new Map(),
    composition: new Map(),
    histograms: new Map(),
    selectors: new Map(),
    signalTrees: new Map(),
    signalTreeActive: new Map(),
    navigatorOptions: new Map(),
    axes: new Map(),
    offsetPick: new Map(),
    activeActions: new Map(),
    subItems: new Map(),
    computingWindows: new Set<number>(),
    status: 'Starting…',
    ready: false,
    dashboardUrl: null,
    activeWindowId: null,
    streamLines: [],
    logEntries: [],
    logLevel: 'INFO',
    navShapePrompt: null,
    loading: { busy: false, text: '' },
    signalTypes: new Map(),
    backendExited: null,
    envSetup: null,
    playback: { playing: false, speed: 1, loop: false },
    consoleResult: null,
    consoleVars: [],
    consoleCompletions: null,
    consolePreview: null,
  })

  // The figure bridge — iframe registry, state retention and replay — now lives
  // in @de/shell-renderer, shared with de-groundcrew. It exposes its maps as
  // `{current}` boxes, so `iframeRefs` / `latestStates` / `latestBinaryStates`
  // keep the exact shape the components that read them already expect (a
  // React.MutableRefObject<Map<…>>), and none of them needed changing.
  //
  // The three subtleties that used to live here — replay taking an explicit
  // target so a doubly-mounted figure serves ITSELF, binary frames stashed per
  // PANEL (`geom::pixelField`) rather than per pixel field, and replaying a COPY
  // because postMessage transfers and detaches — are documented on the bridge.
  const figureBridge = useFigureBridge(dlog)
  const iframeRefs = figureBridge.iframes
  const latestStates = figureBridge.states
  const latestBinaryStates = figureBridge.binaryStates
  // Mirror reportFigures into a ref so the (stable-identity, deps-[]) message
  // effect below can read the LATEST cell→figure map synchronously — e.g. to
  // find a report cell's OLD figId before it's replaced, so its shadow state
  // (latestStates/latestBinaryStates) can be evicted. Declared here (before the
  // message effect) so the effect's closure captures a ref whose `.current` is
  // always fresh; also read by the harvest effect further below.
  const reportFiguresRef = useRef(state.reportFigures)
  reportFiguresRef.current = state.reportFigures
  // Mirror the authoritative report doc into a ref for the same reason (the
  // deps-[] message effect + the e2e test hook read the LATEST doc synchronously).
  const reportRef = useRef(state.report)
  reportRef.current = state.report
  const tileWindowsRef = useRef<(() => void) | null>(null)
  const [stackDialogOpen, setStackDialogOpen] = useState(false)
  const [updateDialogOpen, setUpdateDialogOpen] = useState(false)
  const [gpuStatusDialogOpen, setGpuStatusDialogOpen] = useState(false)
  const [gpuHelpDialogOpen, setGpuHelpDialogOpen] = useState(false)
  const [reportDialogOpen, setReportDialogOpen] = useState(false)
  const [dragKind, setDragKind] = useState<'window' | null>(null)

  // Replay is the bridge's; see @de/shell-renderer/figureBridge for why it
  // takes an explicit target (a figure mounted in both the report sidebar and
  // a presented slide registers twice under one figId, and a freshly-loaded
  // frame must serve itself rather than whichever mount won the map).
  const replayState = figureBridge.replay

  /**
   * What a figure would replay into a freshly-mounted iframe.
   *
   * A report figure is mounted TWICE — once in the sidebar cell, once on the
   * presented slide — and both register under the SAME figId, so the present
   * copy draws entirely from what `replayState` re-sends. A multi-panel figure
   * whose panels arrive as separate binary pixel states will therefore render
   * only the panels still present in the stash: a panel whose binary state is
   * missing draws NOTHING, while its HTML scale-bar overlay survives, which is
   * exactly the reported "empty panel with a scale bar and no ticks".
   *
   * So the discriminator is simply: how many panels does the spec have, and how
   * many binary pixel states are stashed for that figure? Run
   * `__spydeFigureDump()` in DevTools while the bad slide is up.
   */
  const figureDump = React.useCallback(() => {
    // The classifier is SpyDE's: WHERE a figure is mounted is what makes this
    // dump diagnostic, and only this app has a report sidebar and a slide deck.
    const rows = figureBridge.dump((el) =>
      el?.closest('[data-testid="present-slide"]') ? 'present-slide'
        : el?.closest('[data-testid="report-sidebar"]') ? 'report-sidebar'
          : el ? 'other' : 'NONE')
    // eslint-disable-next-line no-console
    console.table(rows)
    return rows
  }, [figureBridge])

  React.useEffect(() => {
    ;(window as unknown as Record<string, unknown>).__spydeFigureDump = figureDump
  }, [figureDump])

  // Ask a figure's iframe for a rendered PNG (the anyplotlib export protocol).
  // Posts `{type:'anyplotlib_export_png', requestId, opts}` into the iframe and
  // resolves on the matching `anyplotlib_export_png_result` window message.
  // Resolves null on timeout / no iframe / error — so a save NEVER blocks on a
  // figure that can't answer (the backend falls back to its baked PNG).
  const requestFigurePng = React.useCallback(
    (figId: string, timeoutMs = 1500): Promise<string | null> => {
      const iframe = iframeRefs.current.get(figId)
      if (!iframe?.contentWindow) return Promise.resolve(null)
      const requestId = `png_${figId}_${Date.now()}_${Math.random().toString(36).slice(2)}`
      return new Promise<string | null>((resolve) => {
        let done = false
        const finish = (v: string | null) => {
          if (done) return
          done = true
          window.removeEventListener('message', onMsg)
          clearTimeout(timer)
          resolve(v)
        }
        const onMsg = (e: MessageEvent) => {
          const d = e.data
          if (d?.type === 'anyplotlib_export_png_result' && d.requestId === requestId) {
            finish(typeof d.dataUrl === 'string' ? d.dataUrl : null)
          }
        }
        window.addEventListener('message', onMsg)
        const timer = setTimeout(() => finish(null), timeoutMs)
        try {
          iframe.contentWindow!.postMessage(
            // includeWidgets: a PNG harvested while a cell is in EDIT MODE keeps
            // the annotation widgets (harmless otherwise; anyplotlib never exports
            // the edit chrome / grab handles).
            { type: 'anyplotlib_export_png', requestId, opts: { includeWidgets: true } }, '*',
          )
        } catch { finish(null) }
      })
    },
    [],
  )

  // ── Python → Renderer message dispatch ──────────────────────────────────

  // Log-record COALESCING. The backend can spray hundreds of records a second
  // (a [NAV-PROFILE] trace, a dask storm). One dispatch per record = one React
  // commit per record, each of which re-renders every consumer of this context.
  // Buffer them and flush ONE batched dispatch per animation frame instead, so a
  // burst costs one render, not N. The setTimeout is a backstop: rAF does not
  // fire while the window is hidden/minimised, and records must still land.
  const pendingLogs = useRef<LogEntry[]>([])
  const logRaf = useRef<number | null>(null)
  const logTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const logSeq = useRef(0)

  const flushLogs = useCallback(() => {
    if (logRaf.current != null) { cancelAnimationFrame(logRaf.current); logRaf.current = null }
    if (logTimer.current != null) { clearTimeout(logTimer.current); logTimer.current = null }
    const batch = pendingLogs.current
    if (batch.length === 0) return
    pendingLogs.current = []
    dispatch({ type: 'LOG', entries: batch })
  }, [])

  const queueLog = useCallback((entry: LogEntry) => {
    entry.seq = logSeq.current++
    const pending = pendingLogs.current
    pending.push(entry)
    // Never let the pending queue outgrow what the buffer can hold (a flood
    // while the window is hidden would otherwise pin unbounded memory).
    if (pending.length > LOG_MAX) pending.splice(0, pending.length - LOG_MAX)
    if (logRaf.current == null) logRaf.current = requestAnimationFrame(flushLogs)
    if (logTimer.current == null) logTimer.current = setTimeout(flushLogs, 250)
  }, [flushLogs])

  useEffect(() => {
    const handleMessage = (raw: Record<string, unknown>) => {
      // Narrow the raw IPC payload into the discriminated PlotAppMessage union;
      // the `switch (msg.type)` below then narrows each field per-variant, so the
      // handlers read typed fields instead of casting them one-by-one.
      const msg = asPlotAppMessage(raw)
      switch (msg.type) {
        case 'ready':
        case 'dask_ready':
          dispatch({ type: 'READY', dashboardUrl: msg.dashboard })
          break

        case 'status':
          dispatch({ type: 'STATUS', text: msg.text })
          break

        case 'error':
          dispatch({ type: 'STATUS', text: `⚠ ${msg.text}` })
          break

        case 'backend_exited':
          // The Python sidecar died (synthesised by runner.ts) OR a packaged
          // first-launch env-setup failure (synthesised by index.ts, which also
          // passes a `reason`). Either way, surface the blocking overlay — every
          // sendAction after this no-ops.
          dispatch({ type: 'BACKEND_EXITED', code: msg.code ?? null, reason: msg.reason })
          break

        case 'env_setup':
          // First-run `uv sync` progress (index.ts). Drives the floating setup
          // overlay so a multi-minute download never looks frozen.
          if (msg.event === 'start') dispatch({ type: 'ENV_SETUP_START' })
          else if (msg.event === 'done') dispatch({ type: 'ENV_SETUP_DONE' })
          else dispatch({
            type: 'ENV_SETUP_PROGRESS',
            phase: msg.phase, step: msg.step,
            percent: (typeof msg.percent === 'number' ? msg.percent : null),
            raw: String(msg.raw ?? ''),
          })
          break

        case 'figure': {
          // Normal path: main process wrote the HTML and gave us file_url.
          // Test path: html is injected directly → fall back to a data URL.
          let fileUrl = msg.file_url ?? null
          if (!fileUrl && msg.html) {
            fileUrl = 'data:text/html;charset=utf-8,' +
              encodeURIComponent(msg.html)
          }
          // A report-hosted figure (host:"report", cell_id set) belongs to a
          // report figure cell — route it to reportFigures (NOT the MDI
          // windows). It still uses the SAME figId-keyed iframeRefs/replayState
          // binary-replay path, so its iframe recovers pre-mount frames.
          if (msg.host === 'report' && msg.cell_id) {
            // A cell can be re-rendered (Refresh from live / rebind / compose
            // edit) in place, which mints a BRAND NEW anyplotlib figId for the
            // SAME cell. The old figId's shadow state (latestStates /
            // latestBinaryStates — the awi_state replay stash) is now orphaned:
            // nothing will ever look it up again (report cells are keyed by
            // cell id, not figId, everywhere except these two maps), so it just
            // grows the maps forever across a long report-editing session. Evict
            // it here, BEFORE the new figure lands, using the figId we know is
            // being replaced (only ever a figId that belonged to a report cell —
            // never touches an MDI figure's entry).
            const prev = reportFiguresRef.current.get(msg.cell_id)
            if (prev && prev.figId && prev.figId !== msg.fig_id) {
              figureBridge.evict(prev.figId)
            }
            dispatch({
              type: 'REPORT_FIGURE',
              cellId: msg.cell_id,
              figure: {
                figId: msg.fig_id,
                windowId: msg.window_id,
                filePath: fileUrl,
                title: msg.title || 'Figure',
                isNavigator: false,
              },
            })
            break
          }
          dispatch({
            type: 'FIGURE',
            windowId: msg.window_id,
            figId: msg.fig_id,
            fileUrl,
            title: msg.title || 'Plot',
            isNavigator: msg.is_navigator || false,
            aspect: msg.aspect,
            view: msg.view,
            viewLabel: msg.view_label,
            viewKind: msg.view_kind,
            strainComponents: msg.strain_components,
          })
          break
        }

        case 'window_title':
          // Lightweight title update (a rename) — updates every listed window's
          // header name WITHOUT re-emitting the figure (which would reload the
          // iframe). window_ids covers the whole tree so the signal + navigator
          // windows' shared [Name] segment both refresh.
          for (const wid of (msg.window_ids || [])) {
            dispatch({ type: 'WINDOW_TITLE', windowId: wid, title: msg.title || '' })
          }
          break

        case 'toolbar_config':
          dispatch({
            type: 'TOOLBAR_CONFIG',
            windowId: msg.window_id,
            plotId: msg.plot_id,
            actions: msg.toolbar_actions || [],
          })
          break

        case 'window_visibility':
          dispatch({
            type: 'WINDOW_VISIBILITY',
            windowId: msg.window_id,
            visible: msg.visible,
          })
          break

        case 'window_closed':
          dispatch({ type: 'WINDOW_CLOSED', windowId: msg.window_id })
          break

        case 'window_computing':
          dispatch({
            type: 'WINDOW_COMPUTING',
            windowId: msg.window_id,
            computing: msg.computing,
          })
          break

        case 'state_update':
          // Forward to the iframe AND remember it, so it can be replayed if the
          // iframe (re)loads after this arrived.
          figureBridge.applyState(msg.fig_id, msg.key, msg.value)
          break

        case 'state_update_binary':
          // A raw image frame (pixels as a Uint8Array, no base64) — posted to
          // the iframe AND retained, so a figure whose FIRST real paint arrives
          // before its iframe's onLoad (the common case for a console-created
          // window, which gets no organic second paint) can be replayed rather
          // than staying permanently blank. The bridge stashes per PANEL and
          // transfers the buffer; both are documented there.
          figureBridge.applyBinary(
            msg.fig_id, msg.key, msg.header, msg.buffer as Uint8Array)
          break

        case 'composition':
          dispatch({
            type: 'COMPOSITION',
            windowIds: msg.window_ids ?? [],
            composition: {
              elements: msg.elements ?? [],
              percentages: msg.percentages ?? {},
            },
          })
          break

        case 'metadata':
          dispatch({
            type: 'METADATA',
            windowIds: msg.window_ids ?? [],
            metadata: msg.metadata ?? {},
            editable: msg.editable,
            info: msg.info,
            chunking: msg.chunking,
          })
          break

        case 'axes_info':
          dispatch({
            type: 'AXES',
            windowIds: msg.window_ids ?? [],
            axes: msg.axes ?? [],
          })
          break

        case 'offset_pick':
          dispatch({
            type: 'OFFSET_PICK',
            windowId: msg.window_id,
            on: !!msg.on,
          })
          break

        case 'action_active':
          dispatch({
            type: 'ACTION_ACTIVE',
            windowId: msg.window_id,
            name: msg.name,
            active: msg.active,
          })
          break

        case 'sub_item':
          dispatch({
            type: 'SUB_ITEM',
            windowId: msg.window_id,
            action: msg.action,
            name: msg.name,
            color: msg.color ?? '#89b4fa',
            vtype: msg.vtype,
            calculation: msg.calculation,
            active: msg.active,
          })
          break

        case 'histogram':
          dispatch({
            type: 'HISTOGRAM',
            windowId: msg.window_id,
            histogram: {
              counts: msg.counts ?? [],
              edges: msg.edges ?? [],
              vmin: msg.vmin,
              vmax: msg.vmax,
              threshold: msg.threshold ?? null,
              dataMin: msg.data_min,
              dataMax: msg.data_max,
              clipped: msg.clipped ?? false,
            },
          })
          break

        case 'nav_shape_prompt':
          dispatch({ type: 'NAV_SHAPE_PROMPT', prompt: msg })
          break

        // Examples → Show Example Data Directory. The BACKEND owns where that
        // is (em-database, overridable via EM_DATABASE_DATA_DIR) but only the
        // main process can open a folder, so it round-trips through here.
        case 'open_path':
          if (msg.path) window.electron.openPath(String(msg.path))
          break

        case 'loading':
          dispatch({ type: 'LOADING', busy: Boolean(msg.busy), text: String(msg.text ?? '') })
          break

        case 'signal_type_info':
          dispatch({
            type: 'SIGNAL_TYPE',
            windowIds: msg.window_ids ?? [],
            current: String(msg.current ?? ''),
            options: msg.options ?? [],
          })
          break

        case 'selector_info':
          dispatch({
            type: 'SELECTOR_INFO',
            info: {
              windowId: msg.window_id,
              selectorId: msg.selector_id,
              mode: msg.mode ?? 'crosshair',
              // Omit absent fields so the reducer's merge keeps the
              // creation-time title/colour on a mode-only re-emit.
              ...(msg.title != null ? { title: msg.title } : {}),
              ...(msg.color != null ? { color: msg.color } : {}),
              ...(msg.sum_frames != null
                ? { sumFrames: Number(msg.sum_frames) } : {}),
              ...(msg.nav_size != null ? { navSize: Number(msg.nav_size) } : {}),
              ...(msg.nav_scale != null
                ? { navScale: Number(msg.nav_scale) } : {}),
              ...(msg.raw_per_plane != null
                ? { rawPerPlane: Number(msg.raw_per_plane) } : {}),
            },
          })
          break

        case 'selector_removed':
          dispatch({ type: 'SELECTOR_REMOVED', selectorId: msg.selector_id })
          break

        case 'signal_tree':
          if (msg.tree) {
            dispatch({
              type: 'SIGNAL_TREE',
              windowId: msg.window_id,
              tree: msg.tree,
              activeSignalId: msg.active_signal_id,
            })
          }
          break

        case 'navigator_options':
          dispatch({
            type: 'NAVIGATOR_OPTIONS',
            windowId: msg.window_id,
            names: msg.names ?? [],
            current: msg.current,
          })
          break

        case 'playback_state':
          // The movie clock changed state (play/pause, speed cycle, or an
          // auto-stop at the movie end). Drives the Play toggle highlight + the
          // Fast Forward "×N" speed badge.
          dispatch({
            type: 'PLAYBACK',
            playing: Boolean(msg.playing),
            speed: Number(msg.speed ?? 1),
            loop: Boolean(msg.loop),
          })
          break

        case 'console_result':
          dispatch({
            type: 'CONSOLE_RESULT',
            result: {
              execId: msg.exec_id,
              ok: Boolean(msg.ok),
              valueRepr: String(msg.value_repr ?? ''),
              stdout: String(msg.stdout ?? ''),
              error: String(msg.error ?? ''),
              traceback: String(msg.traceback ?? ''),
              durationMs: Number(msg.duration_ms ?? 0),
              result: msg.result ?? null,
            },
          })
          break

        case 'console_vars':
          dispatch({ type: 'CONSOLE_VARS', vars: msg.vars ?? [] })
          break

        case 'console_completions':
          dispatch({
            type: 'CONSOLE_COMPLETIONS',
            completeId: msg.complete_id,
            matches: msg.matches ?? [],
          })
          break

        case 'console_preview_result':
          // Live-preview reply (the eye-toggled slot). Kept in the reducer so
          // ConsoleBar can gate newest-wins on preview_id; the camelCase mirror
          // matches ConsolePreviewResult.
          dispatch({
            type: 'CONSOLE_PREVIEW_RESULT',
            preview: {
              previewId: Number(msg.preview_id ?? 0),
              kind: (msg.kind as ConsolePreviewResult['kind']) ?? 'unavailable',
              w: Number(msg.w ?? 0),
              h: Number(msg.h ?? 0),
              dataB64: String(msg.data_b64 ?? ''),
              points: (msg.points as (number | null)[] | undefined) ?? null,
              text: String(msg.text ?? ''),
              shape: (msg.shape as number[] | null | undefined) ?? null,
              dtype: (msg.dtype as string | null | undefined) ?? null,
              reason: String(msg.reason ?? ''),
              elapsedMs: Number(msg.elapsed_ms ?? 0),
            },
          })
          break

        case 'report_state': {
          // The authoritative report document. Mirrored into state so the
          // sidebar + cells re-render. The reducer prunes `reportFigures` to
          // the surviving cell ids (or clears it on report-closed) — but the
          // figId-keyed shadow state (latestStates/latestBinaryStates/
          // iframeRefs) lives OUTSIDE the reducer as refs, so evict any figIds
          // that are about to fall out of `reportFigures` HERE, using the map
          // as it stood just before this update.
          const report = msg.report as ReportDocState | undefined
          if (report) {
            const prevFigures = reportFiguresRef.current
            const liveIds = report.open ? new Set(report.cells.map(c => c.id)) : null
            for (const [cellId, fig] of prevFigures) {
              if (!liveIds || !liveIds.has(cellId)) {
                if (fig.figId) {
                  figureBridge.evict(fig.figId)
                }
              }
            }
            dispatch({ type: 'REPORT_STATE', report })
          }
          break
        }

        case 'report_saved':
          // Zip written — surface a transient status (the sidebar reads
          // report.dirty for the persistent indicator).
          dispatch({ type: 'STATUS', text: `Report saved: ${msg.path}` })
          break

        case 'report_vectors_choice':
          // Deferred vectors-figure drop — the sidebar owns the prompt.
          window.dispatchEvent(new CustomEvent('spyde:report_vectors_choice', { detail: msg }))
          break

        case 'report_need_snapshots':
          // The backend needs a fresh PNG per cell before it writes the zip.
          // Re-broadcast as a DOM CustomEvent; the provider's snapshot effect
          // (below) harvests via requestFigurePng and replies with
          // report_snapshots {token, images}. Doing it there keeps requestFigurePng
          // out of this message-effect's closure (which has no deps).
          window.dispatchEvent(new CustomEvent('spyde:report_need_snapshots', { detail: msg }))
          break

        case 'log':
          // Coalesced (see queueLog) — NOT dispatched per record. `area` is the
          // backend's authoritative subsystem tag (log_stream._area_for); it used
          // to be dropped here, forcing the panel to re-derive a worse guess from
          // the logger name for every row on every render.
          queueLog({
            level: String(msg.level), name: String(msg.name),
            area: msg.area != null ? String(msg.area) : undefined,
            msg: String(msg.msg), time: Number(msg.time),
          })
          break

        case 'log_backfill': {
          // Authoritative history replace — the backend's ring already contains
          // anything still queued, so drop the pending batch rather than letting
          // it re-append duplicates after the replace.
          pendingLogs.current = []
          const entries = (msg.entries ?? []) as LogEntry[]
          for (const e of entries) e.seq = logSeq.current++
          dispatch({ type: 'LOG_BACKFILL', entries })
          break
        }

        case 'log_level':
          dispatch({ type: 'LOG_LEVEL', level: String(msg.level) })
          break

        // Wizard-scoped events (live fit readout + library-ready) + the
        // workflow-node bind ack. Re-broadcast as DOM CustomEvents so the
        // relevant component (a caret; ConsoleBar for console_node_bound) can
        // subscribe without threading them through the global reducer.
        case 'vom_fit':
        case 'vom_library_ready':
        case 'om_library_ready':
        // EBSD Indexing caret: the dictionary-ready ack and the live
        // best-match readout under the crosshair.
        case 'ebsd_dictionary_ready':
        case 'ebsd_match':
        // The Examples menu's contents (em-database): techniques, sizes,
        // shapes and which datasets are already downloaded. Consumed by
        // MenuBar, which asks for it every time the menu opens.
        case 'example_catalogue':
        // Fit wizard (spyde/actions/fit_action.py) — `fit_catalogue` is the
        // component picker's shapes, sent once on open; `fit_state` is the
        // whole model after every edit. Consumed by FitWizard.
        case 'fit_catalogue':
        case 'fit_state':
        case 'bg_state':
        case 'fv_auto_params':
        case 'fv_models':
        case 'fv_calibration':
        case 'cod_results':
        case 'cod_cif_ready':
        case 'gpu_status_result':
        case 'first_run_result':
        case 'console_node_bound':
        case 'layers_state':
        case 'repfig_compose_options':
        case 'report_panel_selected':
        case 'report_exported':
        // Movie BLOCK editor (spyde/actions/report/movie.py) — the full-screen
        // editor's authoritative state, export-done, and the "open this cell in the
        // editor" signal (from the sidebar Movie card / add-with-open). Consumed by
        // MovieGate / MovieEditor.
        case 'movie_state':
        case 'movie_frame':
        case 'movie_done':
        case 'movie_edit_open':
        // Examples-menu download progress/terminal — consumed by DownloadToasts
        // (app-global, not wizard-scoped, but the same re-broadcast fits).
        case 'download_progress':
        case 'download_done':
        // Drift Correction caret (spyde/actions/drift_action.py) — caret state,
        // the ROI discovery preview (~20 frames aligned on the box, with its
        // sharpening gain), whole-movie solve progress, the streamed dy/dx
        // batches, and the solved model. Consumed by DriftWizard; the dy/dx
        // curve itself is painted by the backend into its own figure window.
        case 'drift_state':
        case 'drift_preview':
        case 'drift_trace':
        case 'drift_progress':
        case 'drift_result':
        // DPC caret (spyde/actions/dpc_action.py) — the measured descan +
        // available vacuum datasets, the fitted scan/detector rotation, and the
        // derived field's stats. Consumed by DpcWizard; the map itself is a
        // backend-painted figure window.
        case 'dpc_state':
        case 'dpc_estimate':
        case 'dpc_result':
        case 'dpc_region':
        // Cluster telemetry — consumed by the StatusBar DaskMonitor HUD.
        case 'dask_stats':
        // Read-throughput readout — consumed by the StatusBar IoThroughput HUD.
        case 'io_throughput':
          window.dispatchEvent(new CustomEvent(`spyde:${msg.type}`, { detail: msg }))
          break

        case 'progress': {
          // Heavy backend actions (movie export, …) stream progress via
          // emit_progress {done,total,label}. Surface it through the EXISTING
          // StatusBar busy/status line (LOADING), so we reuse the app's one
          // progress affordance instead of building a new bar. done>=total (or
          // total<=0) clears the busy state. Also re-broadcast as a CustomEvent
          // so a wizard can show a % in its own footer.
          const done = Number(msg.done ?? 0)
          const total = Number(msg.total ?? 0)
          const label = String(msg.label ?? '')
          const busy = total > 0 && done < total
          const pct = total > 0 ? Math.round((done / total) * 100) : 0
          const text = label ? (total > 0 ? `${label} (${pct}%)` : label) : ''
          dispatch({ type: 'LOADING', busy, text })
          window.dispatchEvent(new CustomEvent('spyde:progress', { detail: msg }))
          break
        }
      }
    }

    const disposeMessage = window.electron.onMessage(handleMessage)

    // Test-only window hooks (Playwright e2e). NEVER attached in a packaged
    // production build: a production renderer must not expose a generic message
    // injector / state inspectors on `window`. Gated TRUE in dev (`npm run dev`)
    // and under the e2e (which launches the BUILT bundle by path → app.isPackaged
    // is false → preload's isPackaged is false), FALSE only in `npm run dist`.
    const testHooksEnabled = import.meta.env.DEV || !window.electron?.isPackaged
    if (testHooksEnabled) {
      // Expose test injection hook for Playwright tests
      window._spyde_test_inject = handleMessage

      // Test hook: return the parsed overlay widgets of a figure's latest panel
      // state, so a test can post the awi_event a selector would post (without
      // pixel-perfect mouse grabbing of a tiny handle).
      window._spyde_test_widgets = (figId: string) => {
        const states = latestStates.current.get(figId)
        if (!states) return []
        const widgets: Array<{ panel_id: string; id: string; type: string; data: Record<string, unknown> }> = []
        for (const [key, value] of states) {
          if (!key.startsWith('panel_') || !key.endsWith('_json')) continue
          const panelId = key.slice('panel_'.length, -'_json'.length)
          try {
            const d = JSON.parse(value as string)
            for (const w of d.overlay_widgets ?? []) {
              widgets.push({ panel_id: panelId, id: w.id, type: w.type, data: w })
            }
          } catch { /* */ }
        }
        return widgets
      }

      // Test hook: the raw panel-state JSON of a figure, so a spec can read the
      // LINES as well as the widgets. Checking that a drag handle sits ON its
      // component's curve needs both, and a screenshot cannot tell "on the
      // curve" from "a few pixels off it".
      window._spyde_test_panel_json = (figId: string) => {
        const states = latestStates.current.get(figId)
        if (!states) return []
        const out: string[] = []
        for (const [key, value] of states) {
          if (key.startsWith('panel_') && key.endsWith('_json')) out.push(value as string)
        }
        return out
      }

      // Test hook: return the authoritative report doc (read-only snapshot) so a
      // Playwright spec can read backend-assigned ids that never surface in the
      // DOM — e.g. a figure-level annotation's `id` (needed to inject a
      // figure-marker drag pointer_up). Reads the ref so it's always current.
      window._spyde_test_report = () => {
        try { return JSON.parse(JSON.stringify(reportRef.current)) } catch { return null }
      }

      // Test hook: a cheap signature of a figure's latest image data (length +
      // sampled chars of the base64 image), so a test can detect that the image
      // actually changed without decoding the canvas.
      window._spyde_test_image_sig = (figId: string) => {
        const states = latestStates.current.get(figId)
        if (!states) return ''
        // Hash the FULL base64 image so a change anywhere in the frame is detected
        // (a prefix slice misses bright pixels deeper in the buffer).
        const hash = (s: string) => {
          let h = 5381
          for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0
          return h
        }
        let sig = ''
        for (const [key, value] of states) {
          if (!key.startsWith('panel_')) continue
          try {
            const d = JSON.parse(value as string)
            const b64 = d.image_b64 || ''
            sig += `${key}:${b64.length}:${hash(b64)}|`
          } catch { /* */ }
        }
        return sig
      }
    }

    const disposeStream = window.electron.onStream((text, kind) => {
      dispatch({ type: 'STREAM', text, kind })
    })

    // File → Load Stack… opens the in-app reorderable StackDialog.
    const disposeStackDialog = window.electron.onOpenStackDialog(() =>
      setStackDialogOpen(true),
    )

    // Help → Check for Updates… / GPU Status… (native menu; MenuBar.tsx's HTML
    // dropdown on Windows/Linux calls openUpdateDialog/openGpuStatusDialog directly).
    const disposeUpdateDialog = window.electron.onOpenUpdateDialog(() =>
      setUpdateDialogOpen(true),
    )
    const disposeGpuStatusDialog = window.electron.onOpenGpuStatusDialog(() =>
      setGpuStatusDialogOpen(true),
    )
    const disposeGpuHelpDialog = window.electron.onOpenGpuHelpDialog(() =>
      setGpuHelpDialogOpen(true),
    )
    const disposeReportDialog = window.electron.onOpenReportDialog(() =>
      setReportDialogOpen(true),
    )

    // Forward iframe events to Python
    const onMessage = (e: MessageEvent) => {
      if (e.data?.type === 'awi_event' && e.data.figId) {
        window.electron.figureEvent(e.data.figId, e.data.data)
        // Also mirror the event to renderer components as a CustomEvent so UI
        // like the report cell's floating annotation popover can react to
        // widget clicks WITHOUT touching the raw message channel. The payload
        // is the parsed event dict (event_json is a JSON string).
        try {
          const parsed = typeof e.data.data === 'string'
            ? JSON.parse(e.data.data) : e.data.data
          window.dispatchEvent(new CustomEvent('spyde:figure_event', {
            detail: { figId: e.data.figId, event: parsed },
          }))
        } catch { /* malformed event_json — backend still got the raw form */ }
      }
    }
    window.addEventListener('message', onMessage)
    return () => {
      // MUST remove the ipcRenderer listeners (StrictMode runs this effect twice
      // in dev; without cleanup the second run stacks a duplicate listener →
      // every message dispatched twice, growing over time → doubled logs + lag).
      disposeMessage?.()
      disposeStream?.()
      // Drop any coalesced log batch still waiting on its rAF/timeout — a
      // StrictMode remount would otherwise flush into the new reducer instance.
      if (logRaf.current != null) { cancelAnimationFrame(logRaf.current); logRaf.current = null }
      if (logTimer.current != null) { clearTimeout(logTimer.current); logTimer.current = null }
      pendingLogs.current = []
      disposeStackDialog?.()
      disposeUpdateDialog?.()
      disposeGpuStatusDialog?.()
      disposeGpuHelpDialog?.()
      disposeReportDialog?.()
      window.removeEventListener('message', onMessage)
      if (testHooksEnabled) {
        delete window._spyde_test_inject
        delete window._spyde_test_widgets
        delete window._spyde_test_panel_json
        delete window._spyde_test_report
        delete window._spyde_test_image_sig
      }
    }
  }, [])

  const sendAction = (
    action: string,
    payload: Record<string, unknown> = {},
    windowId?: number,
  ) => window.electron.action(action, payload, windowId)

  // Snapshot harvest: when the backend requests PNGs before a save
  // (report_need_snapshots), grab one per cell via the export protocol and
  // reply with report_snapshots {token, images}. Per-cell PNGs are gathered in
  // parallel (each self-times-out at 1.5 s in requestFigurePng); we send
  // whatever succeeded — the backend has its own baked-PNG fallback, so a save
  // must never block. sendAction is recreated each render, so route through a
  // ref to keep this effect's identity stable (it must attach ONCE).
  const sendActionRef = useRef(sendAction)
  sendActionRef.current = sendAction
  // (reportFiguresRef is declared above, before the message effect, so both it
  // and this harvest effect share the same ref.)
  useEffect(() => {
    const onNeed = async (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        token?: string; cells?: Array<{ cell_id: string; fig_id: string }>
      }
      const token = detail?.token
      const cells = detail?.cells ?? []
      if (!token) return
      const images: Record<string, string> = {}
      await Promise.all(cells.map(async ({ cell_id }) => {
        // The backend's `fig_id` field is the CELL id (its state() ships
        // fig_id === cell.id). The real iframe key is the anyplotlib figId of
        // the report figure for this cell — resolve it via reportFigures.
        const realFigId = reportFiguresRef.current.get(cell_id)?.figId
        if (!realFigId) return
        const dataUrl = await requestFigurePng(realFigId, 1500)
        if (dataUrl) images[cell_id] = dataUrl
      }))
      sendActionRef.current('report_snapshots', { token, images })
    }
    window.addEventListener('spyde:report_need_snapshots', onNeed)
    return () => window.removeEventListener('spyde:report_need_snapshots', onNeed)
  }, [requestFigurePng])

  // Global drag-kind tracking: a window-level dragstart listener inspects the
  // drag's MIME TYPES (readable during a drag; the payload is not) to classify
  // the in-flight drag. A window/figure pill carries the spyde window MIMEs →
  // dragKind='window', which the MDI overlay-drop shield + report compose-drop
  // shield key on. Cleared on dragend/drop so the shields unmount as soon as the
  // drag ends (whether or not it landed on a window).
  //
  // BUBBLE (not capture) phase deliberately: the drag SOURCE is a Pill whose own
  // onDragStart is what stamps the MIMEs into the DataTransfer. React dispatches
  // that at its root container, so a native BUBBLE listener on `window` (above
  // the React root) runs AFTER it — by which time dataTransfer.types is
  // populated. A capture-phase window listener would run BEFORE the Pill's
  // setData and see an empty types list. StrictMode-safe: added in an effect
  // with cleanup, so a re-mount never stacks duplicate listeners.
  useEffect(() => {
    const WINDOW_TYPES = [WINDOW_DRAG_MIME, FIGURE_DRAG_MIME]
    const classify = (e: DragEvent): 'window' | null => {
      const types = e.dataTransfer?.types
      if (!types) return null
      const arr = Array.from(types)
      // Only classify as 'window' when a window/figure MIME is present; leave it
      // unchanged (return null → no-op below) for drags that carry NO spyde types
      // at all yet (some browsers withhold custom types on dragover) so a bare
      // file/text dragover doesn't clobber an active window drag.
      return WINDOW_TYPES.some(t => arr.includes(t)) ? 'window' : null
    }
    const onDragStart = (e: DragEvent) => {
      const k = classify(e)
      dlog('2.classify/dragstart', {
        types: e.dataTransfer ? Array.from(e.dataTransfer.types) : null,
        kind: k,
        target: (e.target as HTMLElement)?.getAttribute?.('data-testid') ?? null,
      })
      setDragKind(k)
    }
    // dragover is a belt-and-braces re-classify: if dragstart's read raced the
    // source's setData (ordering across the React-root vs window listeners), the
    // first dragover — where types are reliably populated — sets it. Only ever
    // PROMOTES to 'window'; never clears (clearing is dragend/drop's job).
    let promoted = false
    // Trail of WHAT the drag is actually over. A drop that never fires means no
    // dragover preventDefault()ed under the cursor — and the only way to know
    // why is to see which element was receiving them. Logged on TARGET CHANGE
    // (dragover fires ~60 Hz) with the topmost hit-test result alongside, since
    // an out-of-process iframe swallows the event entirely and never appears as
    // a target here at all.
    let lastPath = ''
    const describe = (el: Element | null): string => {
      if (!el) return 'null'
      const t = el.getAttribute?.('data-testid')
      if (t) return t
      const tag = el.tagName.toLowerCase()
      return tag === 'iframe' ? 'IFRAME' : tag
    }
    const onDragOver = (e: DragEvent) => {
      if (classify(e) === 'window') {
        if (!promoted) {
          promoted = true
          dlog('2b.classify/first-dragover', {
            types: e.dataTransfer ? Array.from(e.dataTransfer.types) : null,
          })
        }
        setDragKind(prev => (prev === 'window' ? prev : 'window'))
      }
    }
    // CAPTURE phase for the trail: the shield calls stopPropagation() when it
    // accepts, so a bubble-phase listener goes silent exactly when things are
    // working and can't distinguish that from the event never arriving.
    const onDragOverCapture = (e: DragEvent) => {
      const tgt = describe(e.target as Element)
      const top = describe(document.elementFromPoint(e.clientX, e.clientY))
      const path = `${tgt}|${top}`
      if (path !== lastPath) {
        lastPath = path
        dlog('2c.dragover-over', { target: tgt, topmostAtPoint: top })
      }
    }
    // Clearing the in-process payload stash here (not in the drag source's own
    // dragend) keeps it alive for the whole drag — including the drop, which
    // fires BEFORE dragend — while guaranteeing it can never leak into a later,
    // unrelated drop.
    const clear = (e: Event) => {
      dlog(`6.end/${e.type}`, {
        target: (e.target as HTMLElement)?.getAttribute?.('data-testid') ?? null,
      })
      dragDumpToConsole(e.type)
      promoted = false
      setDragKind(null)
      stashWindowDrag(null)
    }
    window.addEventListener('dragstart', onDragStart)
    window.addEventListener('dragover', onDragOver)
    window.addEventListener('dragover', onDragOverCapture, true)
    window.addEventListener('dragend', clear)
    window.addEventListener('drop', clear)
    return () => {
      window.removeEventListener('dragstart', onDragStart)
      window.removeEventListener('dragover', onDragOver)
      window.removeEventListener('dragover', onDragOverCapture, true)
      window.removeEventListener('dragend', clear)
      window.removeEventListener('drop', clear)
    }
  }, [])

  const setActiveWindow = (windowId: number) => {
    dispatch({ type: 'SET_ACTIVE', windowId })
    // Tell the backend too, so window-less actions (e.g. the File→Save menu,
    // which can't know the focused window) can resolve the active plot.
    window.electron.action('set_active', { window_id: windowId }, windowId)
  }

  const clearNavShapePrompt = () => dispatch({ type: 'NAV_SHAPE_PROMPT', prompt: null })
  const openStackDialog = () => setStackDialogOpen(true)
  const closeStackDialog = () => setStackDialogOpen(false)
  const openUpdateDialog = () => setUpdateDialogOpen(true)
  const closeUpdateDialog = () => setUpdateDialogOpen(false)
  const openGpuStatusDialog = () => setGpuStatusDialogOpen(true)
  const closeGpuStatusDialog = () => setGpuStatusDialogOpen(false)
  const openGpuHelpDialog = () => setGpuHelpDialogOpen(true)
  const closeGpuHelpDialog = () => setGpuHelpDialogOpen(false)
  const openReportDialog = () => setReportDialogOpen(true)
  const closeReportDialog = () => setReportDialogOpen(false)

  return (
    <SpyDEContext.Provider value={{
      state, iframeRefs, latestStates, sendAction, setActiveWindow, replayState,
      requestFigurePng, clearNavShapePrompt,
      stackDialogOpen, openStackDialog, closeStackDialog,
      updateDialogOpen, openUpdateDialog, closeUpdateDialog,
      gpuStatusDialogOpen, openGpuStatusDialog, closeGpuStatusDialog,
      gpuHelpDialogOpen, openGpuHelpDialog, closeGpuHelpDialog,
      reportDialogOpen, openReportDialog, closeReportDialog,
      tileWindowsRef, dragKind,
    }}>
      {children}
      {state.envSetup && !state.backendExited && (
        <EnvSetupOverlay setup={state.envSetup} />
      )}
      {state.backendExited && (
        <BackendExitedOverlay
          code={state.backendExited.code}
          reason={state.backendExited.reason}
          streamLines={state.streamLines}
        />
      )}
    </SpyDEContext.Provider>
  )
}

// Blocking, non-dismissable overlay shown when the Python analysis backend dies
// (or fails to build on a packaged first launch, which passes `reason`). Without
// this the UI silently freezes (every sendAction no-ops). It now embeds the last
// raw stdout/stderr lines the backend/uv emitted — that captured text was
// previously written to `streamLines` and NEVER rendered, so a startup crash left
// the user with an empty Log panel and no clue. The overlay is self-diagnosing.
function BackendExitedOverlay({ code, reason, streamLines }: {
  code: number | null
  reason?: string
  streamLines: Array<{ text: string; kind: 'stdout' | 'stderr' }>
}) {
  // Last ~40 lines — enough to show a traceback / uv error without a wall of text.
  const tail = streamLines.slice(-40)
  const isSetupFailure = Boolean(reason)
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 99999,
      background: 'rgba(0,0,0,0.78)', display: 'flex',
      alignItems: 'center', justifyContent: 'center', userSelect: 'text',
      padding: 24,
    }} data-testid="backend-exited-overlay">
      <div style={{
        maxWidth: 640, width: '100%', maxHeight: '90%', overflow: 'hidden',
        display: 'flex', flexDirection: 'column',
        padding: '26px 30px', borderRadius: 10,
        background: '#1e1e2e', border: '1px solid #f38ba8',
        color: '#cdd6f4', fontFamily: 'system-ui, sans-serif',
      }}>
        <div style={{ fontSize: 18, fontWeight: 600, color: '#f38ba8', marginBottom: 10 }}>
          {isSetupFailure ? 'Python environment setup failed' : 'Analysis backend stopped'}
        </div>
        <div style={{ fontSize: 14, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
          {reason ?? (
            <>
              The Python process powering SpyDE exited
              {code != null ? <> (exit code <code>{code}</code>)</> : null}.
              Compute and file operations are unavailable. Please restart SpyDE.
              The captured output below shows why.
            </>
          )}
        </div>
        <div style={{
          marginTop: 14, fontSize: 11, color: '#a6adc8', textTransform: 'uppercase',
          letterSpacing: 0.5,
        }}>
          Raw output {tail.length ? `(last ${tail.length} lines)` : ''}
        </div>
        <div
          data-testid="backend-exited-raw"
          style={{
            marginTop: 6, flex: 1, minHeight: 60, maxHeight: 260, overflow: 'auto',
            background: '#11111b', border: '1px solid #313244', borderRadius: 6,
            padding: '8px 10px',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 11.5, lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}
        >
          {tail.length === 0
            ? <span style={{ color: '#6c7086', fontStyle: 'italic' }}>No output was captured.</span>
            : tail.map((l, i) => (
                <div key={i} style={{ color: l.kind === 'stderr' ? '#f9c0c9' : '#a6adc8' }}>
                  {l.text.replace(/\n$/, '')}
                </div>
              ))}
        </div>
      </div>
    </div>
  )
}

export function useSpyDE(): SpyDEContextValue {
  const ctx = useContext(SpyDEContext)
  if (!ctx) throw new Error('useSpyDE must be used inside SpyDEProvider')
  return ctx
}
