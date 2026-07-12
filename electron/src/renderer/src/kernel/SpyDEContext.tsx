/**
 * SpyDEContext.tsx — React context for the SpyDE Python backend state.
 *
 * Listens for PLOTAPP: messages, maintains the window/figure registry,
 * toolbar configs, and status, and re-renders the MDI when things change.
 */
import React, {
  createContext, useContext, useEffect, useReducer, useRef, useState,
} from 'react'
import { asPlotAppMessage } from './protocol'
import type { ReportDocState, ReportCell } from './protocol'
import { WINDOW_DRAG_MIME, FIGURE_DRAG_MIME } from './dnd'

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
export interface Composition { elements: string[]; percentages: Record<string, number> }
export interface Histogram {
  counts: number[]
  edges: number[]
  vmin: number
  vmax: number
  threshold?: number | null   // dotted marker line (Find-Vectors detector threshold)
}

/** One application-log record streamed from the Python backend. */
export interface LogEntry { level: string; name: string; area?: string; msg: string; time: number }

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
}
/** The named navigators a navigator window offers (its top chip strip). */
export interface NavigatorOptions { names: string[]; current?: string | null }
export interface SubItem { name: string; color: string; vtype?: string; calculation?: string }
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

interface State {
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
  histograms: Map<number, Histogram>
  selectors: Map<number, SelectorInfo>
  signalTrees: Map<number, TreeNode>
  signalTreeActive: Map<number, number>   // windowId → active node signal_id
  navigatorOptions: Map<number, NavigatorOptions>   // navigator windowId → named navigators
  axes: Map<number, AxisRow[]>
  composition: Map<number, Composition>     // windowId → sample elements + percentages
  activeActions: Map<number, Set<string>>   // windowId → action names with live output
  subItems: Map<number, Map<string, SubItem[]>>  // windowId → action → dynamic chips
  status: string
  ready: boolean
  dashboardUrl: string | null
  activeWindowId: number | null
  streamLines: Array<{ text: string; kind: 'stdout' | 'stderr' }>
  logEntries: LogEntry[]          // application-log records (the log panel)
  logLevel: string                // current backend verbosity (DEBUG…CRITICAL)
  navShapePrompt: NavShapePrompt | null   // pending scan-shape/step-size dialog
  loading: { busy: boolean; text: string }   // long file-read busy indicator
  signalTypes: Map<number, { current: string; options: string[] }>   // windowId → signal-type info
  backendExited: { code: number | null } | null   // set when the Python sidecar dies; surfaces a blocking banner
  playback: { playing: boolean; speed: number; loop: boolean }   // movie playback clock (session-wide)
  consoleResult: ConsoleResult | null       // last-executed cell (the ConsoleBar echo strip)
  consoleVars: ConsoleVarEntry[]            // live variable table (chips + signal-ref resolution)
  consoleCompletions: { completeId: number; matches: string[] } | null
  consolePreview: ConsolePreviewResult | null   // last live-preview reply (the eye-toggled slot)
}

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
  | { type: 'READY'; dashboardUrl?: string }
  | { type: 'STATUS'; text: string }
  | { type: 'FIGURE'; windowId: number; figId: string; fileUrl: string | null; title: string; isNavigator: boolean; aspect?: number; view?: string; viewLabel?: string; viewKind?: string; strainComponents?: string[] }
  | { type: 'WINDOW_TITLE'; windowId: number; title: string }
  | { type: 'TOOLBAR_CONFIG'; windowId: number; plotId: number; actions: ToolbarAction[] }
  | { type: 'WINDOW_VISIBILITY'; windowId: number; visible: boolean }
  | { type: 'WINDOW_CLOSED'; windowId: number }
  | { type: 'SET_ACTIVE'; windowId: number }
  | { type: 'METADATA'; windowIds: number[]; metadata: MetadataDict }
  | { type: 'COMPOSITION'; windowIds: number[]; composition: Composition }
  | { type: 'AXES'; windowIds: number[]; axes: AxisRow[] }
  | { type: 'ACTION_ACTIVE'; windowId: number; name: string; active: boolean }
  | { type: 'SUB_ITEM'; windowId: number; action: string; name: string; color: string; vtype?: string; calculation?: string; active: boolean }
  | { type: 'HISTOGRAM'; windowId: number; histogram: Histogram }
  | { type: 'NAV_SHAPE_PROMPT'; prompt: NavShapePrompt | null }
  | { type: 'LOADING'; busy: boolean; text: string }
  | { type: 'SIGNAL_TYPE'; windowIds: number[]; current: string; options: string[] }
  | { type: 'SELECTOR_INFO'; info: SelectorInfo }
  | { type: 'SIGNAL_TREE'; windowId: number; tree: TreeNode; activeSignalId?: number }
  | { type: 'NAVIGATOR_OPTIONS'; windowId: number; names: string[]; current?: string | null }
  | { type: 'STREAM'; text: string; kind: 'stdout' | 'stderr' }
  | { type: 'LOG'; entry: LogEntry }
  | { type: 'LOG_BACKFILL'; entries: LogEntry[] }
  | { type: 'LOG_LEVEL'; level: string }
  | { type: 'BACKEND_EXITED'; code: number | null }
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

    case 'STATUS':
      return { ...state, status: action.text }

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
      return {
        ...state,
        windows: newWindows,
        selectors,
        histograms: drop(state.histograms),
        metadata: drop(state.metadata),
        axes: drop(state.axes),
        composition: drop(state.composition),
        signalTrees: drop(state.signalTrees),
        signalTreeActive: drop(state.signalTreeActive),
        navigatorOptions: drop(state.navigatorOptions),
        activeActions: drop(state.activeActions),
        subItems: drop(state.subItems),
        activeWindowId,
      }
    }

    case 'SET_ACTIVE':
      return { ...state, activeWindowId: action.windowId }

    case 'METADATA': {
      const metadata = new Map(state.metadata)
      for (const wid of action.windowIds) metadata.set(wid, action.metadata)
      return { ...state, metadata }
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

    case 'ACTION_ACTIVE': {
      const activeActions = new Map(state.activeActions)
      const set = new Set(activeActions.get(action.windowId) ?? [])
      if (action.active) set.add(action.name)
      else set.delete(action.name)
      activeActions.set(action.windowId, set)
      return { ...state, activeActions }
    }

    case 'SUB_ITEM': {
      const subItems = new Map(state.subItems)
      const byAction = new Map(subItems.get(action.windowId) ?? new Map<string, SubItem[]>())
      const list = (byAction.get(action.action) ?? []).filter(i => i.name !== action.name)
      if (action.active) list.push({
        name: action.name, color: action.color,
        vtype: action.vtype, calculation: action.calculation,
      })
      byAction.set(action.action, list)
      subItems.set(action.windowId, byAction)
      return { ...state, subItems }
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

    case 'REPORT_STATE':
      return { ...state, report: action.report }

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
      if (action.activeSignalId != null) signalTreeActive.set(action.windowId, action.activeSignalId)
      return { ...state, signalTrees, signalTreeActive }
    }

    case 'STREAM':
      return {
        ...state,
        streamLines: [...state.streamLines.slice(-500), { text: action.text, kind: action.kind }],
      }

    case 'LOG':
      return { ...state, logEntries: [...state.logEntries.slice(-999), action.entry] }

    case 'LOG_BACKFILL':
      return { ...state, logEntries: action.entries.slice(-1000) }

    case 'LOG_LEVEL':
      return { ...state, logLevel: action.level }

    case 'NAV_SHAPE_PROMPT':
      return { ...state, navShapePrompt: action.prompt }

    case 'LOADING':
      return { ...state, loading: { busy: action.busy, text: action.text } }

    case 'SIGNAL_TYPE': {
      const signalTypes = new Map(state.signalTypes)
      for (const wid of action.windowIds)
        signalTypes.set(wid, { current: action.current, options: action.options })
      return { ...state, signalTypes }
    }

    case 'BACKEND_EXITED':
      return { ...state, backendExited: { code: action.code }, ready: false, status: 'Backend stopped' }

    default:
      return state
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
  replayState: (figId: string) => void
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
    composition: new Map(),
    histograms: new Map(),
    selectors: new Map(),
    signalTrees: new Map(),
    signalTreeActive: new Map(),
    navigatorOptions: new Map(),
    axes: new Map(),
    activeActions: new Map(),
    subItems: new Map(),
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
    playback: { playing: false, speed: 1, loop: false },
    consoleResult: null,
    consoleVars: [],
    consoleCompletions: null,
    consolePreview: null,
  })

  const iframeRefs = useRef<Map<string, HTMLIFrameElement>>(new Map())
  const latestStates = useRef<Map<string, Map<string, unknown>>>(new Map())
  // Latest RAW BINARY frame per figure (key → {header, buffer}), mirroring
  // latestStates but for PLOTBIN pushes. Without this, a figure whose FIRST
  // real paint arrives as a binary frame before its iframe's onLoad fires (the
  // common case for a console-created window, which — unlike a
  // navigator-driven one — gets no organic second paint) stays permanently
  // blank: postMessage to an unmounted iframe silently no-ops and, prior to
  // this stash, replayState() had nothing to give it on load. Keeps only the
  // LATEST frame per key (retained across postMessage's transfer, which
  // detaches the original ArrayBuffer) — matches the "latest-wins" paint
  // philosophy used throughout the nav/paint pipeline (see CLAUDE.md).
  const latestBinaryStates = useRef<Map<string, Map<string, { header: unknown; buffer: Uint8Array }>>>(new Map())
  const tileWindowsRef = useRef<(() => void) | null>(null)
  const [stackDialogOpen, setStackDialogOpen] = useState(false)
  const [updateDialogOpen, setUpdateDialogOpen] = useState(false)
  const [gpuStatusDialogOpen, setGpuStatusDialogOpen] = useState(false)
  const [dragKind, setDragKind] = useState<'window' | null>(null)

  // Post every stored state for a figure to its iframe (called on iframe load).
  const replayState = (figId: string) => {
    const iframe = iframeRefs.current.get(figId)
    if (!iframe?.contentWindow) return
    const states = latestStates.current.get(figId)
    if (states) {
      for (const [key, value] of states) {
        iframe.contentWindow.postMessage({ type: 'awi_state', key, value }, '*')
      }
    }
    const binStates = latestBinaryStates.current.get(figId)
    if (binStates) {
      for (const [key, { header, buffer }] of binStates) {
        // Send a COPY (transfer detaches the buffer) — the stash must survive
        // a later iframe remount (e.g. window re-tile / dev StrictMode).
        const copy = buffer.slice()
        iframe.contentWindow.postMessage(
          { type: 'awi_state_binary', key, header, buffer: copy },
          '*',
          [copy.buffer],
        )
      }
    }
  }

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
            { type: 'anyplotlib_export_png', requestId, opts: {} }, '*',
          )
        } catch { finish(null) }
      })
    },
    [],
  )

  // ── Python → Renderer message dispatch ──────────────────────────────────

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
          // The Python sidecar died (synthesised by runner.ts, not a PLOTAPP line).
          // Surface a blocking banner — every sendAction after this no-ops.
          dispatch({ type: 'BACKEND_EXITED', code: msg.code ?? null })
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

        case 'state_update':
          // Forward anyplotlib state to the iframe AND remember it, so it can
          // be replayed if/when the iframe (re)loads after this arrived.
          {
            const figId = msg.fig_id
            const key = msg.key
            if (!latestStates.current.has(figId)) {
              latestStates.current.set(figId, new Map())
            }
            latestStates.current.get(figId)!.set(key, msg.value)
            const iframe = iframeRefs.current.get(figId)
            iframe?.contentWindow?.postMessage(
              { type: 'awi_state', key, value: msg.value },
              '*',
            )
          }
          break

        case 'state_update_binary':
          // A raw image frame (pixels as a Uint8Array, no base64). Post it to the
          // iframe as `awi_state_binary`; the figure ESM uses the ArrayBuffer as
          // the texture/ImageData bytes directly (no atob). ALSO stash a retained
          // copy in latestBinaryStates (mirrors latestStates) so replayState() can
          // recover a figure whose iframe wasn't mounted yet when this arrived —
          // otherwise the postMessage below silently no-ops (no listener) and the
          // frame is lost forever: a static (e.g. console-created) window that
          // never repaints organically stayed permanently blank.
          {
            const figId = msg.fig_id
            const key = msg.key
            const bytes = msg.buffer as Uint8Array
            if (bytes) {
              if (!latestBinaryStates.current.has(figId)) {
                latestBinaryStates.current.set(figId, new Map())
              }
              latestBinaryStates.current.get(figId)!.set(
                key, { header: msg.header, buffer: bytes.slice() },
              )
            }
            const iframe = iframeRefs.current.get(figId)
            iframe?.contentWindow?.postMessage(
              { type: 'awi_state_binary', key, header: msg.header, buffer: bytes },
              '*',
              bytes?.buffer ? [bytes.buffer] : [],   // TRANSFER the ArrayBuffer
            )
          }
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
          })
          break

        case 'axes_info':
          dispatch({
            type: 'AXES',
            windowIds: msg.window_ids ?? [],
            axes: msg.axes ?? [],
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
            },
          })
          break

        case 'nav_shape_prompt':
          dispatch({ type: 'NAV_SHAPE_PROMPT', prompt: msg })
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
            },
          })
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

        case 'report_state':
          // The authoritative report document. Mirrored into state so the
          // sidebar + cells re-render.
          if (msg.report) dispatch({ type: 'REPORT_STATE', report: msg.report as ReportDocState })
          break

        case 'report_saved':
          // Zip written — surface a transient status (the sidebar reads
          // report.dirty for the persistent indicator).
          dispatch({ type: 'STATUS', text: `Report saved: ${msg.path}` })
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
          dispatch({ type: 'LOG', entry: {
            level: String(msg.level), name: String(msg.name),
            msg: String(msg.msg), time: Number(msg.time),
          } })
          break

        case 'log_backfill':
          dispatch({ type: 'LOG_BACKFILL', entries: msg.entries ?? [] })
          break

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
        case 'fv_auto_params':
        case 'fv_models':
        case 'cod_results':
        case 'cod_cif_ready':
        case 'gpu_status_result':
        case 'console_node_bound':
        case 'layers_state':
        case 'repfig_compose_options':
        case 'report_exported':
        case 'mvx_state':
        case 'mvx_done':
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

    // Forward iframe events to Python
    const onMessage = (e: MessageEvent) => {
      if (e.data?.type === 'awi_event' && e.data.figId) {
        window.electron.figureEvent(e.data.figId, e.data.data)
      }
    }
    window.addEventListener('message', onMessage)
    return () => {
      // MUST remove the ipcRenderer listeners (StrictMode runs this effect twice
      // in dev; without cleanup the second run stacks a duplicate listener →
      // every message dispatched twice, growing over time → doubled logs + lag).
      disposeMessage?.()
      disposeStream?.()
      disposeStackDialog?.()
      disposeUpdateDialog?.()
      disposeGpuStatusDialog?.()
      window.removeEventListener('message', onMessage)
      if (testHooksEnabled) {
        delete window._spyde_test_inject
        delete window._spyde_test_widgets
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
  // Mirror reportFigures into a ref so the harvest (a stable-identity effect)
  // reads the LATEST cell→figure map without re-subscribing per figure.
  const reportFiguresRef = useRef(state.reportFigures)
  reportFiguresRef.current = state.reportFigures
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
    const onDragStart = (e: DragEvent) => setDragKind(classify(e))
    // dragover is a belt-and-braces re-classify: if dragstart's read raced the
    // source's setData (ordering across the React-root vs window listeners), the
    // first dragover — where types are reliably populated — sets it. Only ever
    // PROMOTES to 'window'; never clears (clearing is dragend/drop's job).
    const onDragOver = (e: DragEvent) => {
      if (classify(e) === 'window') setDragKind(prev => (prev === 'window' ? prev : 'window'))
    }
    const clear = () => setDragKind(null)
    window.addEventListener('dragstart', onDragStart)
    window.addEventListener('dragover', onDragOver)
    window.addEventListener('dragend', clear)
    window.addEventListener('drop', clear)
    return () => {
      window.removeEventListener('dragstart', onDragStart)
      window.removeEventListener('dragover', onDragOver)
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

  return (
    <SpyDEContext.Provider value={{
      state, iframeRefs, latestStates, sendAction, setActiveWindow, replayState,
      requestFigurePng, clearNavShapePrompt,
      stackDialogOpen, openStackDialog, closeStackDialog,
      updateDialogOpen, openUpdateDialog, closeUpdateDialog,
      gpuStatusDialogOpen, openGpuStatusDialog, closeGpuStatusDialog,
      tileWindowsRef, dragKind,
    }}>
      {children}
      {state.backendExited && <BackendExitedOverlay code={state.backendExited.code} />}
    </SpyDEContext.Provider>
  )
}

// Blocking, non-dismissable overlay shown when the Python analysis backend dies.
// Without this the UI silently freezes (every sendAction no-ops). Restart is a
// follow-up; for 0.1.0 we make the death visible and tell the user to relaunch.
function BackendExitedOverlay({ code }: { code: number | null }) {
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 99999,
      background: 'rgba(0,0,0,0.78)', display: 'flex',
      alignItems: 'center', justifyContent: 'center', userSelect: 'text',
    }}>
      <div style={{
        maxWidth: 460, padding: '28px 32px', borderRadius: 10,
        background: '#1e1e2e', border: '1px solid #f38ba8',
        color: '#cdd6f4', fontFamily: 'system-ui, sans-serif', textAlign: 'center',
      }}>
        <div style={{ fontSize: 18, fontWeight: 600, color: '#f38ba8', marginBottom: 10 }}>
          Analysis backend stopped
        </div>
        <div style={{ fontSize: 14, lineHeight: 1.5 }}>
          The Python process powering SpyDE exited
          {code != null ? <> (exit code <code>{code}</code>)</> : null}.
          Compute and file operations are unavailable. Please restart SpyDE.
          Check the Log panel for details.
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
