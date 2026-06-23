/**
 * SpyDEContext.tsx — React context for the SpyDE Python backend state.
 *
 * Listens for PLOTAPP: messages, maintains the window/figure registry,
 * toolbar configs, and status, and re-renders the MDI when things change.
 */
import React, {
  createContext, useContext, useEffect, useReducer, useRef, useState,
} from 'react'

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

export interface SelectorInfo { windowId: number; mode: 'crosshair' | 'integrate'; title: string }
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
  metadata: Map<number, MetadataDict>
  histograms: Map<number, Histogram>
  selectors: Map<number, SelectorInfo>
  signalTrees: Map<number, TreeNode>
  signalTreeActive: Map<number, number>   // windowId → active node signal_id
  axes: Map<number, AxisRow[]>
  composition: Map<number, Composition>     // windowId → sample elements + percentages
  ipfKey: Map<number, string>               // windowId → IPF colour-key triangle (PNG data URL)
  strainRings: Map<number, { rings: number[]; selected: number[] }>   // windowId → strain ring selection
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
  | { type: 'TOOLBAR_CONFIG'; windowId: number; plotId: number; actions: ToolbarAction[] }
  | { type: 'WINDOW_VISIBILITY'; windowId: number; visible: boolean }
  | { type: 'WINDOW_CLOSED'; windowId: number }
  | { type: 'SET_ACTIVE'; windowId: number }
  | { type: 'METADATA'; windowIds: number[]; metadata: MetadataDict }
  | { type: 'COMPOSITION'; windowIds: number[]; composition: Composition }
  | { type: 'IPF_KEY'; windowId: number; dataUrl: string }
  | { type: 'STRAIN_RINGS'; windowId: number; rings: number[]; selected: number[] }
  | { type: 'AXES'; windowIds: number[]; axes: AxisRow[] }
  | { type: 'ACTION_ACTIVE'; windowId: number; name: string; active: boolean }
  | { type: 'SUB_ITEM'; windowId: number; action: string; name: string; color: string; vtype?: string; calculation?: string; active: boolean }
  | { type: 'HISTOGRAM'; windowId: number; histogram: Histogram }
  | { type: 'NAV_SHAPE_PROMPT'; prompt: NavShapePrompt | null }
  | { type: 'LOADING'; busy: boolean; text: string }
  | { type: 'SIGNAL_TYPE'; windowIds: number[]; current: string; options: string[] }
  | { type: 'SELECTOR_INFO'; info: SelectorInfo }
  | { type: 'SIGNAL_TREE'; windowId: number; tree: TreeNode; activeSignalId?: number }
  | { type: 'STREAM'; text: string; kind: 'stdout' | 'stderr' }
  | { type: 'LOG'; entry: LogEntry }
  | { type: 'LOG_BACKFILL'; entries: LogEntry[] }
  | { type: 'LOG_LEVEL'; level: string }

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
      // A secondary view figure (e.g. the IPF 3-D explorer, view="3d") must NOT
      // rename the window or flip its navigator flag — those belong to the
      // primary figure.
      if (!action.view) {
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
      return {
        ...state,
        windows: newWindows,
        selectors: drop(state.selectors),
        histograms: drop(state.histograms),
        metadata: drop(state.metadata),
        axes: drop(state.axes),
        composition: drop(state.composition),
        ipfKey: drop(state.ipfKey),
        strainRings: drop(state.strainRings),
        signalTrees: drop(state.signalTrees),
        signalTreeActive: drop(state.signalTreeActive),
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

    case 'IPF_KEY': {
      const ipfKey = new Map(state.ipfKey)
      ipfKey.set(action.windowId, action.dataUrl)
      return { ...state, ipfKey }
    }

    case 'STRAIN_RINGS': {
      const strainRings = new Map(state.strainRings)
      strainRings.set(action.windowId, { rings: action.rings, selected: action.selected })
      return { ...state, strainRings }
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
      const selectors = new Map(state.selectors)
      selectors.set(action.info.windowId, action.info)
      return { ...state, selectors }
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
  clearNavShapePrompt: () => void
  // Load Stack dialog (renderer-only UI state, opened from the File menu).
  stackDialogOpen: boolean
  openStackDialog: () => void
  closeStackDialog: () => void
}

const SpyDEContext = createContext<SpyDEContextValue | null>(null)

export function SpyDEProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(spydeReducer, {
    windows: new Map(),
    figures: new Map(),
    metadata: new Map(),
    composition: new Map(),
    ipfKey: new Map(),
    strainRings: new Map(),
    histograms: new Map(),
    selectors: new Map(),
    signalTrees: new Map(),
    signalTreeActive: new Map(),
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
  })

  const iframeRefs = useRef<Map<string, HTMLIFrameElement>>(new Map())
  const latestStates = useRef<Map<string, Map<string, unknown>>>(new Map())
  const [stackDialogOpen, setStackDialogOpen] = useState(false)

  // Post every stored state for a figure to its iframe (called on iframe load).
  const replayState = (figId: string) => {
    const iframe = iframeRefs.current.get(figId)
    const states = latestStates.current.get(figId)
    if (!iframe?.contentWindow || !states) return
    for (const [key, value] of states) {
      iframe.contentWindow.postMessage({ type: 'awi_state', key, value }, '*')
    }
  }

  // ── Python → Renderer message dispatch ──────────────────────────────────

  useEffect(() => {
    const handleMessage = (msg: Record<string, unknown>) => {
      const t = msg.type as string
      switch (t) {
        case 'ready':
          dispatch({ type: 'READY', dashboardUrl: msg.dashboard as string | undefined })
          break

        case 'status':
          dispatch({ type: 'STATUS', text: msg.text as string })
          break

        case 'error':
          dispatch({ type: 'STATUS', text: `⚠ ${msg.text}` })
          break

        case 'figure': {
          // Normal path: main process wrote the HTML and gave us file_url.
          // Test path: html is injected directly → fall back to a data URL.
          let fileUrl = (msg.file_url as string) ?? null
          if (!fileUrl && msg.html) {
            fileUrl = 'data:text/html;charset=utf-8,' +
              encodeURIComponent(msg.html as string)
          }
          dispatch({
            type: 'FIGURE',
            windowId: msg.window_id as number,
            figId: msg.fig_id as string,
            fileUrl,
            title: (msg.title as string) || 'Plot',
            isNavigator: (msg.is_navigator as boolean) || false,
            aspect: msg.aspect as number | undefined,
            view: msg.view as string | undefined,
            viewLabel: msg.view_label as string | undefined,
            viewKind: msg.view_kind as string | undefined,
            strainComponents: msg.strain_components as string[] | undefined,
          })
          break
        }

        case 'toolbar_config':
          dispatch({
            type: 'TOOLBAR_CONFIG',
            windowId: msg.window_id as number,
            plotId: msg.plot_id as number,
            actions: (msg.toolbar_actions as ToolbarAction[]) || [],
          })
          break

        case 'window_visibility':
          dispatch({
            type: 'WINDOW_VISIBILITY',
            windowId: msg.window_id as number,
            visible: msg.visible as boolean,
          })
          break

        case 'window_closed':
          dispatch({ type: 'WINDOW_CLOSED', windowId: msg.window_id as number })
          break

        case 'state_update':
          // Forward anyplotlib state to the iframe AND remember it, so it can
          // be replayed if/when the iframe (re)loads after this arrived.
          {
            const figId = msg.fig_id as string
            const key = msg.key as string
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

        case 'strain_rings':
          dispatch({
            type: 'STRAIN_RINGS',
            windowId: msg.window_id as number,
            rings: (msg.rings as number[]) ?? [],
            selected: (msg.selected as number[]) ?? [],
          })
          break

        case 'ipf_key':
          dispatch({
            type: 'IPF_KEY',
            windowId: msg.window_id as number,
            dataUrl: msg.data_url as string,
          })
          break

        case 'composition':
          dispatch({
            type: 'COMPOSITION',
            windowIds: (msg.window_ids as number[]) ?? [],
            composition: {
              elements: (msg.elements as string[]) ?? [],
              percentages: (msg.percentages as Record<string, number>) ?? {},
            },
          })
          break

        case 'metadata':
          dispatch({
            type: 'METADATA',
            windowIds: (msg.window_ids as number[]) ?? [],
            metadata: (msg.metadata as MetadataDict) ?? {},
          })
          break

        case 'axes_info':
          dispatch({
            type: 'AXES',
            windowIds: (msg.window_ids as number[]) ?? [],
            axes: (msg.axes as AxisRow[]) ?? [],
          })
          break

        case 'action_active':
          dispatch({
            type: 'ACTION_ACTIVE',
            windowId: msg.window_id as number,
            name: msg.name as string,
            active: msg.active as boolean,
          })
          break

        case 'sub_item':
          dispatch({
            type: 'SUB_ITEM',
            windowId: msg.window_id as number,
            action: msg.action as string,
            name: msg.name as string,
            color: (msg.color as string) ?? '#89b4fa',
            vtype: msg.vtype as string | undefined,
            calculation: msg.calculation as string | undefined,
            active: msg.active as boolean,
          })
          break

        case 'histogram':
          dispatch({
            type: 'HISTOGRAM',
            windowId: msg.window_id as number,
            histogram: {
              counts: (msg.counts as number[]) ?? [],
              edges: (msg.edges as number[]) ?? [],
              vmin: msg.vmin as number,
              vmax: msg.vmax as number,
              threshold: (msg.threshold as number | null) ?? null,
            },
          })
          break

        case 'nav_shape_prompt':
          dispatch({ type: 'NAV_SHAPE_PROMPT', prompt: msg as unknown as NavShapePrompt })
          break

        case 'loading':
          dispatch({ type: 'LOADING', busy: Boolean(msg.busy), text: String(msg.text ?? '') })
          break

        case 'signal_type_info':
          dispatch({
            type: 'SIGNAL_TYPE',
            windowIds: (msg.window_ids as number[]) ?? [],
            current: String(msg.current ?? ''),
            options: (msg.options as string[]) ?? [],
          })
          break

        case 'selector_info':
          dispatch({
            type: 'SELECTOR_INFO',
            info: {
              windowId: msg.window_id as number,
              mode: (msg.mode as 'crosshair' | 'integrate') ?? 'crosshair',
              title: (msg.title as string) ?? 'Navigator',
            },
          })
          break

        case 'signal_tree':
          if (msg.tree) {
            dispatch({
              type: 'SIGNAL_TREE',
              windowId: msg.window_id as number,
              tree: msg.tree as TreeNode,
              activeSignalId: msg.active_signal_id as number | undefined,
            })
          }
          break

        case 'dask_ready':
          dispatch({ type: 'READY', dashboardUrl: msg.dashboard as string | undefined })
          break

        case 'log':
          dispatch({ type: 'LOG', entry: {
            level: String(msg.level), name: String(msg.name),
            msg: String(msg.msg), time: Number(msg.time),
          } })
          break

        case 'log_backfill':
          dispatch({ type: 'LOG_BACKFILL', entries: (msg.entries as LogEntry[]) ?? [] })
          break

        case 'log_level':
          dispatch({ type: 'LOG_LEVEL', level: String(msg.level) })
          break

        // Wizard-scoped events (live fit readout + library-ready). Re-broadcast
        // as DOM CustomEvents so the relevant caret component can subscribe
        // without threading them through the global reducer.
        case 'vom_fit':
        case 'vom_library_ready':
        case 'om_library_ready':
        case 'fv_auto_params':
        case 'cod_results':
        case 'cod_cif_ready':
          window.dispatchEvent(new CustomEvent(`spyde:${t}`, { detail: msg }))
          break
      }
    }

    const disposeMessage = window.electron.onMessage(handleMessage)

    // Expose test injection hook for Playwright tests
    ;(window as Record<string, unknown>)['_spyde_test_inject'] = handleMessage

    // Test hook: return the parsed overlay widgets of a figure's latest panel
    // state, so a test can post the awi_event a selector would post (without
    // pixel-perfect mouse grabbing of a tiny handle).
    ;(window as Record<string, unknown>)['_spyde_test_widgets'] = (figId: string) => {
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
    ;(window as Record<string, unknown>)['_spyde_test_image_sig'] = (figId: string) => {
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

    const disposeStream = window.electron.onStream((text, kind) => {
      dispatch({ type: 'STREAM', text, kind })
    })

    // File → Load Stack… opens the in-app reorderable StackDialog.
    const disposeStackDialog = window.electron.onOpenStackDialog(() =>
      setStackDialogOpen(true),
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
      window.removeEventListener('message', onMessage)
      delete (window as Record<string, unknown>)['_spyde_test_inject']
    }
  }, [])

  const sendAction = (
    action: string,
    payload: Record<string, unknown> = {},
    windowId?: number,
  ) => window.electron.action(action, payload, windowId)

  const setActiveWindow = (windowId: number) => {
    dispatch({ type: 'SET_ACTIVE', windowId })
    // Tell the backend too, so window-less actions (e.g. the File→Save menu,
    // which can't know the focused window) can resolve the active plot.
    window.electron.action('set_active', { window_id: windowId }, windowId)
  }

  const clearNavShapePrompt = () => dispatch({ type: 'NAV_SHAPE_PROMPT', prompt: null })
  const openStackDialog = () => setStackDialogOpen(true)
  const closeStackDialog = () => setStackDialogOpen(false)

  return (
    <SpyDEContext.Provider value={{ state, iframeRefs, latestStates, sendAction, setActiveWindow, replayState, clearNavShapePrompt, stackDialogOpen, openStackDialog, closeStackDialog }}>
      {children}
    </SpyDEContext.Provider>
  )
}

export function useSpyDE(): SpyDEContextValue {
  const ctx = useContext(SpyDEContext)
  if (!ctx) throw new Error('useSpyDE must be used inside SpyDEProvider')
  return ctx
}
