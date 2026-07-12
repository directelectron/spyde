/**
 * protocol.ts — typed PLOTAPP IPC message protocol (Python backend → renderer).
 *
 * The Python backend emits JSON messages over stdout (the `PLOTAPP:` protocol
 * from `anyplotlib._electron`); the Electron main process relays them to the
 * renderer via `window.electron.onMessage`. Each message is discriminated by its
 * `type` field. `PlotAppMessage` below is the discriminated union the renderer's
 * dispatcher (`SpyDEContext.tsx`) switches over, so individual field reads are
 * narrowed by `msg.type` instead of cast one-by-one.
 *
 * NOTE: this types the SHAPE the renderer relies on; every variant also carries
 * an index signature (via `MsgBase`) so accessing a not-yet-typed field is still
 * allowed (it surfaces as `unknown`) rather than a compile error — keeping this a
 * proportionate type-safety pass, not a full schema. A `backend_exited` message
 * is synthesised by `runner.ts`, not a real PLOTAPP line.
 */
import type {
  ToolbarAction,
  MetadataDict,
  AxisRow,
  TreeNode,
  LogEntry,
  NavShapePrompt,
} from './SpyDEContext'

/** Any field not explicitly modelled is still readable (as `unknown`). */
interface MsgBase {
  [k: string]: unknown
}

export interface ReadyMessage extends MsgBase {
  type: 'ready' | 'dask_ready'
  dashboard?: string
}

export interface StatusMessage extends MsgBase {
  type: 'status'
  text: string
}

export interface ErrorMessage extends MsgBase {
  type: 'error'
  text: string
}

export interface BackendExitedMessage extends MsgBase {
  type: 'backend_exited'
  code: number | null
}

export interface FigureMessage extends MsgBase {
  type: 'figure'
  window_id: number
  fig_id: string
  file_url?: string | null
  html?: string
  title?: string
  is_navigator?: boolean
  aspect?: number
  view?: string
  view_label?: string
  view_kind?: string
  strain_components?: string[]
  /** When present + === "report", this figure belongs to a report figure cell
   *  (routed to `reportFigures`, keyed by `cell_id`) — NOT the MDI windows. */
  host?: string
  /** The report cell id owning this figure (only set when host === "report"). */
  cell_id?: string
}

export interface ToolbarConfigMessage extends MsgBase {
  type: 'toolbar_config'
  window_id: number
  plot_id: number
  toolbar_actions?: ToolbarAction[]
}

export interface WindowVisibilityMessage extends MsgBase {
  type: 'window_visibility'
  window_id: number
  visible: boolean
}

export interface WindowClosedMessage extends MsgBase {
  type: 'window_closed'
  window_id: number
}

/** A lightweight rename — updates the header [Name] of every listed window's
 *  tree WITHOUT re-emitting the figure (no iframe reload). */
export interface WindowTitleMessage extends MsgBase {
  type: 'window_title'
  window_ids?: number[]
  title?: string
}

export interface StateUpdateMessage extends MsgBase {
  type: 'state_update'
  fig_id: string
  key: string
  value: unknown
}

export interface StateUpdateBinaryMessage extends MsgBase {
  type: 'state_update_binary'
  fig_id: string
  key: string
  header: Record<string, unknown>
  buffer: Uint8Array
}

export interface CompositionMessage extends MsgBase {
  type: 'composition'
  window_ids?: number[]
  elements?: string[]
  percentages?: Record<string, number>
}

export interface MetadataMessage extends MsgBase {
  type: 'metadata'
  window_ids?: number[]
  metadata?: MetadataDict
}

export interface AxesInfoMessage extends MsgBase {
  type: 'axes_info'
  window_ids?: number[]
  axes?: AxisRow[]
}

export interface ActionActiveMessage extends MsgBase {
  type: 'action_active'
  window_id: number
  name: string
  active: boolean
}

export interface SubItemMessage extends MsgBase {
  type: 'sub_item'
  window_id: number
  action: string
  name: string
  color?: string
  vtype?: string
  calculation?: string
  active: boolean
}

export interface HistogramMessage extends MsgBase {
  type: 'histogram'
  window_id: number
  counts?: number[]
  edges?: number[]
  vmin: number
  vmax: number
  threshold?: number | null
}

export interface NavShapePromptMessage extends MsgBase, NavShapePrompt {
  type: 'nav_shape_prompt'
}

export interface LoadingMessage extends MsgBase {
  type: 'loading'
  busy?: unknown
  text?: unknown
}

export interface SignalTypeInfoMessage extends MsgBase {
  type: 'signal_type_info'
  window_ids?: number[]
  current?: unknown
  options?: string[]
}

export interface SelectorInfoMessage extends MsgBase {
  type: 'selector_info'
  window_id: number
  /** Stable per-selector key — one navigator window can carry several
   *  selectors, each its own dock row. Absent in older emits (fall back to
   *  window_id). */
  selector_id?: number
  /** The selector's widget colour (the dock row's dot). */
  color?: string | null
  mode?: 'crosshair' | 'integrate'
  title?: string
}

export interface SignalTreeMessage extends MsgBase {
  type: 'signal_tree'
  window_id: number
  tree?: TreeNode
  active_signal_id?: number
}

export interface NavigatorOptionsMessage extends MsgBase {
  type: 'navigator_options'
  window_id: number
  /** Named navigators of the window's tree — the chip strip (shown when ≥2). */
  names?: string[]
  /** The navigator currently displayed on the live figure. */
  current?: string | null
}

export interface PlaybackStateMessage extends MsgBase {
  type: 'playback_state'
  /** True while the movie clock is running. */
  playing: boolean
  /** Current speed multiplier (1/2/4/8) — drives the Fast Forward "×N" badge. */
  speed?: number
  loop?: boolean
}

/** One entry of a `console_result`/`console_vars` value description. */
export interface ConsoleVarKind {
  name: string
  kind?: string
  shape?: number[] | null
  dtype?: string | null
  lazy?: boolean
}

export interface ConsoleResultMessage extends MsgBase {
  type: 'console_result'
  exec_id: number
  ok: boolean
  value_repr?: string | null
  stdout?: string | null
  error?: string | null
  traceback?: string | null
  duration_ms?: number
  result?: ConsoleVarKind | null
}

/** One row of the console's live variable table (the result-chip strip). */
export interface ConsoleVarEntry extends ConsoleVarKind {
  source: 'signal' | 'assign' | 'out'
  /** For source:"signal" entries — the MDI window(s) currently showing it, so
   *  a dropped `SIGNAL_REF_DRAG_MIME` windowId can resolve to this var name. */
  window_ids?: number[] | null
}

export interface ConsoleVarsMessage extends MsgBase {
  type: 'console_vars'
  vars: ConsoleVarEntry[]
}

export interface ConsoleCompletionsMessage extends MsgBase {
  type: 'console_completions'
  complete_id: number
  matches: string[]
}

/**
 * Live-preview reply for a `console_preview` request (the eye-toggled
 * thumbnail/sparkline/scalar). `preview_id` echoes the requester's id so the
 * renderer can drop stale replies (newest-wins). `kind` selects the render:
 *   • image      — `data_b64` is raw uint8 GRAYSCALE, row-major, length w*h
 *   • sparkline  — `points` (≤512; `null` = a gap in the line)
 *   • scalar     — `text` (a short repr)
 *   • unavailable— `reason` (may be "" → render quietly / keep prior content)
 */
export interface ConsolePreviewResultMessage extends MsgBase {
  type: 'console_preview_result'
  preview_id: number
  kind: 'image' | 'sparkline' | 'scalar' | 'unavailable'
  w?: number
  h?: number
  data_b64?: string
  points?: (number | null)[]
  text?: string
  shape?: number[] | null
  dtype?: string | null
  reason?: string
  elapsed_ms?: number
}

/** A workflow-node bind completed; the backend assigned it `name`. */
export interface ConsoleNodeBoundMessage extends MsgBase {
  type: 'console_node_bound'
  name: string
}

export interface LogMessage extends MsgBase {
  type: 'log'
  level: unknown
  name: unknown
  msg: unknown
  time: unknown
}

export interface LogBackfillMessage extends MsgBase {
  type: 'log_backfill'
  entries?: LogEntry[]
}

export interface LogLevelMessage extends MsgBase {
  type: 'log_level'
  level: unknown
}

// ── Report ─────────────────────────────────────────────────────────────────

/** One cell of the report document (markdown text or an embedded figure). */
export interface ReportCell {
  id: string
  cell_type: 'markdown' | 'figure'
  /** markdown cells: the source text. */
  source?: string
  /** figure cells: the caption (alt text). */
  caption?: string
  /** figure cells: a template placeholder (dashed drop-zone, no figure yet). */
  placeholder?: boolean
  /** figure cells: the live figure id (null when its data is offline). */
  fig_id?: string | null
  /** figure cells: the SignalRef couldn't be rebound → show the baked PNG. */
  data_offline?: boolean
  /** figure cells: a data-URL PNG fallback (present only for offline cells). */
  png?: string
}

/** The authoritative report document (mirrored by the renderer for editing). */
export interface ReportDocState {
  open: boolean
  path: string | null
  title: string
  template: boolean
  dirty: boolean
  cells: ReportCell[]
}

/** Full report document — the backend re-broadcasts this on every change. */
export interface ReportStateMessage extends MsgBase {
  type: 'report_state'
  report: ReportDocState
}

/** The backend needs a fresh PNG snapshot per listed cell before it can save.
 *  The renderer harvests each via the figure iframe export protocol and replies
 *  with `report_snapshots {token, images:{cell_id: dataUrl}}`. */
export interface ReportNeedSnapshotsMessage extends MsgBase {
  type: 'report_need_snapshots'
  token: string
  cells: Array<{ cell_id: string; fig_id: string }>
}

/** The report was written to disk at `path`. */
export interface ReportSavedMessage extends MsgBase {
  type: 'report_saved'
  path: string
}

/**
 * Wizard-scoped events re-broadcast verbatim as DOM CustomEvents (the caret
 * components subscribe directly). The payload beyond `type` is consumer-defined,
 * so it stays untyped here (the `MsgBase` index signature covers field access).
 */
export interface WizardEventMessage extends MsgBase {
  type:
    | 'vom_fit'
    | 'vom_library_ready'
    | 'om_library_ready'
    | 'fv_auto_params'
    | 'fv_models'
    | 'cod_results'
    | 'cod_cif_ready'
    | 'gpu_status_result'
}

/** The discriminated union the renderer dispatches over. */
export type PlotAppMessage =
  | ReadyMessage
  | StatusMessage
  | ErrorMessage
  | BackendExitedMessage
  | FigureMessage
  | ToolbarConfigMessage
  | WindowVisibilityMessage
  | WindowClosedMessage
  | WindowTitleMessage
  | StateUpdateMessage
  | StateUpdateBinaryMessage
  | CompositionMessage
  | MetadataMessage
  | AxesInfoMessage
  | ActionActiveMessage
  | SubItemMessage
  | HistogramMessage
  | NavShapePromptMessage
  | LoadingMessage
  | SignalTypeInfoMessage
  | SelectorInfoMessage
  | SignalTreeMessage
  | NavigatorOptionsMessage
  | PlaybackStateMessage
  | ConsoleResultMessage
  | ConsoleVarsMessage
  | ConsoleCompletionsMessage
  | ConsolePreviewResultMessage
  | ConsoleNodeBoundMessage
  | LogMessage
  | LogBackfillMessage
  | LogLevelMessage
  | ReportStateMessage
  | ReportNeedSnapshotsMessage
  | ReportSavedMessage
  | WizardEventMessage

/**
 * Narrow a raw incoming message (`Record<string, unknown>` from the IPC bridge)
 * into the `PlotAppMessage` union. Runtime is a single `typeof type === string`
 * check — the discriminated-union machinery does the rest at the call site via
 * the `switch (msg.type)`. Messages with an unrecognised `type` still flow
 * through (handled by the dispatcher's default/no-op); this only asserts the
 * discriminant exists.
 */
export function asPlotAppMessage(msg: Record<string, unknown>): PlotAppMessage {
  return msg as PlotAppMessage
}
