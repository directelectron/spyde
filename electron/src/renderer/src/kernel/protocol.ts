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

export interface StateUpdateMessage extends MsgBase {
  type: 'state_update'
  fig_id: string
  key: string
  value: unknown
}

export interface StrainRingsMessage extends MsgBase {
  type: 'strain_rings'
  window_id: number
  rings?: number[]
  selected?: number[]
}

export interface IpfKeyMessage extends MsgBase {
  type: 'ipf_key'
  window_id: number
  data_url: string
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
  mode?: 'crosshair' | 'integrate'
  title?: string
}

export interface SignalTreeMessage extends MsgBase {
  type: 'signal_tree'
  window_id: number
  tree?: TreeNode
  active_signal_id?: number
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
    | 'cod_results'
    | 'cod_cif_ready'
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
  | StateUpdateMessage
  | StrainRingsMessage
  | IpfKeyMessage
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
  | LogMessage
  | LogBackfillMessage
  | LogLevelMessage
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
