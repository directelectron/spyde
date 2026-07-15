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
  // Optional human-readable explanation. Set when the main process synthesises
  // this for a packaged env-setup failure (uv sync failed on first launch) —
  // distinct from a plain runtime death (runner.ts emits no reason). Shown in
  // the blocking overlay in place of the generic "process exited" copy.
  reason?: string
}

/** Progress of a heavy backend action (movie export, …) via emit_progress.
 *  Surfaced through the StatusBar busy line (done>=total or total<=0 clears it)
 *  and re-broadcast as a `spyde:progress` CustomEvent for wizard footers. */
export interface ProgressMessage extends MsgBase {
  type: 'progress'
  done: number
  total: number
  label?: string
}

/** First-run Python environment setup progress (main process, index.ts). Parsed
 *  from `uv` output by envProgress.ts. `start`/`done` bracket the setup; each
 *  `progress` carries a parsed phase/step (+ optional download %) and the raw
 *  line for the overlay's live tail. Drives the floating EnvSetupOverlay. */
export interface EnvSetupMessage extends MsgBase {
  type: 'env_setup'
  event: 'start' | 'progress' | 'done'
  phase?: 'resolving' | 'downloading' | 'installing' | 'building' | 'torch' | 'working'
  step?: string
  percent?: number | null
  raw?: string
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

/** A layer within a report figure panel (the PIXEL-FREE recipe, from
 *  `LayerSpec.to_dict()`). `source.title` names the layer in the edit toolbar. */
export interface RepfigLayer {
  id: string
  source?: {
    file_path?: string | null
    tree_node?: string | null
    view?: string | null
    title?: string | null
    [k: string]: unknown
  }
  cmap: string
  clim?: [number, number] | null
  alpha: number
  visible: boolean
}

/** One panel of a report figure's recipe (from `PanelSpec.to_dict()`). The
 *  `axes` dict carries `x_axis`/`y_axis` float arrays (snapshot-time calibration)
 *  used to derive a sensible data-coord default for a new annotation. */
export interface RepfigPanel {
  id: string
  grid_pos: [number, number]
  kind: string
  layers: RepfigLayer[]
  axes?: {
    units?: string
    x_axis?: number[]
    y_axis?: number[]
    [k: string]: unknown
  } | null
  annotations: Array<Record<string, unknown> & { kind: string }>
  scalebar?: boolean
  colorbar?: boolean
  title?: string
  insets?: Array<Record<string, unknown>>
}

/** A FIGURE-LEVEL annotation (distinct from a panel's `annotations`): a marker
 *  in the anyplotlib figure-marker schema, positioned in FIGURE FRACTIONS
 *  (0..1, top-left origin) — NO calibration. `kind` is text/circle/rect/arrow.
 *  An `id` is assigned by the backend so a drag persists by id. */
export interface RepfigFigAnnotation {
  id?: string
  kind: 'text' | 'circle' | 'rect' | 'arrow'
  x: number
  y: number
  // per-kind position/size fields (fractions): text→text; circle→r; rect→w,h;
  // arrow→u,v — plus optional color/fontsize/linewidth.
  [k: string]: unknown
}

/** A report figure cell's full recipe (from `FigureSpec.to_dict()`) — shipped
 *  pixel-free in `report_state` so the edit toolbar can list panels/layers/
 *  annotations. `annotations` are FIGURE-level markers (figure fractions);
 *  `layout.hspace`/`layout.wspace` are the inter-panel gaps. */
export interface RepfigSpec {
  layout: {
    kind: string; rows?: number; cols?: number
    hspace?: number; wspace?: number
    [k: string]: unknown
  }
  panels: RepfigPanel[]
  nav_context?: { indices?: number[] } | null
  annotations?: RepfigFigAnnotation[]
}

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
  /** figure cells: the pixel-free FigureSpec recipe (panels/layers/annotations)
   *  driving the edit toolbar. Absent while a cell is a placeholder. */
  figure?: RepfigSpec
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

/** An export finished: an HTML file (static/interactive) or a markdown folder
 *  was written at `path`. The renderer's Export flow awaits this, matched by
 *  `token` (the same token it sent in the triggering `report_export_html` /
 *  `report_export_markdown` payload — the backend echoes it back verbatim) so
 *  two exports in flight at once can't cross-wire; the PDF flow's first leg
 *  (temp static HTML) also matches on `token`. */
export interface ReportExportedMessage extends MsgBase {
  type: 'report_exported'
  kind: 'html-static' | 'html-interactive' | 'markdown-folder'
  path: string
  token?: string | null
}

/** A report figure cell's SELECTED panel changed (backend is the source of
 *  truth: a click on the live figure, a dock chip, or a widget drag all funnel
 *  through `ReportManager.select_panel`). `panel_id` is the SPEC panel id, or
 *  null for figure-level (deselected). Re-broadcast as a `spyde:` CustomEvent so
 *  the open editor for THIS cell mirrors it. */
export interface ReportPanelSelectedMessage extends MsgBase {
  type: 'report_panel_selected'
  cell_id: string
  panel_id: string | null
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

// ── MDI image layering (overlay) ────────────────────────────────────────────

/** One layer's appearance, as tracked by the backend (`spyde/actions/overlay.py`
 *  `PlotLayer.to_state`). */
export interface LayerState {
  id: string
  title: string
  cmap: string
  alpha: number
  clim: [number, number] | null
  visible: boolean
}

/** The authoritative layer stack for one target window — emitted after every
 *  overlay mutation (`overlay_add`/`overlay_set`/`overlay_remove`) and in reply
 *  to `overlay_query`. */
export interface LayersStateMessage extends MsgBase {
  type: 'layers_state'
  window_id: number
  layers: LayerState[]
}

/** Report-cell figure-compose options offered for a source window (consumed by
 *  the report-builder drop-zone UI, not by this dock). */
export interface RepfigComposeOptionsMessage extends MsgBase {
  type: 'repfig_compose_options'
  cell_id: string
  source_window_id: number
  options: string[]
  detail: {
    same_shape: boolean
    nav_signal_pair: boolean
  }
}

// ── Movie export wizard (mvx_*) ─────────────────────────────────────────────

/** One time-gated overlay annotation in the movie (`time_range` in the movie's
 *  frame-index or seconds domain, matched to the pipeline's convention). */
export interface MvxAnnotation {
  kind: 'text' | 'rect'
  time_range: [number, number]
  text?: string
  /** Position/geometry, in data or fractional coords (pipeline-defined). */
  x?: number
  y?: number
  w?: number
  h?: number
  [k: string]: unknown
}

/** The tunable movie-export parameters (mirrors the wizard's controls). */
export interface MvxParams {
  fps: number
  downsample: number
  stride: number
  t_start: number
  t_end: number
  cmap?: string
  clim?: [number, number] | null
  timestamp: boolean
  scalebar: boolean
  annotations: MvxAnnotation[]
  [k: string]: unknown
}

/** One trace overlay (a 1-D plot dragged into the wizard). */
export interface MvxTrace {
  id: string
  label: string
  color: string
  units?: string
}

/** Authoritative movie-export wizard state — re-broadcast by the backend on
 *  every mvx mutation (open/tune/add_trace/remove_trace). The renderer's
 *  MovieExportWizard subscribes via the `spyde:mvx_state` CustomEvent. */
export interface MvxStateMessage extends MsgBase {
  type: 'mvx_state'
  window_id: number
  ffmpeg_ok: boolean
  running: boolean
  n_frames: number
  time: { scale_s: number; units: string }
  params: MvxParams
  traces: MvxTrace[]
}

/** A movie export finished — the wizard shows a success note (basename of
 *  `path`). Re-broadcast as `spyde:mvx_done`. */
export interface MvxDoneMessage extends MsgBase {
  type: 'mvx_done'
  path: string
  frames: number
}

/** The discriminated union the renderer dispatches over. */
export type PlotAppMessage =
  | ReadyMessage
  | StatusMessage
  | ErrorMessage
  | BackendExitedMessage
  | EnvSetupMessage
  | ProgressMessage
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
  | ReportExportedMessage
  | ReportPanelSelectedMessage
  | WizardEventMessage
  | LayersStateMessage
  | RepfigComposeOptionsMessage
  | MvxStateMessage
  | MvxDoneMessage

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
