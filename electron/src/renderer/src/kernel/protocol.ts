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
  MetadataInfo,
  ChunkInfo,
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

/** Per-window compute lifecycle — start/stop bracket a long backend compute
 *  painting into this window (the progressive navigator fill, a streamed
 *  virtual image, …). Drives the floating translucent "Calculating…" overlay
 *  centered on the plot's figure area (WindowContent.tsx). The backend
 *  guarantees a matching `computing:false` even on a cancelled/failed compute
 *  (see `spyde.actions.lifecycle.window_computing`) — the renderer additionally
 *  clears on window close / tree-node switch as a staleness backstop. */
export interface WindowComputingMessage extends MsgBase {
  type: 'window_computing'
  window_id: number
  computing: boolean
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
  // {group: {prop: raw}} — the config-declared cells with a writable hyperspy
  // metadata `key` (as opposed to a derived attr/function entry like Dtype/
  // Dim., or the synthetic Dataset section). Cell PRESENCE gates which cells
  // the panel lets you click into (mirroring the axes table's per-field
  // editability); the VALUE is the current RAW unit-free value ("" when unset)
  // the inline editor pre-fills with — the `metadata` strings have units baked
  // in ("12000.5 x") and would fail the backend's numeric parse if sent back.
  editable?: Record<string, Record<string, string>>
  // {group: {prop: {description, key, units, derived}}} — the STATIC config
  // text behind each field, for the dock's detail popover. The panel shows a
  // curated summary, so this is what explains a field the summary abbreviates
  // (or hides entirely, like Mode / Cam.).
  info?: MetadataInfo
  // Dask block layout of the displayed node, or null/absent when it is eager.
  // Numbers, not dask's own HTML: the dock draws it in the app's palette and
  // flags a nav/signal split dask has no way to know about.
  chunking?: ChunkInfo | null
}

export interface AxesInfoMessage extends MsgBase {
  type: 'axes_info'
  window_ids?: number[]
  axes?: AxisRow[]
}

/**
 * The Axes table's "+" origin-pick state for one window. The BACKEND owns the
 * draggable crosshair (`plot._offset_cross`), so it also owns this flag: the
 * dock renders the button from the last `offset_pick` it heard for the ACTIVE
 * window rather than from a local boolean, keeping "button ON ⟺ crosshair
 * alive" true across window switches, node switches and refused toggles.
 */
export interface OffsetPickMessage extends MsgBase {
  type: 'offset_pick'
  window_id: number
  on: boolean
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
  // Full data extent, and whether the bins cover less than it (the tail was
  // clipped into the end bins — see Plot._hist_range).
  data_min?: number
  data_max?: number
  clipped?: boolean
}

export interface NavShapePromptMessage extends MsgBase, NavShapePrompt {
  type: 'nav_shape_prompt'
}

/** Reveal a local directory in the OS file manager (Examples → Show Example
 *  Data Directory). The backend owns the location, the main process opens it. */
export interface OpenPathMessage extends MsgBase {
  type: 'open_path'
  path: string
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

/** A selector's driven signal window closed — the backend deregistered it
 *  (`MultiplotManager.remove_navigation_selector`) and the dock should drop
 *  its row directly (WINDOW_CLOSED alone can't: selectors are keyed by the
 *  NAVIGATOR's window_id, not the closed signal window's id). */
export interface SelectorRemovedMessage extends MsgBase {
  type: 'selector_removed'
  selector_id: number
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
  /** Subsystem tag from log_stream._area_for (e.g. 'navigator', 'dask'). */
  area?: unknown
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
  /** Clear→colour intensity-ramp hex (overlay tint); absent/null = colormap
   *  display (`cmap` stays stored as the revert value while tinted). */
  tint?: string | null
  /** Line-panel curve styling (kind === "line" only): CSS stroke colour,
   *  stroke width in px, and the legend label. Absent on image layers. */
  color?: string | null
  linewidth?: number | null
  label?: string | null
}

/** One panel of a report figure's recipe (from `PanelSpec.to_dict()`). The
 *  `axes` dict carries `x_axis`/`y_axis` float arrays (snapshot-time calibration)
 *  used to derive a sensible data-coord default for a new annotation. */
export interface RepfigPanel {
  id: string
  grid_pos: [number, number]
  /** "image" | "line" | "scene3d" — a scene3d panel is a 3-D scatter scene
   *  (IPF sphere); its layers paint nothing (the point cloud lives backend-side)
   *  and the slim bar hides annotation/layer controls for it. */
  kind: string
  layers: RepfigLayer[]
  /** Per-panel text sizes ({title,x_label,y_label,ticks,legend,colorbar} → px),
   *  emitted only when set — the double-click size popover edits these. */
  text_sizes?: Record<string, number> | null
  /** scene3d recompute params ({kind:'ipf3d', direction, point_size, bounds,
   *  camera?}) — small params only, never pixels. Absent on image/line panels. */
  scene?: Record<string, unknown> | null
  /** EPHEMERAL (stamped at emit time only): navigation dimensionality of the
   *  panel's layer-0 source signal — gates the fresh-slice callout buttons.
   *  Never part of the persisted spec/YAML. */
  nav_dims?: number
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
  /** Drop-time vectors embed choice for a vectors-carrying source: 'viewer' (the
   *  default when the tree has vectors — the sidebar cell hosts the live 2-panel
   *  explorer) | 'image' (a static snapshot). Absent for a non-vectors cell. */
  vectors_mode?: string
}

/** One cell of the report document (markdown text, an embedded figure, a
 *  dropped/pasted/browsed photo, or a SPLIT block — text BESIDE a figure/photo). */
export interface ReportCell {
  id: string
  cell_type: 'markdown' | 'figure' | 'image' | 'split' | 'movie'
  /** markdown + split cells: the (text side's) source markdown. */
  source?: string
  /** figure + image + split cells: the caption (alt text). */
  caption?: string
  /** image (photo) cells: the raw image inlined as a data URL (rendered inline
   *  as a resizable <img>). */
  image?: string
  /** image cells: the asset extension (png/jpg/gif/webp) — informs the MIME. */
  image_ext?: string
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
  /** figure cells in EDIT MODE only: live edit-widget id → its spec annotation
   *  (panel_id + index), so a widget click resolves to the annotation whose
   *  style popover should open. Ephemeral — never persisted. */
  ann_widgets?: Record<string, { panel_id: string; index: number }>
  /** Present mode (Phase 6): this cell STARTS a new slide (cells accumulate
   *  onto the current slide until the next break). Absent → false. */
  slide_break?: boolean
  /** Present mode (Phase 6): an optional "go live" excursion — Present mode
   *  turns it into a "Launch live ▶" button. e.g. { tutorial: 'strain',
   *  guide: 'strain' }. Absent → no button on the slide. */
  live_action?: { tutorial?: string; guide?: string } | null
  /** Present mode (presentation polish): the per-SLIDE kind, carried on the
   *  slide's FIRST cell. '' (content, default) renders normally; 'title' makes
   *  the whole slide a big-centered TITLE / SECTION slide. */
  slide_kind?: string
  /** Present mode: the per-SLIDE background/heading preset, on the slide's first
   *  cell. '' / 'default' the standard dark stage; 'plain' flat darker; 'accent'
   *  a subtle accent-tinted gradient. */
  slide_style?: string
  /** Present mode (presenter view): the slide's SPEAKER NOTES — free multi-line
   *  markdown carried on the slide's FIRST cell. Shown only in the presenter
   *  view; NEVER to the audience. Absent → '' (no notes). */
  notes?: string
  /** SPLIT cells (Wave A): which side the TEXT sits on — 'text-left' (text left,
   *  figure right — the default) | 'text-right' (mirror). The figure side rides
   *  in `figure` (a FigureSpec) or `image` (a photo data URL), like a
   *  figure/image cell; `source` carries the text side. */
  split_layout?: string
  /** SPLIT cells: the figure side is EMPTY (no figure/photo dropped yet) — the
   *  renderer draws a drop zone. Absent/false → the figure side is filled. */
  split_empty?: boolean
  /** MOVIE cells: the pixel-free MovieSpec recipe (source ref + render/edit
   *  state) driving the card summary + the full-screen editor. */
  movie?: MovieSpec
  /** MOVIE cells: true once a source in-situ signal is assigned (placeholder=false).
   *  A placeholder movie shows the "pick a signal" drop-zone card. */
  has_source?: boolean
  /** MOVIE cells: a data-URL PNG of a representative frame (baked on export /
   *  loaded from the zip) shown as the card poster. Absent → no poster yet. */
  poster?: string
}

/** A 1-D-signal-as-text overlay: a live value painted as text (e.g. "T = 812.3 °C").
 *  The value is resampled from the referenced 1-D signal onto the movie time base
 *  at render time; `fmt` is a Python-style format string over {label,value,units}. */
export interface MovieTextOverlay {
  source?: Record<string, unknown>   // a SignalRef dict (opaque to the renderer)
  label?: string
  units?: string
  fmt?: string
  xy?: [number, number]              // source-pixel position
  size?: number
  color?: string
  time_range?: [number, number]      // seconds; absent → always shown
}

/** A time-gated overlay drawn on the movie (text / rect / circle / arrow). ROIs are
 *  persistent rect/circle annotations. Positions/sizes are in SOURCE pixels. */
export interface MovieAnnotation {
  kind: 'text' | 'rect' | 'circle' | 'arrow'
  time_range?: [number, number]      // seconds; absent → always shown
  text?: string
  xy?: [number, number]
  xy2?: [number, number]             // arrow head
  wh?: [number, number]              // rect
  radius?: number                    // circle
  color?: string
  width?: number
  size?: number                      // text px
}

/** A freeze hold: linger on frame `t` for `hold_s` seconds (repeats the frame).
 *  Superseded by MovieSpeedSegment (a 0× segment) but kept for back-compat. */
export interface MovieFreeze { t: number; hold_s: number }

/** A variable-speed segment: source time inside `time_range` (seconds) plays at
 *  `speed`× (0 = hold/freeze, <1 slow-mo, >1 fast-forward). The export resamples
 *  source→output time through these, emitting more frames where slow. */
export interface MovieSpeedSegment { time_range: [number, number]; speed: number }

/** The pixel-free MovieSpec recipe for a movie cell (mirrors the backend). */
export interface MovieSpec {
  source?: Record<string, unknown> | null   // a SignalRef dict (opaque)
  params?: MovieParams
  annotations?: MovieAnnotation[]
  text_overlays?: MovieTextOverlay[]
  freezes?: MovieFreeze[]
  speed_segments?: MovieSpeedSegment[]
  overlay_image?: Record<string, unknown> | null
  crop?: [number, number, number, number] | null   // [x0,y0,x1,y1] source px
  out_size?: [number, number] | null                // [w,h] output px
}

/** The navigator's current frame during scrub / playback (spyde:movie_frame). */
export interface MovieFrameMessage extends MsgBase {
  type: 'movie_frame'
  cell_id: string
  t: number
  playing?: boolean
}

/** The base render params for a movie (matches the backend params dict). */
export interface MovieParams {
  fps?: number
  downsample?: number
  stride?: number
  cmap?: string
  clim?: [number, number] | null
  timestamp?: boolean
  scalebar?: boolean
  axes?: boolean                        // draw calibrated axis ticks (default true)
  t_start?: number
  t_end?: number
}

/** The authoritative full-screen Movie editor state (spyde:movie_state). */
export interface MovieStateMessage extends MsgBase {
  type: 'movie_state'
  cell_id: string
  open: boolean
  has_source: boolean
  ffmpeg_ok: boolean
  running: boolean
  n_frames: number
  time: { scale_s: number; units: string }
  sig: { scale_x: number; units: string }
  source_title: string
  params: MovieParams
  annotations: MovieAnnotation[]
  text_overlays: MovieTextOverlay[]
  freezes: MovieFreeze[]
  speed_segments: MovieSpeedSegment[]
  crop: [number, number, number, number] | null
  out_size: [number, number] | null
  frame_size: [number, number]        // source frame [w,h] — the editor coord space
  /** Authoritative exported {frames, w, h} (freeze-expanded, even-crop, out_size-
   *  aware) — the readout uses this so it can't drift from what export writes. */
  output_info?: { frames: number; w: number; h: number }
  /** The tree's LIVE 2-D signal figure the editor re-parents into its preview area
   *  (a figId is 1:1 with an iframe, so mounting it supersedes the MDI iframe while
   *  the editor is open). null when no source is resolved. */
  signal_fig_id: string | null
  signal_window_id: number | null     // the MDI window holding the signal figure
  nav_fig_id: string | null           // the 1-D navigator figure (shown beside, opt)
  current_index: number               // the navigator's current time index
}

/** Export finished (spyde:movie_done). */
export interface MovieDoneMessage extends MsgBase {
  type: 'movie_done'
  cell_id: string
  path: string
  frames: number
}

/** "Open this movie cell in the full-screen editor" (spyde:movie_edit_open) —
 *  dispatched by the sidebar Movie card / add-with-open. */
export interface MovieEditOpenMessage extends MsgBase {
  type: 'movie_edit_open'
  cell_id: string
}

/** The authoritative report document (mirrored by the renderer for editing). */
export interface ReportDocState {
  open: boolean
  path: string | null
  title: string
  template: boolean
  /** Wave A — the document TYPE: 'report' (a scrolling article — the default and
   *  every legacy document) | 'presentation' (a slide deck) | 'movie' (reserved).
   *  Absent on an older backend/file → 'report'. */
  type?: string
  dirty: boolean
  /** Label of the action the next undo would reverse (e.g. "Delete figure
   *  cell"), or null/absent when the stack is empty. Drives the Undo button's
   *  label and enabled state. */
  undo?: string | null
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

/** A figure drop whose source tree carries diffraction vectors: the backend
 *  deferred the cell and asks how HTML exports should embed it. The sidebar
 *  prompts and re-sends `report_add_figure` with the original payload plus
 *  `vectors_mode: 'viewer' | 'image'`. Re-broadcast as a `spyde:` CustomEvent. */
export interface ReportVectorsChoiceMessage extends MsgBase {
  type: 'report_vectors_choice'
  source_window_id: number
  index?: number | null
  at_cell?: string | null
  caption?: string
  count?: number
}

/** An export finished: an HTML file (static/interactive) or a markdown folder
 *  was written at `path`. The renderer's Export flow awaits this, matched by
 *  `token` (the same token it sent in the triggering `report_export_html` /
 *  `report_export_markdown` payload — the backend echoes it back verbatim) so
 *  two exports in flight at once can't cross-wire; the PDF flow's first leg
 *  (temp static HTML) also matches on `token`. */
export interface ReportExportedMessage extends MsgBase {
  type: 'report_exported'
  kind: 'html-static' | 'html-interactive' | 'html-slides' | 'markdown-folder'
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

/** One in-flight Examples-menu download (pooch fetch on the backend). Emitted
 *  throttled (~5 Hz) while bytes flow; `total` 0 = size unknown. Consumed by
 *  DownloadToasts (bottom-right card with a progress bar + Cancel, which sends
 *  the `download_cancel` action with this `token`). */
export interface DownloadProgressMessage extends MsgBase {
  type: 'download_progress'
  token: string
  label: string
  done: number
  total: number
}

/** Terminal state for a download toast — remove the card. `cancelled` downloads
 *  also get a status-line note from the backend. */
export interface DownloadDoneMessage extends MsgBase {
  type: 'download_done'
  token: string
  ok: boolean
  cancelled: boolean
}

/** One live cluster sample (~every 2 s while the cluster is up) for the
 *  StatusBar Dask monitor HUD. `gpu` present only on NVIDIA machines. */
export interface DaskStatsMessage extends MsgBase {
  type: 'dask_stats'
  workers: {
    name: string
    cpu: number          // percent (per worker process)
    mem: number          // bytes
    mem_limit: number    // bytes (0 = unknown)
    executing: number    // tasks running now
    ready: number        // tasks queued on the worker
  }[]
  tasks: { executing: number; queued: number }
  gpu?: { util: number; vram_used: number; vram_total: number }  // %, MB, MB
  host_cpu?: number      // whole-machine CPU percent
  host_mem?: number      // whole-machine RAM percent (paging = the freeze killer)
  /** Effective compute limits (the popover settings rows; `compute_configure`
   *  applies changes by restarting the cluster). */
  config?: {
    mem_fraction: number       // cluster RAM budget, fraction of the machine
    compute_fraction: number   // CPU budget, fraction of logical cores
    gpu_workers: string        // GPU-feeding workers: "1".."8" | "one" | "all" | "off"
  }
}

/** Measured read-throughput readout for the StatusBar MB/s HUD. The backend
 *  records the COLD nav-frame / chunk reads it already performs (bytes/elapsed,
 *  EMA-smoothed; cache-hits excluded) and emits this throttled (~1 Hz, only on
 *  change). Re-broadcast as a `spyde:io_throughput` CustomEvent (same idiom as
 *  `dask_stats`). `color` mirrors the HUD legend: green > 1 GB/s, yellow
 *  0.1-1 GB/s, red < 100 MB/s. See `spyde.backend.io_throughput`. */
export interface IoThroughputMessage extends MsgBase {
  type: 'io_throughput'
  window_id: number | null
  mbps: number
  color: 'green' | 'yellow' | 'red'
}

// ── Particle table dock ─────────────────────────────────────────────────────

/** One column of the particle table, as the backend describes it. Mirrors the
 *  renderer-side `DataColumn` (components/DataTable.tsx) minus the function
 *  fields, which cannot cross IPC. */
export interface ParticlesTableColumn {
  /** Property read from each row — e.g. a name from
   *  `spyde.signals.particles.COLUMNS`. */
  key: string
  label: string
  /** Fixed pixel width; omitted → the column flexes. */
  width?: number
  align?: 'left' | 'center' | 'right'
  /** Right-aligned + tabular figures. */
  numeric?: boolean
  /** Default true. */
  sortable?: boolean
  /** 'swatch' prefixes a colour chip keyed on the cell value (track id). */
  kind?: 'text' | 'swatch'
  /** Decimal places for a numeric cell. */
  precision?: number
  /** Appended to the value ("nm", "nm²"). */
  units?: string
}

/** One row of the Events tab — exactly `ParticleEvent.to_dict()` from
 *  `spyde/particles/track.py`. */
export interface ParticleEventRecord {
  frame: number
  /** 'birth' | 'death' | 'merge' | 'split' (track.py EVENT_KINDS). */
  kind: string
  tracks: number[]
  particles: number[]
}

/**
 * The particle/track table for one result window, in reply to a
 * `particles_query {window_id}` action (and pushed again when a segmentation
 * batch finalises). Re-broadcast as a `spyde:particles_table` CustomEvent and
 * consumed by BottomDock — the renderer holds no reducer state for it.
 *
 * `columns` is backend-supplied so the dock stays data-agnostic: sending a
 * different column set is the whole mechanism for showing tracks instead of
 * particles. `events` is fixed-shape and drives the Events tab.
 */
export interface ParticlesTableMessage extends MsgBase {
  type: 'particles_table'
  /** The window this table belongs to; `null` = unscoped (always shown). */
  window_id: number | null
  /** Dock header text. Defaults to "Particles" when absent. */
  title?: string
  columns: ParticlesTableColumn[]
  /** One object per row, keyed by `columns[].key`. An optional numeric `id`
   *  field is used as the stable selection key when present. */
  rows: Record<string, unknown>[]
  events?: ParticleEventRecord[]
  /** Calibrated unit of the length columns (`SpyDEParticles.units`). */
  units?: string
  /** True while a batch is still streaming rows in — shows a "streaming…" pill. */
  partial?: boolean
}

// ── Segment Particles caret (spyde/actions/particles_action.py, plan B7) ─────

/** One scribble class as the backend reports it — `ScribbleClass.to_dict()`
 *  plus the labelled-pixel count `SegmentWizard.class_report()` adds.
 *
 *  `pixels` is NOT decoration: under-training a class is the failure mode
 *  plan §B3 calls out, and this count is how the user notices. A class with
 *  zero pixels is present in the list rather than absent from it. */
export interface SegClassInfo {
  id: number
  name: string
  /** CSS hex — the renderer's units; the backend never converts it. */
  colour: string
  /** Counts toward the foreground probability map (several classes may). */
  particle: boolean
  /** Marks the SEAM between two touching particles (the ilastik third class).
   *  Painting it lets the backend split by connected components and skip the
   *  distance transform and watershed entirely. Mutually exclusive with
   *  `particle` — the backend raises if a class claims both. */
  boundary: boolean
  pixels: number
}

/** The caret's authoritative state, emitted by `_emit_state` after open,
 *  method switch, paint and train. `params` are the EFFECTIVE (coerced)
 *  values, not what the caret last sent. */
export interface SegStateMessage extends MsgBase {
  type: 'seg_state'
  window_id: number | null
  /** 'classical' | 'scribble' | 'prompt'. */
  method: string
  /** Navigator frame the preview is showing. */
  frame: number
  n_frames: number
  frame_shape: [number, number]
  classes: SegClassInfo[]
  /** Frame indices carrying at least one scribble. */
  labelled_frames: number[]
  trained: boolean
  params: Record<string, unknown>
}

/** One frame's preview result. `min_size` is the EFFECTIVE value that ran and
 *  `min_size_floored` says the backend raised what the caret asked for — the
 *  caret must show the effective number (plan §0.9: at min_size=0 the split
 *  returns background speckle as particles). `count` is already post-filter. */
export interface SegPreviewMessage extends MsgBase {
  type: 'seg_preview'
  window_id: number | null
  frame: number
  count: number
  /** Per-instance areas in calibrated units², capped at 2000 entries. */
  areas: number[]
  median_area: number
  /** Signal-axis unit ("nm"); areas are in unit². */
  units: string
  method: string
  min_size: number
  min_size_floored: boolean
  elapsed_ms: number
  /**
   * `[y0, x0, h, w]` when the frame was too big to preview whole, else null.
   *
   * A 4096² frame costs 8.4 s to segment, so the backend previews a centred
   * megapixel crop at FULL resolution instead (cropping rather than
   * downsampling, so the preview runs the identical algorithm the real run
   * will). The caret MUST surface this: otherwise "12 particles on this frame"
   * is a lie about a frame where only the middle sixteenth was looked at.
   */
  preview_box: [number, number, number, number] | null
}

/** Scribble classifier fit report (`ScribbleClassifier.fit`). */
export interface SegTrainedMessage extends MsgBase {
  type: 'seg_trained'
  window_id: number | null
  report: {
    device?: string
    n_pixels?: number
    n_channels?: number
    n_classes?: number
    labelled_frames?: number[]
    train_accuracy?: number
    [k: string]: unknown
  }
}

// ── Drift Correction caret (spyde/actions/drift_action.py, plan A8) ──────────

/** Caret state. `window_id` is the SOURCE plot's window (where the caret
 *  lives); `check_window_id` is the bare-figure Drift Check window (whole-movie
 *  before/after sums on top, the ROI discovery pair beneath) and
 *  `trace_window_id` the bare-figure dy/dx window the solve opens. */
export interface DriftStateMessage extends MsgBase {
  type: 'drift_state'
  window_id: number | null
  check_window_id: number | null
  trace_window_id: number | null
  /** 'rigid' | 'rigid_affine' | 'nonrigid' — only rigid has a solver. */
  method: string
  solved: boolean
  /** Whether the FULL solve restricts itself to the alignment box. The preview
   *  always uses the box regardless — the toggle is the commitment. */
  use_roi: boolean
  /** The alignment box as `[y0, x0, h, w]` in IMAGE PIXELS (what
   *  `solve_translation(roi=…)` takes), or null when there is no usable box. */
  roi: number[] | null
  params: Record<string, unknown>
  /** Present on the open/ready emissions. */
  n_frames?: number
}

/** The discovery preview: ~20 frames sampled across the whole movie, aligned on
 *  the box alone. `gain` is the gradient energy of the aligned sum over the raw
 *  sum, measured on the pixels both cover — > 1.5 means a usable landmark,
 *  <= 1 means aligning there changes nothing. */
export interface DriftPreviewMessage extends MsgBase {
  type: 'drift_preview'
  window_id: number | null
  roi: number[] | null
  frames: number
  gain: number
  max_abs_shift: number
  params: Record<string, unknown>
}

/** A batch of solved shifts streamed from `solve_translation`'s `on_shift`
 *  while the solve runs — `[frame_index, dy, dx]` each. The backend paints
 *  these into the dy/dx window itself; the message exists so any host can
 *  follow the curve live. */
export interface DriftTraceMessage extends MsgBase {
  type: 'drift_trace'
  window_id: number | null
  points: [number, number, number][]
}

/** Whole-movie solve progress (also emitted as a plain `progress` message). */
export interface DriftProgressMessage extends MsgBase {
  type: 'drift_progress'
  window_id: number | null
  done: number
  total: number
}

/** The solved model. `shifts[i]` is the correction ADDED to frame i, `[dy, dx]`
 *  in pixels; a cancelled solve leaves NaN rows for frames it never reached. */
export interface DriftResultMessage extends MsgBase {
  type: 'drift_result'
  window_id: number | null
  shifts: [number, number][]
  kind: string
  reference: string
  max_abs_shift: number
  /** The box the solve correlated on (`use_roi`), or null for whole-frame. */
  roi: number[] | null
  /** Whole-movie sharpening: gradient energy of the corrected sum over the raw
   *  one. The same number the preview reports, now for what actually ran. */
  gain: number
  /** Frames dropped from the running reference by the outlier rejector. */
  rejected: number
  cancelled: boolean
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
    // EBSD Indexing: the dictionary-ready ack and the live best-match readout.
    | 'ebsd_dictionary_ready'
    | 'ebsd_match'
    // The Examples menu's contents, from em-database.
    | 'example_catalogue'
    // Fit wizard: the component palette's sampled shapes (once, on open) and
    // the whole model after every edit. See spyde/actions/fit_action.py.
    | 'fit_catalogue'
    | 'fit_state'
    // Remove Background: the model kind and where the dragged band is. The
    // caret follows the band, so it needs this relayed — without it the
    // fields sat at 0..0 while the band was somewhere else.
    | 'bg_state'
    | 'fv_auto_params'
    | 'fv_models'
    | 'fv_calibration'
    | 'cod_results'
    | 'cod_cif_ready'
    | 'gpu_status_result'
    | 'first_run_result'
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
  | WindowComputingMessage
  | WindowTitleMessage
  | StateUpdateMessage
  | StateUpdateBinaryMessage
  | CompositionMessage
  | MetadataMessage
  | AxesInfoMessage
  | OffsetPickMessage
  | ActionActiveMessage
  | SubItemMessage
  | HistogramMessage
  | NavShapePromptMessage
  | OpenPathMessage
  | LoadingMessage
  | SignalTypeInfoMessage
  | SelectorInfoMessage
  | SelectorRemovedMessage
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
  | ReportVectorsChoiceMessage
  | ReportExportedMessage
  | ReportPanelSelectedMessage
  | WizardEventMessage
  | LayersStateMessage
  | RepfigComposeOptionsMessage
  | MovieStateMessage
  | MovieFrameMessage
  | MovieDoneMessage
  | MovieEditOpenMessage
  | DownloadProgressMessage
  | DownloadDoneMessage
  | DaskStatsMessage
  | IoThroughputMessage
  | ParticlesTableMessage
  | SegStateMessage
  | SegPreviewMessage
  | SegTrainedMessage
  | DriftStateMessage
  | DriftPreviewMessage
  | DriftTraceMessage
  | DriftProgressMessage
  | DriftResultMessage

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
