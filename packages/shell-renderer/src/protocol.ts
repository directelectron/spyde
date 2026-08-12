/**
 * protocol.ts — the CORE PLOTAPP messages, shared by every shell app.
 *
 * The Python backend emits JSON over stdout; the Electron main process relays it
 * to the renderer. Each message is discriminated by `type`. What is modelled
 * here is the subset with the same meaning in SpyDE, de-groundcrew and
 * de-autopilot — lifecycle, figures, figure state, logging, progress.
 *
 * Everything domain-specific (SpyDE's report/movie/drift/vectors, Ground Crew's
 * frame stats) is the app's own union, which EXTENDS this one. That works
 * additively because every variant carries an index signature: reading a field
 * this file doesn't model is allowed and surfaces as `unknown`, rather than
 * being a compile error. This types the shape the renderer relies on; it is not
 * a schema.
 *
 * `backend_exited` is synthesised by @de/shell-main's backendProcess, not a real
 * PLOTAPP line.
 */

/** Any field not explicitly modelled is still readable (as `unknown`). */
export interface MsgBase {
  [k: string]: unknown
}

export interface ReadyMessage extends MsgBase {
  type: 'ready'
}

export interface StatusMessage extends MsgBase {
  type: 'status'
  text: string
}

export interface ErrorMessage extends MsgBase {
  type: 'error'
  text: string
}

/** Progress of a heavy backend action. `done >= total`, or `total <= 0`, clears. */
export interface ProgressMessage extends MsgBase {
  type: 'progress'
  done: number
  total: number
  label?: string
}

export interface BackendExitedMessage extends MsgBase {
  type: 'backend_exited'
  code: number | null
  /** Set when main synthesises this for a packaged env-setup failure, as
   *  distinct from a plain runtime death. */
  reason?: string
}

/** First-run Python environment setup, parsed from `uv` output by the main
 *  process (envProgress.ts). `start`/`done` bracket the run. */
export interface EnvSetupMessage extends MsgBase {
  type: 'env_setup'
  event: 'start' | 'progress' | 'done'
  phase?: 'resolving' | 'downloading' | 'installing' | 'building' | 'torch' | 'working'
  step?: string
  percent?: number | null
  raw?: string
}

/** A figure to mount. Carries EITHER inline `html` (mounted via srcdoc) or a
 *  `file_url` served through the app's figure scheme — see the note in
 *  FigureFrame about which and why. */
export interface FigureMessage extends MsgBase {
  type: 'figure'
  window_id: number
  fig_id: string
  html?: string
  file_url?: string | null
  title?: string
  is_navigator?: boolean
  /** Image width/height, so a host can size the pane to the data. */
  aspect?: number
}

/** An anyplotlib state change, forwarded into the figure iframe. */
export interface StateUpdateMessage extends MsgBase {
  type: 'state_update'
  fig_id: string
  key: string
  value: unknown
}

/** A raw pixel frame: bytes rather than base64. `header.geom` names the panel —
 *  load-bearing for retention, see figureBridge. */
export interface StateUpdateBinaryMessage extends MsgBase {
  type: 'state_update_binary'
  fig_id: string
  key: string
  header?: Record<string, unknown>
  buffer: Uint8Array
}

export interface WindowClosedMessage extends MsgBase {
  type: 'window_closed'
  window_id: number
}

export interface WindowTitleMessage extends MsgBase {
  type: 'window_title'
  window_id: number
  title: string
}

/** Drives the floating translucent "Calculating…" chip over one window. */
export interface WindowComputingMessage extends MsgBase {
  type: 'window_computing'
  window_id: number
  computing: boolean
}

/** One application-log record streamed from the backend. `area` is the
 *  subsystem tag the app registered (see de_shell.log_stream). */
export interface LogMessage extends MsgBase {
  type: 'log'
  level: string
  name: string
  area: string
  msg: string
  time: number
}

export interface LogBackfillMessage extends MsgBase {
  type: 'log_backfill'
  entries: Array<Record<string, unknown>>
}

export interface LogLevelMessage extends MsgBase {
  type: 'log_level'
  level: string
}

/** The core union. An app extends it:
 *
 *     type MyMessage = ShellMessage | MyDomainMessage | …
 */
export type ShellMessage =
  | ReadyMessage
  | StatusMessage
  | ErrorMessage
  | ProgressMessage
  | BackendExitedMessage
  | EnvSetupMessage
  | FigureMessage
  | StateUpdateMessage
  | StateUpdateBinaryMessage
  | WindowClosedMessage
  | WindowTitleMessage
  | WindowComputingMessage
  | LogMessage
  | LogBackfillMessage
  | LogLevelMessage

/** Narrow a raw IPC payload to the union. The cast is the point: `type` is the
 *  discriminator and every variant tolerates unmodelled fields, so this is a
 *  narrowing step rather than validation. */
export function asShellMessage(msg: Record<string, unknown>): ShellMessage {
  return msg as unknown as ShellMessage
}
