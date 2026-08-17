/**
 * shellState.ts — the app-chrome state every shell app keeps.
 *
 * Status line, busy indicator, the log ring, first-run environment setup, the
 * "backend died" latch, per-window computing overlays, and toolbar action
 * state. None of it knows what the data is; all of it was written twice (SpyDE
 * in a reducer, Ground Crew in ad-hoc `useState`) before landing here.
 *
 * **The window/figure registry is deliberately NOT here.** SpyDE's is entangled
 * with its named-view/chip system (`view`, `view_label`, strain components) and
 * Ground Crew has no window registry at all — one fixed pane. Extracting it now
 * would be generalising from a single consumer, so it stays in SpyDE until a
 * second app actually needs it.
 *
 * ## Composing
 *
 * An app's state EXTENDS `ShellState` and its reducer delegates:
 *
 *     function appReducer(state: AppState, action: AppAction): AppState {
 *       switch (action.type) {
 *         case 'MY_THING': return …
 *         default: return shellReducer(state, action as ShellAction)
 *       }
 *     }
 *
 * `shellReducer` is generic over the state type and returns it unchanged for an
 * action it does not own, so the delegation is safe in either order.
 */

/**
 * One application-log record streamed from the backend.
 *
 * `seq` is a renderer-assigned monotonic id, stamped once as the record enters
 * the buffer. It is the STABLE React key + height-cache key for a virtualised
 * log list: the buffer is a ring (old records drop off the front), so an array
 * INDEX identifies a different record after every shift, which would defeat
 * both row memoisation and a measured-height cache.
 *
 * `area` is the subsystem tag the app registered (see de_shell.log_stream), and
 * is optional because a record can predate registration.
 */
export interface LogEntry {
  level: string
  name: string
  area?: string
  msg: string
  time: number
  seq?: number
}

export interface SubItem {
  name: string
  color: string
  vtype?: string
  calculation?: string
}

export type EnvPhase =
  | 'resolving' | 'downloading' | 'installing' | 'building' | 'torch' | 'working'

export interface EnvSetupState {
  phase: EnvPhase
  /** Friendly current-step headline. */
  step: string
  /** 0–100 for a download we can measure, else null. */
  percent: number | null
  /** Rolling raw output tail (bounded). */
  lines: string[]
}

export interface ShellState {
  status: string
  ready: boolean
  /** Long file-read / open busy indicator. */
  loading: { busy: boolean; text: string }
  /** Raw stdout/stderr from the backend (bounded). */
  streamLines: Array<{ text: string; kind: 'stdout' | 'stderr' }>
  /** Application-log records (the log panel). */
  logEntries: LogEntry[]
  /** Current backend verbosity (DEBUG…CRITICAL). */
  logLevel: string
  /** Set when the Python sidecar dies; surfaces a blocking banner. */
  backendExited: { code: number | null; reason?: string } | null
  /** First-run `uv sync` progress; drives the floating setup overlay. */
  envSetup: EnvSetupState | null
  /** windowIds with a long compute in flight (→ floating overlay). */
  computingWindows: Set<number>
  /** windowId → action names with live output. */
  activeActions: Map<number, Set<string>>
  /** windowId → action → dynamic chips. */
  subItems: Map<number, Map<string, SubItem[]>>
}

/** Max buffered log records (the renderer-side ring buffer). */
export const LOG_MAX = 1000

/** Matching bound for raw stdout/stderr, which is noisier and less useful. */
export const STREAM_MAX = 500

export const shellInitialState: ShellState = {
  status: 'Starting…',
  ready: false,
  loading: { busy: false, text: '' },
  streamLines: [],
  logEntries: [],
  logLevel: 'INFO',
  backendExited: null,
  envSetup: null,
  computingWindows: new Set(),
  activeActions: new Map(),
  subItems: new Map(),
}

export type ShellAction =
  | { type: 'STATUS'; text: string }
  | { type: 'LOADING'; busy: boolean; text: string }
  | { type: 'STREAM'; text: string; kind: 'stdout' | 'stderr' }
  | { type: 'LOG'; entries: LogEntry[] }
  | { type: 'LOG_BACKFILL'; entries: LogEntry[] }
  | { type: 'LOG_LEVEL'; level: string }
  | { type: 'BACKEND_EXITED'; code: number | null; reason?: string }
  | { type: 'ENV_SETUP_START' }
  | { type: 'ENV_SETUP_PROGRESS'; phase?: EnvPhase; step?: string; percent: number | null; raw: string }
  | { type: 'ENV_SETUP_DONE' }
  | { type: 'WINDOW_COMPUTING'; windowId: number; computing: boolean }
  | { type: 'ACTION_ACTIVE'; windowId: number; name: string; active: boolean }
  | { type: 'SUB_ITEM'; windowId: number; action: string; name: string; color: string; vtype?: string; calculation?: string; active: boolean }

/**
 * Reduce the shell's slice. Generic over the app's state so it composes as a
 * `default:` branch; returns *state* untouched for anything it does not own.
 */
export function shellReducer<S extends ShellState>(state: S, action: ShellAction): S {
  switch (action.type) {
    case 'STATUS':
      return { ...state, status: action.text }

    case 'LOADING':
      return { ...state, loading: { busy: action.busy, text: action.text } }

    case 'STREAM':
      return {
        ...state,
        streamLines: [...state.streamLines.slice(-STREAM_MAX),
          { text: action.text, kind: action.kind }],
      }

    // A BATCH of records (hosts coalesce per animation frame) — one array copy
    // and one render for a whole burst, instead of one per line.
    case 'LOG': {
      if (action.entries.length === 0) return state
      const merged = state.logEntries.concat(action.entries)
      return {
        ...state,
        logEntries: merged.length > LOG_MAX ? merged.slice(-LOG_MAX) : merged,
      }
    }

    case 'LOG_BACKFILL':
      return { ...state, logEntries: action.entries.slice(-LOG_MAX) }

    case 'LOG_LEVEL':
      return { ...state, logLevel: action.level }

    case 'BACKEND_EXITED':
      return {
        ...state,
        backendExited: { code: action.code, reason: action.reason },
        // A setup failure surfaces via BACKEND_EXITED — drop the setup overlay
        // so the two don't stack.
        envSetup: null,
        ready: false,
        status: 'Backend stopped',
      }

    case 'ENV_SETUP_START':
      return {
        ...state,
        envSetup: {
          phase: 'resolving',
          step: 'Preparing the analysis environment',
          percent: null,
          lines: [],
        },
        status: 'Setting up the analysis environment…',
      }

    case 'ENV_SETUP_PROGRESS': {
      const prev = state.envSetup ?? {
        phase: 'resolving' as EnvPhase,
        step: 'Preparing the analysis environment',
        percent: null,
        lines: [],
      }
      // Keep the last meaningful step/phase when a noisy line parses to nothing;
      // always append the raw line to the bounded tail so it visibly moves.
      const lines = [...prev.lines, action.raw].slice(-200)
      return {
        ...state,
        envSetup: {
          phase: action.phase ?? prev.phase,
          step: action.step ?? prev.step,
          percent: action.percent,
          lines,
        },
      }
    }

    case 'ENV_SETUP_DONE':
      return { ...state, envSetup: null }

    case 'WINDOW_COMPUTING': {
      const has = state.computingWindows.has(action.windowId)
      if (action.computing === has) return state   // no-op re-emit
      const computingWindows = new Set(state.computingWindows)
      if (action.computing) computingWindows.add(action.windowId)
      else computingWindows.delete(action.windowId)
      return { ...state, computingWindows }
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

    default:
      return state
  }
}

/**
 * Map a backend message to a shell action, or null if the shell does not own it.
 *
 * Lets an app route the chrome messages without restating them:
 *
 *     const shellAction = toShellAction(msg)
 *     if (shellAction) { dispatch(shellAction); return }
 *
 * `log` is deliberately NOT handled here. Hosts coalesce log records per frame
 * before dispatching a batched `LOG` — mapping one message to one action would
 * undo that and re-render per line.
 */
export function toShellAction(msg: Record<string, unknown>): ShellAction | null {
  switch (msg.type) {
    case 'status':
      return { type: 'STATUS', text: String(msg.text ?? '') }
    case 'loading':
      return { type: 'LOADING', busy: Boolean(msg.busy), text: String(msg.text ?? '') }
    case 'log_backfill':
      return { type: 'LOG_BACKFILL', entries: (msg.entries ?? []) as LogEntry[] }
    case 'log_level':
      return { type: 'LOG_LEVEL', level: String(msg.level ?? 'INFO') }
    case 'backend_exited':
      return {
        type: 'BACKEND_EXITED',
        code: (msg.code ?? null) as number | null,
        reason: msg.reason as string | undefined,
      }
    case 'window_computing':
      return {
        type: 'WINDOW_COMPUTING',
        windowId: Number(msg.window_id),
        computing: Boolean(msg.computing),
      }
    case 'env_setup':
      if (msg.event === 'start') return { type: 'ENV_SETUP_START' }
      if (msg.event === 'done') return { type: 'ENV_SETUP_DONE' }
      return {
        type: 'ENV_SETUP_PROGRESS',
        phase: msg.phase as EnvPhase | undefined,
        step: msg.step as string | undefined,
        percent: (msg.percent ?? null) as number | null,
        raw: String(msg.raw ?? ''),
      }
    default:
      return null
  }
}
