/**
 * App.tsx — Ground Crew's fixed-pane layout.
 *
 * The counterpart to SpyDE's MDI workspace, and the reason the split is worth
 * doing: same shell, same backend protocol, completely different arrangement.
 * Panes are fixed — a control sidebar, one viewer, a stats strip, a status bar
 * and a collapsible log — because an operator driving a camera wants controls
 * in the same place every time, not windows to manage.
 *
 * Everything here is app UI. The pieces that clearly want to be shared —
 * the message reducer, the figure iframe host, the log panel, the status bar,
 * the sidebar host — are the shopping list for @de/shell-renderer, and are
 * written to be lifted rather than rewritten.
 */
import React, { useCallback, useEffect, useMemo, useReducer, useState } from 'react'
import {
  FigureFrame, useFigureBridge, shellReducer, shellInitialState, toShellAction,
} from '@de/shell-renderer'
import type { FigureMessage, ShellState, ShellAction, LogEntry } from '@de/shell-renderer'

// ── Backend message shapes ────────────────────────────────────────────────────

/** Frame statistics as the DE Server reports them.
 *
 * Every field is nullable because the server signals "not measured" with an
 * out-of-band sentinel, which the backend maps to `null` rather than letting a
 * NaN reach the UI. These arrive on the SAME `get_result` that returns the
 * pixels, so the strip costs no extra round trip — and they describe the whole
 * frame, not the cropped region on screen. */
interface FrameStats {
  frame: number | null
  min: number | null; max: number | null; mean: number | null; std: number | null
  eppix: number | null; eppixps: number | null
  over: number | null; under: number | null
}

const COLORMAPS = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
const EXPOSURES = [10, 25, 50, 100, 250, 500]

/** Ground Crew's own state, on top of the shell's chrome slice. */
interface AppState extends ShellState {
  figure: FigureMessage | null
  stats: FrameStats | null
  running: boolean
  exposureMs: number
  colormap: string
  error: string | null
}

type AppAction =
  | ShellAction
  | { type: 'FIGURE'; figure: FigureMessage }
  | { type: 'FRAME_STATS'; stats: FrameStats }
  | { type: 'ACQ_STATE'; running: boolean; exposureS?: number }
  | { type: 'ERROR'; text: string | null }
  | { type: 'EXPOSURE'; ms: number }
  | { type: 'COLORMAP'; name: string }

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'FIGURE':
      return { ...state, figure: action.figure }
    case 'FRAME_STATS':
      return { ...state, stats: action.stats }
    case 'ACQ_STATE':
      return {
        ...state,
        running: action.running,
        exposureMs: action.exposureS != null ? action.exposureS * 1000 : state.exposureMs,
      }
    case 'ERROR':
      return { ...state, error: action.text }
    case 'EXPOSURE':
      return { ...state, exposureMs: action.ms }
    case 'COLORMAP':
      return { ...state, colormap: action.name }
    default:
      // The shell owns status, the log ring, env setup and backend death.
      return shellReducer(state, action as ShellAction)
  }
}

const initialState: AppState = {
  ...shellInitialState,
  figure: null,
  stats: null,
  // Idle until the backend says otherwise: connecting is not acquiring, and
  // the camera is not put into free-run without someone asking.
  running: false,
  exposureMs: 50,
  colormap: 'gray',
  error: null,
}

export function App() {
  const [state, dispatch] = useReducer(appReducer, initialState)
  const [logOpen, setLogOpen] = useState(false)
  const bridge = useFigureBridge()
  const { figure, stats, running, exposureMs, colormap, error, status, logEntries } = state

  useEffect(() => {
    // Returns a disposer — see the preload's note on StrictMode double-invoke.
    const dispose = window.groundcrew?.onMessage((raw) => {
      const msg = raw as Record<string, unknown>

      // Chrome messages first — status, log backfill/level, env setup, backend
      // death — so this switch only carries Ground Crew's own.
      const shellAction = toShellAction(msg)
      if (shellAction) { dispatch(shellAction); return }

      switch (msg.type) {
        case 'figure':
          dispatch({ type: 'FIGURE', figure: msg as unknown as FigureMessage })
          break
        case 'state_update':
          bridge.applyState(String(msg.fig_id), String(msg.key), msg.value)
          break
        case 'state_update_binary':
          bridge.applyBinary(
            String(msg.fig_id), String(msg.key),
            (msg.header ?? {}) as Record<string, unknown>,
            msg.buffer as Uint8Array,
          )
          break
        case 'frame_stats':
          dispatch({ type: 'FRAME_STATS', stats: msg as unknown as FrameStats })
          break
        case 'acq_state':
          dispatch({
            type: 'ACQ_STATE',
            running: Boolean(msg.running),
            exposureS: typeof msg.exposure_s === 'number' ? msg.exposure_s : undefined,
          })
          break
        case 'error':
          dispatch({ type: 'ERROR', text: String(msg.text ?? '') })
          break
        case 'log':
          // One record per message rather than the shell's batched LOG: a
          // camera's log rate is nothing like a compute's, so there is no burst
          // to coalesce.
          dispatch({ type: 'LOG', entries: [msg as unknown as LogEntry] })
          break
        default:
          break
      }
    })
    return () => dispose?.()
  }, [bridge])

  const act = useCallback((action: string, payload: Record<string, unknown> = {}) => {
    window.groundcrew?.action(action, payload)
  }, [])

  return (
    <div style={S.root}>
      <header style={S.titlebar}>
        <span style={S.brand}>● Ground Crew</span>
        <span style={S.subtle}>Direct Electron</span>
      </header>

      <div style={S.body}>
        <ControlPanel
          running={running}
          exposureMs={exposureMs}
          colormap={colormap}
          onStart={() => act('start_acquisition')}
          onStop={() => act('stop_acquisition')}
          onSingle={() => act('single_acquisition')}
          /* Optimistic local update AND the backend call: the control should
             respond to the click, not wait for the round trip. The backend's
             acq_state reply is authoritative and corrects it if it disagrees. */
          onExposure={(ms) => {
            dispatch({ type: 'EXPOSURE', ms })
            act('set_property', { name: 'Exposure Time (seconds)', value: ms / 1000 })
          }}
          onColormap={(name) => {
            dispatch({ type: 'COLORMAP', name })
            act('set_colormap', { name })
          }}
        />

        <main style={S.center}>
          <FigurePane figure={figure} bridge={bridge} />
          <StatsStrip stats={stats} />
        </main>
      </div>

      {logOpen && <LogPanel logs={logEntries} onClose={() => setLogOpen(false)} />}

      <StatusBar
        status={status}
        error={error}
        running={running}
        onDismissError={() => dispatch({ type: 'ERROR', text: null })}
        logOpen={logOpen}
        onToggleLog={() => setLogOpen((v) => !v)}
      />
    </div>
  )
}

// ── The fixed left sidebar (the PySide6 app's control_panel) ──────────────────

function ControlPanel(props: {
  running: boolean
  exposureMs: number
  colormap: string
  onStart: () => void
  onStop: () => void
  onSingle: () => void
  onExposure: (ms: number) => void
  onColormap: (name: string) => void
}) {
  return (
    <aside style={S.sidebar} data-testid="control-panel">
      <Section title="Acquisition">
        <div style={S.row}>
          <button
            style={{ ...S.btn, ...(props.running ? S.btnActive : {}) }}
            onClick={props.onStart}
            disabled={props.running}
            data-testid="start-btn"
          >
            ▶ Start
          </button>
          <button
            style={S.btn}
            onClick={props.onStop}
            disabled={!props.running}
            data-testid="stop-btn"
          >
            ■ Stop
          </button>
        </div>
        <button
          style={{ ...S.btn, width: '100%', marginTop: 8 }}
          onClick={props.onSingle}
          disabled={props.running}
          title={props.running ? 'Stop the live view first' : 'Take one exposure'}
          data-testid="single-btn"
        >
          ◉ Single exposure
        </button>
      </Section>

      <Section title="Exposure">
        <div style={S.chips}>
          {EXPOSURES.map((ms) => (
            <button
              key={ms}
              style={{ ...S.chip, ...(props.exposureMs === ms ? S.chipOn : {}) }}
              onClick={() => props.onExposure(ms)}
            >
              {ms} ms
            </button>
          ))}
        </div>
      </Section>

      <Section title="Display">
        <label style={S.label}>Colormap</label>
        <select
          style={S.select}
          value={props.colormap}
          onChange={(e) => props.onColormap(e.target.value)}
          data-testid="colormap-select"
        >
          {COLORMAPS.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </Section>

      <Section title="Camera">
        <Field label="Type" value="DESim (simulated)" />
        <Field label="Server" value="127.0.0.1:13240" />
      </Section>
    </aside>
  )
}

// ── The viewer ────────────────────────────────────────────────────────────────

function FigurePane({ figure, bridge }: {
  figure: FigureMessage | null
  bridge: ReturnType<typeof useFigureBridge>
}) {
  // The backend needs the pane's size or the figure renders at anyplotlib's
  // default and overflows. Stable identity, so FigureFrame's ResizeObserver
  // effect does not re-run every render.
  const onResize = useCallback((w: number, h: number) => {
    if (figure) window.groundcrew?.resizeFigure(figure.fig_id, w, h)
  }, [figure?.fig_id])

  if (!figure) {
    return (
      <div style={{ ...S.viewer, ...S.placeholder }} data-testid="viewer-placeholder">
        Waiting for the first frame…
      </div>
    )
  }

  return (
    <div style={S.viewer}>
      {/* srcdoc (`html`), not a URL: this backend inlines the figure's ESM, so
          nothing has to be served off disk. FigureFrame owns registration,
          replay-on-load and the resize reporting. */}
      <FigureFrame
        bridge={bridge}
        figId={figure.fig_id}
        html={figure.html}
        title={figure.title ?? 'Live view'}
        onResize={onResize}
        style={S.iframe}
        data-testid="viewer-frame"
      />
    </div>
  )
}

/** `null` renders as "—": the server reports a value it did not measure with an
 *  out-of-band sentinel, and showing a placeholder is honest where showing 0
 *  would be a claim. */
const num = (v: number | null | undefined, digits = 0) =>
  v == null ? '—' : v.toFixed(digits)

function StatsStrip({ stats }: { stats: FrameStats | null }) {
  if (!stats) return <div style={S.stats} data-testid="stats-strip" />
  return (
    <div style={S.stats} data-testid="stats-strip">
      <Stat label="Frame" value={stats.frame == null ? '—' : String(stats.frame)}
        testId="stat-frame" />
      <Stat label="Min" value={num(stats.min)} testId="stat-min" />
      <Stat label="Max" value={num(stats.max)} testId="stat-max" />
      <Stat label="Mean" value={num(stats.mean, 1)} />
      <Stat label="Std" value={num(stats.std, 1)} />
      {/* Electrons per pixel — the number that says whether the exposure is
          right, and the reason to read statistics off the server rather than
          recompute them from the tile: they describe the FULL frame, not the
          cropped region currently on screen. */}
      <Stat label="e⁻/pix" value={num(stats.eppix, 2)} />
      <Stat label="e⁻/pix/s" value={num(stats.eppixps, 1)} />
      <Stat label="Over" value={num(stats.over, 2)} />
      <Stat label="Under" value={num(stats.under, 2)} />
    </div>
  )
}

function StatusBar(props: {
  status: string; error: string | null; running: boolean
  onDismissError: () => void; logOpen: boolean; onToggleLog: () => void
}) {
  return (
    <footer style={S.statusbar}>
      <span style={{ ...S.dot, background: props.running ? '#41d18a' : '#7a8296' }} />
      <span data-testid="status-text">{props.error ?? props.status}</span>
      {props.error && (
        <button style={S.linkBtn} onClick={props.onDismissError}>dismiss</button>
      )}
      <span style={{ flex: 1 }} />
      <button style={S.linkBtn} onClick={props.onToggleLog}>
        {props.logOpen ? 'Hide log' : 'Log'}
      </button>
    </footer>
  )
}

function LogPanel({ logs, onClose }: { logs: LogEntry[]; onClose: () => void }) {
  const [area, setArea] = useState('all')
  const areas = useMemo(
    () => ['all', ...Array.from(new Set(logs.map((l) => l.area))).sort()], [logs])
  const shown = area === 'all' ? logs : logs.filter((l) => l.area === area)
  return (
    <section style={S.logPanel} data-testid="log-panel">
      <div style={S.logHead}>
        <strong>Log</strong>
        <select style={S.selectSm} value={area} onChange={(e) => setArea(e.target.value)}>
          {areas.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
        <span style={{ flex: 1 }} />
        <button style={S.linkBtn} onClick={onClose}>close</button>
      </div>
      <div style={S.logBody}>
        {shown.slice(-200).map((l, i) => (
          <div key={i} style={S.logLine}>
            <span style={S.logArea}>{l.area}</span>
            <span style={{ color: LEVEL_COLORS[l.level] ?? '#c8cee0' }}>{l.msg}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: '#7a8296', INFO: '#c8cee0', WARNING: '#e2b04a',
  ERROR: '#ef6b6b', CRITICAL: '#ef6b6b',
}

// ── Small presentational helpers ──────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={S.section}>
      <div style={S.sectionTitle}>{title}</div>
      {children}
    </div>
  )
}
function Field({ label, value }: { label: string; value: string }) {
  return (
    <div style={S.field}>
      <span style={S.subtle}>{label}</span>
      <span>{value}</span>
    </div>
  )
}
function Stat({ label, value, testId }: { label: string; value: string; testId?: string }) {
  return (
    <div style={S.stat}>
      <span style={S.statLabel}>{label}</span>
      {/* The testid goes on the VALUE, not the row: the label is uppercased by
          CSS, and innerText reflects text-transform, so scraping the row and
          regexing for "Frame" silently matches nothing. */}
      <span style={S.statValue} data-testid={testId}>{value}</span>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────

const S: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex', flexDirection: 'column', height: '100vh', margin: 0,
    background: '#14161c', color: '#e6e9f0',
    font: '13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    overflow: 'hidden',
  },
  titlebar: {
    display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px',
    borderBottom: '1px solid #262a35', background: '#171a22', flexShrink: 0,
  },
  brand: { fontWeight: 600, color: '#8ab4ff' },
  subtle: { color: '#7a8296' },
  body: { display: 'flex', flex: 1, minHeight: 0 },
  sidebar: {
    width: 240, flexShrink: 0, borderRight: '1px solid #262a35',
    background: '#171a22', padding: 14, overflowY: 'auto',
  },
  center: { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, minHeight: 0 },
  viewer: { flex: 1, minHeight: 0, padding: 12, display: 'flex' },
  placeholder: { alignItems: 'center', justifyContent: 'center', color: '#7a8296' },
  iframe: { flex: 1, border: '1px solid #262a35', borderRadius: 8, background: '#0d0f14' },
  stats: {
    display: 'flex', gap: 18, padding: '8px 14px', flexShrink: 0,
    borderTop: '1px solid #262a35', background: '#171a22', minHeight: 34,
  },
  stat: { display: 'flex', flexDirection: 'column' },
  statLabel: { color: '#7a8296', fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.4 },
  statValue: { fontVariantNumeric: 'tabular-nums' },
  statusbar: {
    display: 'flex', alignItems: 'center', gap: 10, padding: '6px 14px',
    borderTop: '1px solid #262a35', background: '#12141a', flexShrink: 0,
  },
  dot: { width: 8, height: 8, borderRadius: 4, display: 'inline-block' },
  section: { marginBottom: 20 },
  sectionTitle: {
    color: '#7a8296', fontSize: 10, textTransform: 'uppercase',
    letterSpacing: 0.6, marginBottom: 8,
  },
  row: { display: 'flex', gap: 8 },
  btn: {
    flex: 1, padding: '7px 10px', borderRadius: 6, cursor: 'pointer',
    border: '1px solid #2f3442', background: '#1e2230', color: '#e6e9f0',
  },
  btnActive: { borderColor: '#41d18a55', color: '#41d18a' },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 6 },
  chip: {
    padding: '4px 9px', borderRadius: 999, cursor: 'pointer', fontSize: 12,
    border: '1px solid #2f3442', background: '#1e2230', color: '#c8cee0',
  },
  chipOn: { background: '#22304a', borderColor: '#8ab4ff', color: '#8ab4ff' },
  label: { display: 'block', color: '#7a8296', marginBottom: 4 },
  select: {
    width: '100%', padding: '6px 8px', borderRadius: 6,
    border: '1px solid #2f3442', background: '#1e2230', color: '#e6e9f0',
  },
  selectSm: {
    padding: '2px 6px', borderRadius: 5,
    border: '1px solid #2f3442', background: '#1e2230', color: '#e6e9f0',
  },
  field: { display: 'flex', justifyContent: 'space-between', padding: '3px 0' },
  linkBtn: {
    background: 'none', border: 'none', color: '#8ab4ff',
    cursor: 'pointer', padding: 0, font: 'inherit',
  },
  logPanel: {
    height: 200, flexShrink: 0, borderTop: '1px solid #262a35',
    background: '#12141a', display: 'flex', flexDirection: 'column',
  },
  logHead: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '6px 14px', borderBottom: '1px solid #262a35',
  },
  logBody: { flex: 1, overflowY: 'auto', padding: '6px 14px', fontFamily: 'ui-monospace, monospace' },
  logLine: { display: 'flex', gap: 10, fontSize: 12 },
  logArea: { color: '#7a8296', minWidth: 80 },
}
