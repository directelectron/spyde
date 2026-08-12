/**
 * App.tsx — Ground Crew's layout: a mode rail, and fixed panes inside it.
 *
 * The counterpart to SpyDE's MDI workspace, and the reason the split is worth
 * doing: same shell, same backend protocol, completely different arrangement.
 * An operator driving a camera wants controls in the same place every time,
 * not windows to manage.
 *
 *     ┌──────────────────────────────────────────────────────┐
 *     │ brand      temperature · camera position     top bar │
 *     ├────┬──────────────┬───────────────────┬──────────────┤
 *     │mode│  instrument  │      mode pane    │ plot control │
 *     │rail│  state and   │                   │  histogram   │
 *     │    │  control     │                   │  colormap    │
 *     ├────┴──────────────┴───────────────────┴──────────────┤
 *     │ status bar                                       Log │
 *     └──────────────────────────────────────────────────────┘
 *
 * The rail carries the four things this app does. The instrument sidebar and
 * the plot control persist across modes, because camera state is context for
 * all of them and the histogram belongs to whatever image is on screen.
 *
 * Everything here is app UI. The pieces that clearly want to be shared — the
 * message reducer, the figure iframe host, the log panel, the status bar — are
 * the shopping list for @de/shell-renderer, and are written to be lifted.
 */
import React, { useCallback, useEffect, useMemo, useReducer, useState } from 'react'
import {
  FigureFrame, useFigureBridge, shellReducer, shellInitialState, toShellAction,
} from '@de/shell-renderer'
import type { FigureMessage, ShellState, ShellAction, LogEntry } from '@de/shell-renderer'

import { C, FONT_MONO, stateOf } from './theme'
import { Btn, Field, NotBuilt, Pill, Section, Select } from './ui'
import { StatusMode } from './modes/Status'
import type { StatusCard, StatusSummary } from './modes/Status'
import { CalibrateMode, MotionMode } from './modes/Other'

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
  levels?: [number, number] | null
}

interface Connection {
  connected: boolean
  camera?: string | null; server?: string | null
  width?: number; height?: number; fake?: boolean
  error?: string
}

const MODES = [
  { key: 'imaging', label: 'Imaging', glyph: '◉' },
  { key: 'motion', label: 'Motion', glyph: '◈' },
  { key: 'calibrate', label: 'Calibrate', glyph: '⊹' },
  { key: 'status', label: 'Status', glyph: '❋' },
] as const
type ModeKey = typeof MODES[number]['key']

const COLORMAPS = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

/** Properties the instrument sidebar reads and writes. */
const P = {
  fps: 'Frames Per Second',
  exposure: 'Exposure Time (seconds)',
  frameCount: 'Frame Count',
  binX: 'Binning X',
  binY: 'Binning Y',
  hwBinX: 'Hardware Binning X',
  sizeX: 'Image Size X (pixels)',
  sizeY: 'Image Size Y (pixels)',
  autosave: 'Autosave Directory',
  detTemp: 'Temperature - Detector (Celsius)',
  coolSetpoint: 'Temperature - Cool Down Setpoint (Celsius)',
  tempControl: 'Temperature - Control',
  position: 'Camera Position Status',
  positionCtl: 'Camera Position Control',
} as const

// ── State ─────────────────────────────────────────────────────────────────────

interface AppState extends ShellState {
  figure: FigureMessage | null
  stats: FrameStats | null
  running: boolean
  colormap: string
  connection: Connection | null
  props: Record<string, unknown>
  statusCards: StatusCard[]
  statusSummary: StatusSummary | null
  /** The last error, shown in the status bar until dismissed. App-level rather
   *  than shell-level: the shell's chrome slice has no dismissable error. */
  error: string | null
}

type AppAction =
  | ShellAction
  | { type: 'FIGURE'; figure: FigureMessage }
  | { type: 'FRAME_STATS'; stats: FrameStats }
  | { type: 'ACQ_STATE'; running: boolean }
  | { type: 'COLORMAP'; name: string }
  | { type: 'CONNECTION'; connection: Connection }
  | { type: 'PROPERTIES'; values: Record<string, unknown> }
  | { type: 'STATUS_REPORT'; cards: StatusCard[]; summary: StatusSummary }
  | { type: 'APP_ERROR'; text: string | null }

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'FIGURE': return { ...state, figure: action.figure }
    case 'FRAME_STATS': return { ...state, stats: action.stats }
    case 'ACQ_STATE': return { ...state, running: action.running }
    case 'COLORMAP': return { ...state, colormap: action.name }
    case 'CONNECTION': return { ...state, connection: action.connection }
    // Merge rather than replace: different callers poll different subsets, and
    // a partial poll must not blank the fields it did not ask for.
    case 'PROPERTIES': return { ...state, props: { ...state.props, ...action.values } }
    case 'STATUS_REPORT':
      return { ...state, statusCards: action.cards, statusSummary: action.summary }
    case 'APP_ERROR': return { ...state, error: action.text }
    default:
      return shellReducer(state, action as ShellAction)
  }
}

const initialState: AppState = {
  ...shellInitialState,
  figure: null, stats: null, running: false, colormap: 'gray',
  connection: null, props: {}, statusCards: [], statusSummary: null, error: null,
}

// ── App ───────────────────────────────────────────────────────────────────────

export function App() {
  const [state, dispatch] = useReducer(appReducer, initialState)
  const [mode, setMode] = useState<ModeKey>('imaging')
  const [logOpen, setLogOpen] = useState(false)
  const bridge = useFigureBridge()
  const { figure, stats, running, colormap, error, status, logEntries,
          connection, props, statusCards, statusSummary } = state

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
          dispatch({ type: 'FIGURE', figure: msg as unknown as FigureMessage }); break
        case 'state_update':
          bridge.applyState(String(msg.fig_id), String(msg.key), msg.value); break
        case 'state_update_binary':
          bridge.applyBinary(
            String(msg.fig_id), String(msg.key),
            (msg.header ?? {}) as Record<string, unknown>,
            msg.buffer as Uint8Array,
          ); break
        case 'frame_stats':
          dispatch({ type: 'FRAME_STATS', stats: msg as unknown as FrameStats }); break
        case 'acq_state':
          dispatch({ type: 'ACQ_STATE', running: Boolean(msg.running) }); break
        case 'connection':
          dispatch({ type: 'CONNECTION', connection: msg as unknown as Connection }); break
        case 'properties':
          dispatch({
            type: 'PROPERTIES',
            values: (msg.values ?? {}) as Record<string, unknown>,
          }); break
        case 'status_report':
          dispatch({
            type: 'STATUS_REPORT',
            cards: (msg.cards ?? []) as StatusCard[],
            summary: msg.summary as StatusSummary,
          }); break
        case 'error':
          dispatch({ type: 'APP_ERROR', text: String(msg.text ?? '') }); break
        case 'log':
          // One record per message rather than the shell's batched LOG: a
          // camera's log rate is nothing like a compute's, so there is no
          // burst to coalesce.
          dispatch({ type: 'LOG', entries: [msg as unknown as LogEntry] }); break
        default: break
      }
    })
    return () => dispose?.()
  }, [bridge])

  const act = useCallback((action: string, payload: Record<string, unknown> = {}) => {
    window.groundcrew?.action(action, payload)
  }, [])

  const setProp = useCallback((name: string, value: unknown) => {
    act('set_property', { name, value })
  }, [act])

  // Read the board when Status is opened, not on a timer. Every property is a
  // round trip on the one connection the camera has, so polling a screen
  // nobody is looking at would contend with acquisition for nothing.
  useEffect(() => {
    if (mode === 'status' && connection?.connected) act('refresh_status')
  }, [mode, connection?.connected, act])

  return (
    <div style={S.root}>
      <TopBar connection={connection} props={props} onSet={setProp} />

      <div style={S.body}>
        <ModeRail mode={mode} onMode={setMode} summary={statusSummary} />
        <InstrumentPanel props={props} connection={connection} onSet={setProp}
          onRefresh={() => act('refresh_properties')} />

        <main style={S.center}>
          <div style={S.pane}>
            {mode === 'imaging' && (
              <ImagingMode figure={figure} bridge={bridge} running={running}
                onStart={() => act('start_acquisition')}
                onStop={() => act('stop_acquisition')}
                onSingle={() => act('single_acquisition')} />
            )}
            {mode === 'motion' && <MotionMode />}
            {mode === 'calibrate' && <CalibrateMode />}
            {mode === 'status' && (
              <StatusMode cards={statusCards} summary={statusSummary}
                onRefresh={() => act('refresh_status')} />
            )}
          </div>
          {mode === 'imaging' && <StatsStrip stats={stats} />}
        </main>

        <PlotControl stats={stats} colormap={colormap}
          onColormap={(name) => { dispatch({ type: 'COLORMAP', name }); act('set_colormap', { name }) }} />
      </div>

      {logOpen && <LogPanel logs={logEntries} onClose={() => setLogOpen(false)} />}

      <StatusBar status={status} error={error} running={running}
        onDismissError={() => dispatch({ type: 'APP_ERROR', text: null })}
        logOpen={logOpen} onToggleLog={() => setLogOpen((v) => !v)} />
    </div>
  )
}

// ── Top bar ───────────────────────────────────────────────────────────────────

/**
 * Temperature and camera position live up here, not in a mode.
 *
 * They are the two pieces of hardware state that matter regardless of what
 * you are doing, and the two whose controls are dangerous enough to want in a
 * fixed, predictable place. Both degrade to "—" when the server does not
 * report them, and their controls disable rather than disappear: a control
 * that vanishes reads as a missing feature, one that is greyed out reads as an
 * unavailable reading, which is the truth.
 */
function TopBar({ connection, props, onSet }: {
  connection: Connection | null
  props: Record<string, unknown>
  onSet: (name: string, value: unknown) => void
}) {
  const [openTemp, setOpenTemp] = useState(false)
  const temp = props[P.detTemp]
  const position = props[P.position]
  const hasTemp = temp != null
  const hasPosition = position != null

  return (
    <header style={S.topbar}>
      <span style={S.brand}>● Ground Crew</span>
      <span style={{ ...S.subtle, fontSize: 12 }}>
        {connection?.camera ?? 'Direct Electron'}
        {connection?.fake && <span style={S.simTag}>SIMULATED</span>}
      </span>

      <span style={{ flex: 1 }} />

      <div style={{ position: 'relative' }}>
        <button
          style={{ ...S.topBtn, opacity: hasTemp ? 1 : 0.5 }}
          disabled={!hasTemp}
          onClick={() => setOpenTemp((v) => !v)}
          data-testid="temp-btn"
          title={hasTemp ? 'Set the cooling setpoint'
                         : 'This server does not report detector temperature'}
        >
          <span style={{ color: C.textMuted, fontSize: 11 }}>SENSOR</span>
          <span style={{ fontFamily: FONT_MONO, fontVariantNumeric: 'tabular-nums' }}>
            {hasTemp ? `${Number(temp).toFixed(1)} °C` : '—'}
          </span>
        </button>
        {openTemp && hasTemp && (
          <div style={S.popover}>
            <Field label="Cool to" value={props[P.coolSetpoint] as number} unit="°C"
              onCommit={(v) => onSet(P.coolSetpoint, Number(v))} testId="cool-setpoint" />
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              <Btn wide onClick={() => onSet(P.tempControl, 'Cool Down')}>Cool</Btn>
              <Btn wide onClick={() => onSet(P.tempControl, 'Warm Up')}>Warm</Btn>
              <Btn wide onClick={() => onSet(P.tempControl, 'Off')}>Off</Btn>
            </div>
          </div>
        )}
      </div>

      <div style={S.topGroup}>
        <span style={{ color: C.textMuted, fontSize: 11 }}>CAMERA</span>
        <span data-testid="position-value" style={{ fontSize: 12.5 }}>
          {hasPosition ? String(position) : '—'}
        </span>
        <Btn disabled={!hasPosition} testId="extend-btn"
          title={hasPosition ? 'Insert the camera'
                             : 'This server does not report camera position'}
          onClick={() => onSet(P.positionCtl, 'Extend')}>Extend</Btn>
        <Btn disabled={!hasPosition} testId="retract-btn" tone="danger"
          onClick={() => onSet(P.positionCtl, 'Retract')}>Retract</Btn>
      </div>
    </header>
  )
}

// ── Mode rail ─────────────────────────────────────────────────────────────────

function ModeRail({ mode, onMode, summary }: {
  mode: ModeKey; onMode: (m: ModeKey) => void; summary: StatusSummary | null
}) {
  return (
    <nav style={S.rail} data-testid="mode-rail">
      {MODES.map((m) => {
        const on = m.key === mode
        return (
          <button key={m.key} onClick={() => onMode(m.key)}
            data-testid={`mode-${m.key}`} data-active={on}
            style={{
              ...S.railBtn,
              background: on ? C.accentSunken : 'transparent',
              color: on ? C.accent : C.textMuted,
              borderLeftColor: on ? C.accent : 'transparent',
            }}>
            <span style={{ fontSize: 17, lineHeight: 1 }}>{m.glyph}</span>
            <span style={{ fontSize: 10.5, letterSpacing: '.02em' }}>{m.label}</span>
            {/* The board's verdict rides on the rail, so a fault is visible
                from any mode without going to look for it. */}
            {m.key === 'status' && summary && summary.overall !== 'ok' && (
              <span data-testid="rail-status-dot" style={{
                position: 'absolute', top: 7, right: 12, width: 7, height: 7,
                borderRadius: 999, background: stateOf(summary.overall).fg,
              }} />
            )}
          </button>
        )
      })}
    </nav>
  )
}

// ── Instrument sidebar ────────────────────────────────────────────────────────

/**
 * Camera state and control, as editable fields rather than cards.
 *
 * A value the server does not report shows "—" and is not editable, instead of
 * showing a plausible default that would be written back on the first commit.
 */
function InstrumentPanel({ props, connection, onSet, onRefresh }: {
  props: Record<string, unknown>
  connection: Connection | null
  onSet: (name: string, value: unknown) => void
  onRefresh: () => void
}) {
  const has = (k: string) => props[k] != null
  return (
    <aside style={S.sidebar} data-testid="control-panel">
      <Section title="Acquisition" right={
        <button style={S.linkBtn} onClick={onRefresh} data-testid="props-refresh">
          refresh
        </button>
      }>
        <Field label="Exposure" value={props[P.exposure] as number} unit="s"
          testId="field-exposure"
          onCommit={has(P.exposure) ? (v) => onSet(P.exposure, Number(v)) : undefined} />
        <Field label="Frames / s" value={props[P.fps] as number} unit="fps"
          testId="field-fps"
          onCommit={has(P.fps) ? (v) => onSet(P.fps, Number(v)) : undefined} />
        <Field label="Frame count" value={props[P.frameCount] as number}
          onCommit={has(P.frameCount) ? (v) => onSet(P.frameCount, Number(v)) : undefined} />
      </Section>

      <Section title="Sensor">
        <Field label="Size" value={
          connection?.width ? `${connection.width}×${connection.height}` : null} />
        <Field label="Binning" value={props[P.binX] as number}
          onCommit={has(P.binX) ? (v) => onSet(P.binX, Number(v)) : undefined} />
        <Field label="Hardware bin" value={props[P.hwBinX] as number}
          onCommit={has(P.hwBinX) ? (v) => onSet(P.hwBinX, Number(v)) : undefined} />
      </Section>

      <Section title="Output">
        <Field label="Autosave" value={props[P.autosave] as string}
          hint={String(props[P.autosave] ?? '')} />
      </Section>

      <Section title="Server">
        <Field label="Version" value={connection?.server} />
        <Field label="Camera" value={connection?.camera} />
      </Section>
    </aside>
  )
}

// ── Imaging ───────────────────────────────────────────────────────────────────

function ImagingMode({ figure, bridge, running, onStart, onStop, onSingle }: {
  figure: FigureMessage | null; bridge: ReturnType<typeof useFigureBridge>
  running: boolean; onStart: () => void; onStop: () => void; onSingle: () => void
}) {
  const onResize = useCallback((w: number, h: number) => {
    if (figure) window.groundcrew?.resizeFigure(figure.fig_id, w, h)
  }, [figure?.fig_id])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={S.paneBar}>
        <Btn onClick={onStart} disabled={running} tone="go" testId="start-btn"
          active={running}>▶ Live</Btn>
        <Btn onClick={onStop} disabled={!running} testId="stop-btn">■ Stop</Btn>
        <Btn onClick={onSingle} disabled={running} testId="single-btn"
          title={running ? 'Stop the live view first' : 'Take one exposure'}>
          ◉ Single
        </Btn>
        <span style={{ flex: 1 }} />
        {running && <Pill state="ok">LIVE</Pill>}
      </div>

      {!figure ? (
        <div style={{ ...S.viewer, ...S.placeholder }} data-testid="viewer-placeholder">
          Waiting for the camera…
        </div>
      ) : (
        <div style={S.viewer}>
          {/* srcdoc (`html`), not a URL: this backend inlines the figure's ESM,
              so nothing has to be served off disk. FigureFrame owns
              registration, replay-on-load and the resize reporting. */}
          <FigureFrame bridge={bridge} figId={figure.fig_id} html={figure.html}
            title={figure.title ?? 'Live view'} onResize={onResize}
            style={S.iframe} data-testid="viewer-frame" />
        </div>
      )}
    </div>
  )
}

// ── Plot control ──────────────────────────────────────────────────────────────

/**
 * The right-hand strip: histogram, colormap, display range.
 *
 * The histogram comes from the server on the same `get_result` as the pixels,
 * so it describes the WHOLE frame — it does not shift as you pan a zoomed
 * image, which is the failure of computing it from what is on screen.
 */
function PlotControl({ stats, colormap, onColormap }: {
  stats: FrameStats | null; colormap: string; onColormap: (n: string) => void
}) {
  return (
    <aside style={S.plotControl} data-testid="plot-control">
      <Section title="Display">
        <Select label="Colormap" value={colormap} options={COLORMAPS}
          onChange={onColormap} testId="colormap-select" />
        <Field label="Range low" value={stats?.levels?.[0] ?? null} />
        <Field label="Range high" value={stats?.levels?.[1] ?? null} />
      </Section>

      <Section title="Histogram">
        <Histogram stats={stats} />
      </Section>

      <Section title="Frame">
        <Field label="Min" value={stats?.min} />
        <Field label="Max" value={stats?.max} />
        <Field label="Mean" value={stats?.mean == null ? null : stats.mean.toFixed(2)} />
        <Field label="Std" value={stats?.std == null ? null : stats.std.toFixed(2)} />
      </Section>
    </aside>
  )
}

/**
 * A minimal range strip.
 *
 * NOT the binned histogram: the backend has the counts, but sending 256 bins
 * on every frame of a live view is bandwidth spent on a decoration. This shows
 * where the display range sits inside the data range, which is the question
 * the control actually answers — "am I clipping?". The full histogram belongs
 * here once there is an interaction that needs it.
 */
function Histogram({ stats }: { stats: FrameStats | null }) {
  if (!stats || stats.min == null || stats.max == null) {
    return <div style={{ fontSize: 11.5, color: C.textMuted }}>No frame yet</div>
  }
  const span = Math.max(stats.max - stats.min, 1e-9)
  const lo = stats.levels ? (stats.levels[0] - stats.min) / span : 0
  const hi = stats.levels ? (stats.levels[1] - stats.min) / span : 1
  return (
    <div>
      <div style={{
        position: 'relative', height: 34, background: C.bgSunken,
        border: `1px solid ${C.border}`, borderRadius: 5, overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `${Math.max(0, lo) * 100}%`,
          width: `${Math.min(1, hi - lo) * 100}%`,
          background: `linear-gradient(90deg, ${C.accent}22, ${C.accent}55)`,
          borderLeft: `1px solid ${C.accent}`, borderRight: `1px solid ${C.accent}`,
        }} />
      </div>
      <div style={{
        display: 'flex', justifyContent: 'space-between', marginTop: 4,
        fontSize: 10.5, color: C.textMuted, fontFamily: FONT_MONO,
      }}>
        <span>{stats.min}</span>
        <span>display range</span>
        <span>{stats.max}</span>
      </div>
    </div>
  )
}

// ── Stats strip ───────────────────────────────────────────────────────────────

/** `null` renders as "—": the server reports a value it did not measure with an
 *  out-of-band sentinel, and a placeholder is honest where 0 would be a claim. */
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
          recompute them from the tile. */}
      <Stat label="e⁻/pix" value={num(stats.eppix, 2)} />
      <Stat label="e⁻/pix/s" value={num(stats.eppixps, 1)} />
      <Stat label="Over" value={num(stats.over, 2)} />
      <Stat label="Under" value={num(stats.under, 2)} />
    </div>
  )
}

function Stat({ label, value, testId }: { label: string; value: string; testId?: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{
        fontSize: 9.5, letterSpacing: '.08em', textTransform: 'uppercase',
        color: C.textMuted,
      }}>{label}</span>
      {/* The testid goes on the VALUE, not the row: the label is uppercased by
          CSS, and innerText reflects that, so a test matching the row would be
          matching text that does not exist in the source. */}
      <span data-testid={testId} style={{
        fontSize: 13, fontFamily: FONT_MONO, fontVariantNumeric: 'tabular-nums',
      }}>{value}</span>
    </div>
  )
}

// ── Chrome ────────────────────────────────────────────────────────────────────

function StatusBar(props: {
  status: string; error: string | null; running: boolean
  onDismissError: () => void; logOpen: boolean; onToggleLog: () => void
}) {
  return (
    <footer style={S.statusbar}>
      <span style={{
        width: 7, height: 7, borderRadius: 999,
        background: props.running ? '#41d18a' : C.textMuted,
      }} />
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
            <span style={{ color: LEVEL_COLORS[l.level] ?? C.textDim }}>{l.msg}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: C.textMuted, INFO: C.textDim, WARNING: '#e2b04a',
  ERROR: '#ef6b6b', CRITICAL: '#ef6b6b',
}

// ── Styles ────────────────────────────────────────────────────────────────────

const S: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex', flexDirection: 'column', height: '100vh',
    background: C.bg, color: C.text,
    font: '13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif',
    overflow: 'hidden',
  },
  topbar: {
    display: 'flex', alignItems: 'center', gap: 12, padding: '0 14px',
    height: 46, flex: '0 0 46px', background: C.panel,
    borderBottom: `1px solid ${C.border}`,
  },
  brand: { fontWeight: 650, color: C.accent, fontSize: 13.5 },
  subtle: { color: C.textMuted },
  simTag: {
    marginLeft: 8, padding: '1px 6px', borderRadius: 4,
    background: '#e2b04a1a', color: '#e2b04a',
    fontSize: 9.5, fontWeight: 700, letterSpacing: '.06em',
  },
  topBtn: {
    display: 'flex', alignItems: 'center', gap: 8, padding: '5px 11px',
    background: C.panelRaised, border: `1px solid ${C.border}`,
    borderRadius: 6, color: C.text, fontSize: 12.5, cursor: 'pointer',
  },
  topGroup: {
    display: 'flex', alignItems: 'center', gap: 8, padding: '4px 10px',
    background: C.panelRaised, border: `1px solid ${C.border}`, borderRadius: 6,
  },
  popover: {
    position: 'absolute', top: 38, right: 0, zIndex: 20, width: 210,
    padding: 12, background: C.panelRaised,
    border: `1px solid ${C.borderStrong}`, borderRadius: 8,
    boxShadow: '0 10px 28px #0008',
  },

  body: { display: 'flex', flex: 1, minHeight: 0 },
  rail: {
    display: 'flex', flexDirection: 'column', width: 74, flex: '0 0 74px',
    background: C.bgSunken, borderRight: `1px solid ${C.border}`, paddingTop: 6,
  },
  railBtn: {
    position: 'relative', display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center', gap: 5,
    height: 62, border: 'none', borderLeft: '2px solid transparent',
    cursor: 'pointer', font: 'inherit',
  },
  sidebar: {
    width: 236, flex: '0 0 236px', padding: '16px 14px', overflowY: 'auto',
    background: C.panel, borderRight: `1px solid ${C.border}`,
  },
  center: { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 },
  pane: { flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' },
  paneBar: {
    display: 'flex', alignItems: 'center', gap: 8, padding: '9px 14px',
    borderBottom: `1px solid ${C.border}`,
  },
  plotControl: {
    width: 236, flex: '0 0 236px', padding: '16px 14px', overflowY: 'auto',
    background: C.panel, borderLeft: `1px solid ${C.border}`,
  },

  viewer: {
    flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
    minHeight: 0, padding: 10,
  },
  iframe: { width: '100%', height: '100%', border: 'none', background: 'transparent' },
  placeholder: { color: C.textMuted, fontSize: 13 },

  stats: {
    display: 'flex', gap: 22, padding: '9px 16px', flex: '0 0 auto',
    background: C.panel, borderTop: `1px solid ${C.border}`,
  },

  statusbar: {
    display: 'flex', alignItems: 'center', gap: 9, padding: '0 14px',
    height: 27, flex: '0 0 27px', background: C.panel,
    borderTop: `1px solid ${C.border}`, fontSize: 11.5, color: C.textDim,
  },
  linkBtn: {
    background: 'none', border: 'none', color: C.accent, cursor: 'pointer',
    fontSize: 11.5, padding: 0, font: 'inherit',
  },

  logPanel: {
    height: 190, flex: '0 0 190px', display: 'flex', flexDirection: 'column',
    background: C.bgSunken, borderTop: `1px solid ${C.border}`,
  },
  logHead: {
    display: 'flex', alignItems: 'center', gap: 9, padding: '5px 12px',
    borderBottom: `1px solid ${C.border}`, fontSize: 11.5,
  },
  logBody: { flex: 1, overflowY: 'auto', padding: '5px 12px' },
  logLine: {
    display: 'flex', gap: 9, fontSize: 11, fontFamily: FONT_MONO, lineHeight: 1.55,
  },
  logArea: { color: C.textMuted, minWidth: 82 },
  selectSm: {
    background: C.panelRaised, color: C.text, border: `1px solid ${C.border}`,
    borderRadius: 4, fontSize: 11, padding: '1px 5px',
  },
}
