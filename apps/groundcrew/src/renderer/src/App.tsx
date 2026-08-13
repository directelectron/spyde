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
 * The rail carries the four things this app does. Both side panels belong to
 * an IMAGE — acquisition settings and the contrast of what is on screen — so
 * they persist across the modes that HAVE one and are dropped in Status, which
 * is a full-width board about the camera. Leaving them up there would crowd
 * the cards and imply the board is a view of the current frame.
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
import {
  Btn, Field, FieldPair, NotBuilt, Pill, Section, SegBtn, Segmented, Select,
} from './ui'
import { Histogram } from './Histogram'
import { StatusMode } from './modes/Status'
import type { StatusCard, StatusSummary } from './modes/Status'
import { CalibrateMode } from './modes/Other'
import { MOTION_INITIAL, MotionMode } from './modes/Motion'
import type { MotionShifts, MotionState } from './modes/Motion'

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
  /** Histogram counts from the server, for the Plot Control curve. */
  bins?: number[] | null
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

/** Which kind of acquisition is running, if any. A single exposure can be a
 *  60-second integration, so it needs a way out just as much as a live view. */
type AcqMode = 'live' | 'single' | null

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
  waterT: 'Temperature - Chilled Water (Celsius)',
} as const

// ── State ─────────────────────────────────────────────────────────────────────

interface AppState extends ShellState {
  figure: FigureMessage | null
  stats: FrameStats | null
  running: boolean
  acqMode: AcqMode
  colormap: string
  connection: Connection | null
  props: Record<string, unknown>
  statusCards: StatusCard[]
  statusSummary: StatusSummary | null
  /** Motion opens TWO figures (image and FFT), so the single-figure slot is
   *  not enough — figures are keyed by the window id the backend minted. */
  figures: Map<number, FigureMessage>
  motion: MotionState
  motionShifts: MotionShifts | null
  /** The last error, shown in the status bar until dismissed. App-level rather
   *  than shell-level: the shell's chrome slice has no dismissable error. */
  error: string | null
}

type AppAction =
  | ShellAction
  | { type: 'FIGURE'; figure: FigureMessage }
  | { type: 'FRAME_STATS'; stats: FrameStats }
  | { type: 'ACQ_STATE'; running: boolean; mode: AcqMode }
  | { type: 'COLORMAP'; name: string }
  | { type: 'CONNECTION'; connection: Connection }
  | { type: 'PROPERTIES'; values: Record<string, unknown> }
  | { type: 'STATUS_REPORT'; cards: StatusCard[]; summary: StatusSummary }
  | { type: 'MOTION_STATE'; state: MotionState }
  | { type: 'MOTION_SHIFTS'; shifts: MotionShifts }
  | { type: 'APP_ERROR'; text: string | null }

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'FIGURE': {
      const figures = new Map(state.figures)
      figures.set(Number(action.figure.window_id), action.figure)
      // `figure` stays the FIRST one: Imaging opened it and its pane still
      // addresses it directly.
      return { ...state, figures, figure: state.figure ?? action.figure }
    }
    case 'MOTION_STATE': return { ...state, motion: action.state }
    case 'MOTION_SHIFTS': return { ...state, motionShifts: action.shifts }
    case 'FRAME_STATS': return { ...state, stats: action.stats }
    case 'ACQ_STATE':
      return { ...state, running: action.running, acqMode: action.mode }
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
  figure: null, stats: null, running: false, acqMode: null, colormap: 'gray',
  connection: null, props: {}, statusCards: [], statusSummary: null, error: null,
  figures: new Map(), motion: MOTION_INITIAL, motionShifts: null,
}

// ── App ───────────────────────────────────────────────────────────────────────

export function App() {
  const [state, dispatch] = useReducer(appReducer, initialState)
  const [mode, setMode] = useState<ModeKey>('imaging')
  const [logOpen, setLogOpen] = useState(false)
  const bridge = useFigureBridge()
  const { figure, stats, running, acqMode, colormap, error, status, logEntries,
          connection, props, statusCards, statusSummary,
          figures, motion, motionShifts } = state

  // The two side panels belong to an IMAGE. Status is a board about the
  // camera, so they are not merely empty there — they would be misleading.
  const showPanels = mode !== 'status'

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
          dispatch({
            type: 'ACQ_STATE', running: Boolean(msg.running),
            mode: (msg.mode ?? null) as AcqMode,
          }); break
        case 'connection':
          dispatch({ type: 'CONNECTION', connection: msg as unknown as Connection }); break
        case 'properties':
          dispatch({
            type: 'PROPERTIES',
            values: (msg.values ?? {}) as Record<string, unknown>,
          }); break
        case 'motion_state':
          dispatch({ type: 'MOTION_STATE',
            state: msg as unknown as MotionState }); break
        case 'motion_shifts':
          dispatch({ type: 'MOTION_SHIFTS',
            shifts: msg as unknown as MotionShifts }); break
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
    if (mode === 'motion') act('motion_state')
  }, [mode, connection?.connected, act])

  return (
    <div style={S.root}>
      <TopBar connection={connection} props={props} onSet={setProp} />

      <div style={S.body}>
        <ModeRail mode={mode} onMode={setMode} summary={statusSummary} />
        {/* Status is a full-width board about the CAMERA, not about an image:
            the acquisition sidebar and the plot control have nothing to act on
            there, and leaving them up both crowds the cards and implies the
            board is a view of the current frame. */}
        {showPanels && (
          <InstrumentPanel props={props} connection={connection} onSet={setProp}
            onRefresh={() => act('refresh_properties')} />
        )}

        <main style={S.center}>
          <div style={S.pane}>
            {mode === 'imaging' && (
              <ImagingMode figure={figure} bridge={bridge} running={running}
                acqMode={acqMode}
                onStart={() => act('start_acquisition')}
                onStop={() => act('stop_acquisition')}
                onSingle={() => act('single_acquisition')} />
            )}
            {mode === 'motion' && (
              <MotionMode state={motion} shifts={motionShifts} figures={figures}
                bridge={bridge} act={act} />
            )}
            {mode === 'calibrate' && <CalibrateMode />}
            {mode === 'status' && (
              <StatusMode cards={statusCards} summary={statusSummary}
                onRefresh={() => act('refresh_status')} />
            )}
          </div>
          {mode === 'imaging' && <StatsStrip stats={stats} />}
        </main>

        {showPanels && (
          <PlotControl stats={stats} colormap={colormap}
            onColormap={(name) => {
              dispatch({ type: 'COLORMAP', name }); act('set_colormap', { name })
            }}
            onClim={(low, high) => act('set_clim', { low, high })}
            onAuto={() => act('auto_clim')}
            onReset={() => {
              // Reset means the FULL data range, tail included — the same
              // contract as SpyDE's. Auto is the robust 2–98%.
              if (stats?.min != null && stats?.max != null) {
                act('set_clim', { low: stats.min, high: stats.max })
              }
            }} />
        )}
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
  const extended = /extend|insert/i.test(String(position ?? ''))

  return (
    <header style={S.topbar}>
      <span style={S.brand}>● Ground Crew</span>
      <span style={{ ...S.subtle, fontSize: 12 }}>
        {connection?.camera ?? 'Direct Electron'}
        {connection?.fake && <span style={S.simTag}>SIMULATED</span>}
      </span>

      <span style={{ flex: 1 }} />

      {/* Valve position and detector temperature are the two facts that make
          an acquisition worthless if wrong, so they live here rather than in a
          column that scrolls. Both DISABLE rather than disappear when the
          server does not report them: a control that vanishes reads as a
          missing feature, a greyed one reads as an unavailable reading. */}
      <Segmented>
        <SegBtn on={extended} disabled={!hasPosition} testId="extend-btn"
          title={hasPosition ? 'Insert the camera'
                             : 'This server does not report camera position'}
          onClick={() => onSet(P.positionCtl, 'Extend')}>
          {extended ? 'Extended' : 'Extend'}
        </SegBtn>
        <SegBtn disabled={!hasPosition} testId="retract-btn"
          onClick={() => onSet(P.positionCtl, 'Retract')}>Retract</SegBtn>
      </Segmented>

      <Segmented>
        <SegBtn tone="cryo" disabled={!hasTemp} testId="cool-btn"
          onClick={() => onSet(P.tempControl, 'Cool Down')}>Cool</SegBtn>
        <SegBtn disabled={!hasTemp} testId="warm-btn"
          onClick={() => onSet(P.tempControl, 'Warm Up')}>Warm</SegBtn>
        <SegBtn disabled={!hasTemp} testId="tempoff-btn"
          onClick={() => onSet(P.tempControl, 'Off')}>Off</SegBtn>
      </Segmented>

      <div style={{ position: 'relative' }}>
        {/* The temperature is a BUTTON: clicking it sets a target. "Cool"
            alone never says cool to what. */}
        <button
          style={{
            ...S.tempBtn,
            color: hasTemp ? C.cryo : C.textMuted,
            borderColor: openTemp ? C.cryo : C.ctlLine,
            opacity: hasTemp ? 1 : 0.5,
            cursor: hasTemp ? 'pointer' : 'not-allowed',
          }}
          disabled={!hasTemp}
          onClick={() => setOpenTemp((v) => !v)}
          data-testid="temp-btn"
          title={hasTemp ? 'Set the cooling setpoint'
                         : 'This server does not report detector temperature'}
        >
          <span data-testid="temp-value">
            {hasTemp ? `${Number(temp).toFixed(1)} °C` : '—'}
          </span>
          <span style={{ color: C.textMuted }}>⌄</span>
        </button>
        {openTemp && hasTemp && (
          <div style={S.popover}>
            <div style={S.popTitle}>Detector temperature</div>
            <KV label="Current" value={`${Number(temp).toFixed(1)} °C`} tone={C.cryo} />
            <KV label="Position" value={hasPosition ? String(position) : '—'} />
            <div style={{ marginTop: 9 }}>
              <Field label="Setpoint" value={props[P.coolSetpoint] as number} unit="°C"
                onCommit={(v) => onSet(P.coolSetpoint, Number(v))}
                testId="cool-setpoint" />
            </div>
          </div>
        )}
      </div>

      {/* Link state only. Camera position is carried by the segmented control
          above, whose label IS the position — putting both in one chip made a
          server that cannot report position read as "Ready", which is a claim
          about the wrong thing. */}
      <span style={S.readyChip}>
        <span style={{
          width: 7, height: 7, borderRadius: 999,
          background: connection?.connected ? '#41d18a' : C.textMuted,
        }} />
        <span data-testid="link-state">
          {connection?.connected ? 'Ready' : 'Offline'}
        </span>
      </span>
    </header>
  )
}

function KV({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', gap: 8, padding: '2.5px 0',
    }}>
      <span style={{ fontSize: 11.5, color: C.textMuted }}>{label}</span>
      <span style={{
        font: `11.5px ${FONT_MONO}`, fontVariantNumeric: 'tabular-nums',
        color: tone ?? C.textDim,
      }}>{value}</span>
    </div>
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

function ImagingMode({ figure, bridge, running, acqMode, onStart, onStop, onSingle }: {
  figure: FigureMessage | null; bridge: ReturnType<typeof useFigureBridge>
  running: boolean; acqMode: AcqMode
  onStart: () => void; onStop: () => void; onSingle: () => void
}) {
  const onResize = useCallback((w: number, h: number) => {
    if (figure) window.groundcrew?.resizeFigure(figure.fig_id, w, h)
  }, [figure?.fig_id])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={S.paneBar}>
        <Btn onClick={onStart} disabled={running} tone="go" testId="start-btn"
          active={acqMode === 'live'}>▶ Live</Btn>
        {/* Stop ends EITHER. A single exposure can be a long integration, so
            it needs a way out just as much as a live view — and to the camera
            it is the same command. */}
        <Btn onClick={onStop} disabled={!running} testId="stop-btn"
          title={acqMode === 'single' ? 'Stop this exposure'
               : acqMode === 'live' ? 'Stop the live view'
               : 'Nothing is running'}>■ Stop</Btn>
        <Btn onClick={onSingle} disabled={running} testId="single-btn"
          active={acqMode === 'single'}
          title={running ? 'An acquisition is already running' : 'Take one exposure'}>
          ◉ Single
        </Btn>
        <span style={{ flex: 1 }} />
        {acqMode === 'live' && <Pill state="ok">LIVE</Pill>}
        {acqMode === 'single' && <Pill state="warn">EXPOSING</Pill>}
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
function PlotControl({ stats, colormap, onColormap, onClim, onAuto, onReset }: {
  stats: FrameStats | null; colormap: string; onColormap: (n: string) => void
  onClim: (lo: number, hi: number) => void; onAuto: () => void; onReset: () => void
}) {
  // While dragging, the handles must follow the POINTER, not the last frame.
  // Statistics only arrive on a refresh — and a stopped camera never refreshes
  // — so waiting for the backend to echo the new range would leave the handle
  // pinned under the cursor doing nothing. The override holds until Auto or
  // Reset hands the range back.
  const [override, setOverride] = React.useState<[number, number] | null>(null)
  const ready = stats && stats.bins?.length && stats.min != null && stats.max != null
  const levels = override ?? stats?.levels ?? null

  return (
    <aside style={S.plotControl} data-testid="plot-control">
      <Section title="Histogram">
        {ready
          ? <Histogram
              counts={stats.bins!} lo={stats.min!} hi={stats.max!}
              vmin={levels?.[0] ?? stats.min!}
              vmax={levels?.[1] ?? stats.max!}
              onClim={(lo, hi) => { setOverride([lo, hi]); onClim(lo, hi) }}
              onAuto={() => { setOverride(null); onAuto() }}
              onReset={() => { setOverride(null); onReset() }} />
          : <div data-testid="histogram-empty"
              style={{ fontSize: 11.5, color: C.textMuted }}>—</div>}
      </Section>

      <Section title="Colormap">
        <Select label="" value={colormap} options={COLORMAPS}
          onChange={onColormap} testId="colormap-select" wide />
      </Section>

      <Section title="Frame">
        <KV label="Min" value={stats?.min == null ? '—' : String(stats.min)} />
        <KV label="Max" value={stats?.max == null ? '—' : String(stats.max)} />
        <KV label="Mean" value={stats?.mean == null ? '—' : stats.mean.toFixed(2)} />
        <KV label="Std" value={stats?.std == null ? '—' : stats.std.toFixed(2)} />
      </Section>
    </aside>
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
  tempBtn: {
    display: 'flex', alignItems: 'center', gap: 6, padding: '4px 9px',
    background: C.ctl, border: `1px solid ${C.ctlLine}`, borderRadius: 6,
    font: `12px ${FONT_MONO}`, fontVariantNumeric: 'tabular-nums',
  },
  readyChip: {
    display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px',
    background: C.ctl, border: `1px solid ${C.ctlLine}`, borderRadius: 999,
    fontSize: 11.5, color: C.textDim,
  },
  popTitle: {
    font: `600 9.5px/1 ${FONT_MONO}`, letterSpacing: '.11em',
    textTransform: 'uppercase', color: C.textMuted, marginBottom: 8,
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
