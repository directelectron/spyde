/**
 * App.tsx — Autopilot's fixed-pane layout.
 *
 * The third arrangement over the same shell: SpyDE has an MDI workspace, Ground
 * Crew a manual control panel, and this a QUEUE. The sidebar is the recipe, one
 * row per step, lit as the runner reaches it; the centre shows the last frame
 * acquired; the footer carries transport controls and the run's progress.
 *
 * Everything backend-facing is shared — @de/shell-preload for the bridge,
 * @de/shell-renderer for the figure bridge and the chrome reducer. What is
 * app-specific is this file.
 */
import React, { useCallback, useEffect, useReducer, useState } from 'react'
import {
  FigureFrame, useFigureBridge, shellReducer, shellInitialState, toShellAction,
} from '@de/shell-renderer'
import type { FigureMessage, ShellState, ShellAction, LogEntry } from '@de/shell-renderer'

// ── Backend message shapes ────────────────────────────────────────────────────

type StepKind = 'acquire' | 'move' | 'settle'
type RunState = 'idle' | 'running' | 'paused' | 'done' | 'stopped' | 'failed'
type StepState = 'pending' | 'running' | 'done'

interface Step {
  kind: StepKind
  label: string
  exposure_s: number
  x: number
  y: number
  seconds: number
}
interface FrameStats {
  acquired: number
  stage: { x: number; y: number }
  min: number; max: number; mean: number; shape: number[]
}

// ── State ─────────────────────────────────────────────────────────────────────

interface AppState extends ShellState {
  figure: FigureMessage | null
  recipeName: string
  steps: Step[]
  stepStates: StepState[]
  runState: RunState
  stats: FrameStats | null
  progress: { done: number; total: number } | null
  error: string | null
}

type AppAction =
  | ShellAction
  | { type: 'FIGURE'; figure: FigureMessage }
  | { type: 'RECIPE'; name: string; steps: Step[] }
  | { type: 'STEP_STATE'; index: number; state: StepState }
  | { type: 'RUN_STATE'; state: RunState }
  | { type: 'FRAME_STATS'; stats: FrameStats }
  | { type: 'PROGRESS'; done: number; total: number }
  | { type: 'ERROR'; text: string | null }

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'FIGURE':
      return { ...state, figure: action.figure }

    case 'RECIPE':
      return {
        ...state,
        recipeName: action.name,
        steps: action.steps,
        stepStates: action.steps.map(() => 'pending' as StepState),
      }

    case 'STEP_STATE': {
      const stepStates = state.stepStates.slice()
      stepStates[action.index] = action.state
      return { ...state, stepStates }
    }

    case 'RUN_STATE':
      return {
        ...state,
        runState: action.state,
        // A fresh run must clear the previous one's marks, or every row shows
        // "done" from the moment it starts.
        stepStates: action.state === 'running' && state.runState !== 'paused'
          ? state.steps.map(() => 'pending' as StepState)
          : state.stepStates,
      }

    case 'FRAME_STATS':
      return { ...state, stats: action.stats }

    case 'PROGRESS':
      return {
        ...state,
        progress: action.total > 0 ? { done: action.done, total: action.total } : null,
      }

    case 'ERROR':
      return { ...state, error: action.text }

    default:
      return shellReducer(state, action as ShellAction)
  }
}

const initialState: AppState = {
  ...shellInitialState,
  figure: null,
  recipeName: '',
  steps: [],
  stepStates: [],
  runState: 'idle',
  stats: null,
  progress: null,
  error: null,
}

// ── App ───────────────────────────────────────────────────────────────────────

export function App() {
  const [state, dispatch] = useReducer(appReducer, initialState)
  const [logOpen, setLogOpen] = useState(false)
  const bridge = useFigureBridge()
  const {
    figure, recipeName, steps, stepStates, runState, stats, progress, error,
    status, logEntries,
  } = state

  useEffect(() => {
    const dispose = window.autopilot?.onMessage((raw) => {
      const msg = raw as Record<string, unknown>

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
        case 'recipe':
          dispatch({
            type: 'RECIPE',
            name: String(msg.name ?? ''),
            steps: (msg.steps ?? []) as Step[],
          })
          break
        case 'step_state':
          dispatch({
            type: 'STEP_STATE',
            index: Number(msg.index),
            state: String(msg.state) as StepState,
          })
          break
        case 'run_state':
          dispatch({ type: 'RUN_STATE', state: String(msg.state) as RunState })
          break
        case 'frame_stats':
          dispatch({ type: 'FRAME_STATS', stats: msg as unknown as FrameStats })
          break
        case 'progress':
          dispatch({ type: 'PROGRESS', done: Number(msg.done), total: Number(msg.total) })
          break
        case 'error':
          dispatch({ type: 'ERROR', text: String(msg.text ?? '') })
          break
        case 'log':
          dispatch({ type: 'LOG', entries: [msg as unknown as LogEntry] })
          break
        default:
          break
      }
    })
    return () => dispose?.()
  }, [bridge])

  const act = useCallback((action: string, payload: Record<string, unknown> = {}) => {
    window.autopilot?.action(action, payload)
  }, [])

  const onResize = useCallback((w: number, h: number) => {
    if (figure) window.autopilot?.resizeFigure(figure.fig_id, w, h)
  }, [figure?.fig_id])

  const busy = runState === 'running' || runState === 'paused'

  return (
    <div style={S.root}>
      <header style={S.titlebar}>
        <span style={S.brand}>◆ Autopilot</span>
        <span style={S.subtle}>Direct Electron</span>
      </header>

      <div style={S.body}>
        <aside style={S.sidebar} data-testid="recipe-panel">
          <div style={S.sectionTitle}>Recipe</div>
          <div style={S.recipeName}>{recipeName || '—'}</div>
          <ol style={S.steps} data-testid="step-list">
            {steps.map((step, i) => (
              <li key={i} style={{ ...S.step, ...STEP_STYLE[stepStates[i] ?? 'pending'] }}
                  data-testid={`step-${i}`} data-state={stepStates[i] ?? 'pending'}>
                <span style={S.stepKind}>{KIND_GLYPH[step.kind]}</span>
                <span>{step.label}</span>
              </li>
            ))}
          </ol>
        </aside>

        <main style={S.center}>
          <div style={S.viewer}>
            {figure ? (
              <FigureFrame
                bridge={bridge}
                figId={figure.fig_id}
                html={figure.html}
                title={figure.title ?? 'Last acquisition'}
                onResize={onResize}
                style={S.iframe}
                data-testid="viewer-frame"
              />
            ) : (
              <div style={S.placeholder} data-testid="viewer-placeholder">
                Run the recipe to acquire
              </div>
            )}
          </div>
          <StatsStrip stats={stats} />
        </main>
      </div>

      {logOpen && <LogPanel logs={logEntries} onClose={() => setLogOpen(false)} />}

      <footer style={S.footer}>
        <button style={S.btn} onClick={() => act('run_recipe')}
                disabled={runState === 'running'} data-testid="run-btn">
          ▶ {runState === 'paused' ? 'Resume' : 'Run'}
        </button>
        <button style={S.btn} onClick={() => act('pause_recipe')}
                disabled={runState !== 'running'} data-testid="pause-btn">
          ‖ Pause
        </button>
        <button style={S.btn} onClick={() => act('stop_recipe')}
                disabled={!busy} data-testid="stop-btn">
          ■ Stop
        </button>

        <ProgressBar progress={progress} />

        <span style={S.state} data-testid="run-state">{error ?? status}</span>
        <button style={S.linkBtn} onClick={() => setLogOpen((v) => !v)}>
          {logOpen ? 'Hide log' : 'Log'}
        </button>
      </footer>
    </div>
  )
}

function ProgressBar({ progress }: { progress: { done: number; total: number } | null }) {
  // Reserve the space even when idle, so the transport controls don't shift
  // sideways the moment a run starts.
  const pct = progress ? Math.round((100 * progress.done) / progress.total) : 0
  return (
    <div style={S.progressOuter} data-testid="progress">
      {progress && <div style={{ ...S.progressInner, width: `${pct}%` }} />}
      <span style={S.progressLabel}>
        {progress ? `${progress.done} / ${progress.total}` : ''}
      </span>
    </div>
  )
}

function StatsStrip({ stats }: { stats: FrameStats | null }) {
  if (!stats) return <div style={S.stats} data-testid="stats-strip" />
  return (
    <div style={S.stats} data-testid="stats-strip">
      <Stat label="Acquired" value={String(stats.acquired)} testId="stat-acquired" />
      <Stat label="Stage" value={`${stats.stage.x.toFixed(1)}, ${stats.stage.y.toFixed(1)}`} />
      <Stat label="Size" value={stats.shape.join(' × ')} />
      <Stat label="Min" value={stats.min.toFixed(0)} />
      <Stat label="Max" value={stats.max.toFixed(0)} />
      <Stat label="Mean" value={stats.mean.toFixed(1)} />
    </div>
  )
}

function Stat({ label, value, testId }: { label: string; value: string; testId?: string }) {
  return (
    <div style={S.stat}>
      <span style={S.statLabel}>{label}</span>
      {/* testid on the VALUE: the label is uppercased by CSS and innerText
          reflects text-transform, so scraping the row and matching the label's
          own casing silently finds nothing. */}
      <span style={S.statValue} data-testid={testId}>{value}</span>
    </div>
  )
}

function LogPanel({ logs, onClose }: { logs: LogEntry[]; onClose: () => void }) {
  return (
    <section style={S.logPanel} data-testid="log-panel">
      <div style={S.logHead}>
        <strong>Log</strong>
        <span style={{ flex: 1 }} />
        <button style={S.linkBtn} onClick={onClose}>close</button>
      </div>
      <div style={S.logBody}>
        {logs.slice(-200).map((l, i) => (
          <div key={l.seq ?? i} style={S.logLine}>
            <span style={S.logArea}>{l.area ?? ''}</span>
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
const KIND_GLYPH: Record<StepKind, string> = {
  move: '⤢', settle: '⏱', acquire: '◉',
}
const STEP_STYLE: Record<StepState, React.CSSProperties> = {
  pending: { opacity: 0.55 },
  running: { background: '#22304a', borderColor: '#8ab4ff', color: '#8ab4ff' },
  done: { color: '#41d18a' },
}

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
  brand: { fontWeight: 600, color: '#c9a6ff' },
  subtle: { color: '#7a8296' },
  body: { display: 'flex', flex: 1, minHeight: 0 },
  sidebar: {
    width: 260, flexShrink: 0, borderRight: '1px solid #262a35',
    background: '#171a22', padding: 14, overflowY: 'auto',
  },
  sectionTitle: {
    color: '#7a8296', fontSize: 10, textTransform: 'uppercase',
    letterSpacing: 0.6, marginBottom: 6,
  },
  recipeName: { fontWeight: 600, marginBottom: 12 },
  steps: { listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 4 },
  step: {
    display: 'flex', gap: 8, alignItems: 'center', padding: '5px 8px',
    border: '1px solid #2f3442', borderRadius: 6, background: '#1e2230',
  },
  stepKind: { width: 14, textAlign: 'center', color: '#7a8296' },
  center: { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, minHeight: 0 },
  viewer: { flex: 1, minHeight: 0, padding: 12, display: 'flex' },
  placeholder: {
    flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: '#7a8296', border: '1px dashed #2f3442', borderRadius: 8,
  },
  iframe: { flex: 1, border: '1px solid #262a35', borderRadius: 8, background: '#0d0f14' },
  stats: {
    display: 'flex', gap: 18, padding: '8px 14px', flexShrink: 0,
    borderTop: '1px solid #262a35', background: '#171a22', minHeight: 34,
  },
  stat: { display: 'flex', flexDirection: 'column' },
  statLabel: { color: '#7a8296', fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.4 },
  statValue: { fontVariantNumeric: 'tabular-nums' },
  footer: {
    display: 'flex', alignItems: 'center', gap: 10, padding: '8px 14px',
    borderTop: '1px solid #262a35', background: '#12141a', flexShrink: 0,
  },
  btn: {
    padding: '6px 12px', borderRadius: 6, cursor: 'pointer',
    border: '1px solid #2f3442', background: '#1e2230', color: '#e6e9f0',
  },
  progressOuter: {
    position: 'relative', flex: 1, height: 18, borderRadius: 9,
    background: '#1e2230', border: '1px solid #2f3442', overflow: 'hidden',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  progressInner: {
    position: 'absolute', left: 0, top: 0, bottom: 0,
    background: '#3a5a8c', transition: 'width 120ms linear',
  },
  progressLabel: { position: 'relative', fontSize: 11, fontVariantNumeric: 'tabular-nums' },
  state: { color: '#c8cee0', minWidth: 160, textAlign: 'right' },
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
  logBody: {
    flex: 1, overflowY: 'auto', padding: '6px 14px',
    fontFamily: 'ui-monospace, monospace',
  },
  logLine: { display: 'flex', gap: 10, fontSize: 12 },
  logArea: { color: '#7a8296', minWidth: 80 },
}
