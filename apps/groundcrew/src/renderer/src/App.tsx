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
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

// ── Backend message shapes ────────────────────────────────────────────────────

interface FigureMessage {
  type: 'figure'
  fig_id: string
  window_id: number
  html: string
  title?: string
  aspect?: number | null
}
interface FrameStats {
  frame: number; shown: number; dropped: number
  min: number; max: number; mean: number; dtype: string; shape: number[]
}
interface LogEntry { level: string; name: string; area: string; msg: string; time: number }

const COLORMAPS = ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
const EXPOSURES = [10, 25, 50, 100, 250, 500]

/**
 * Relays anyplotlib state into the figure iframe, and remembers the latest of
 * each key.
 *
 * The retention is the load-bearing part. A `state_update` that arrives before
 * the iframe has mounted its listener posts into the void — silently, with no
 * error — and since the backend only sends CHANGES, that frame is gone for
 * good. For a live camera the next frame covers it up; for the FIRST frame,
 * which is exactly the one that races the mount, it means an image that never
 * appears. `replay()` re-posts everything once the frame is ready.
 */
function useFigureBridge() {
  const iframes = useRef(new Map<string, HTMLIFrameElement>())
  const latest = useRef(new Map<string, Map<string, unknown>>())
  const latestBinary = useRef(new Map<string, Map<string, Uint8Array>>())

  const post = useCallback((figId: string, message: Record<string, unknown>) => {
    iframes.current.get(figId)?.contentWindow?.postMessage(message, '*')
  }, [])

  const onState = useCallback((figId: string, key: string, value: unknown) => {
    if (!latest.current.has(figId)) latest.current.set(figId, new Map())
    latest.current.get(figId)!.set(key, value)
    post(figId, { type: 'awi_state', key, value })
  }, [post])

  const onBinary = useCallback((figId: string, key: string, bytes: Uint8Array,
                                header: Record<string, unknown>) => {
    if (!latestBinary.current.has(figId)) latestBinary.current.set(figId, new Map())
    // Keyed by the panel's geom, not by `key`: `key` is the pixel FIELD and is
    // identical across panels, so keying by it alone would let each panel
    // overwrite the last and retain exactly one frame per figure.
    const slot = `${key}:${String(header.geom ?? '')}`
    latestBinary.current.get(figId)!.set(slot, bytes)
    post(figId, { type: 'awi_state_binary', key, header, buffer: bytes })
  }, [post])

  const register = useCallback((figId: string, el: HTMLIFrameElement | null) => {
    if (el) iframes.current.set(figId, el)
    else iframes.current.delete(figId)
  }, [])

  const replay = useCallback((figId: string) => {
    for (const [key, value] of latest.current.get(figId) ?? []) {
      post(figId, { type: 'awi_state', key, value })
    }
    for (const [slot, bytes] of latestBinary.current.get(figId) ?? []) {
      const [key] = slot.split(':')
      post(figId, { type: 'awi_state_binary', key, buffer: bytes })
    }
  }, [post])

  return { register, replay, onState, onBinary }
}

export function App() {
  const [figure, setFigure] = useState<FigureMessage | null>(null)
  const [stats, setStats] = useState<FrameStats | null>(null)
  const [status, setStatus] = useState('Starting…')
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(true)
  const [exposureMs, setExposureMs] = useState(50)
  const [colormap, setColormap] = useState('gray')
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [logOpen, setLogOpen] = useState(false)
  const bridge = useFigureBridge()

  useEffect(() => {
    // Returns a disposer — see the preload's note on StrictMode double-invoke.
    const dispose = window.groundcrew?.onMessage((raw) => {
      const msg = raw as Record<string, unknown>
      switch (msg.type) {
        case 'figure':
          setFigure(msg as unknown as FigureMessage)
          break
        case 'state_update':
          bridge.onState(String(msg.fig_id), String(msg.key), msg.value)
          break
        case 'state_update_binary':
          bridge.onBinary(
            String(msg.fig_id), String(msg.key),
            msg.buffer as Uint8Array,
            (msg.header ?? {}) as Record<string, unknown>,
          )
          break
        case 'frame_stats':
          setStats(msg as unknown as FrameStats)
          break
        case 'acq_state':
          setRunning(Boolean(msg.running))
          if (typeof msg.exposure_s === 'number') setExposureMs(msg.exposure_s * 1000)
          break
        case 'status':
          setStatus(String(msg.text ?? ''))
          break
        case 'error':
          setError(String(msg.text ?? ''))
          break
        case 'log':
          // Bounded ring: a free-running camera logs indefinitely, and an
          // unbounded array would grow for the life of the session.
          setLogs((prev) => [...prev, msg as unknown as LogEntry].slice(-500))
          break
        case 'backend_exited':
          setError(`Backend stopped (code ${msg.code ?? '?'})`)
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
          onExposure={(ms) => { setExposureMs(ms); act('set_exposure', { seconds: ms / 1000 }) }}
          onColormap={(name) => { setColormap(name); act('set_colormap', { name }) }}
        />

        <main style={S.center}>
          <FigurePane figure={figure} bridge={bridge} />
          <StatsStrip stats={stats} />
        </main>
      </div>

      {logOpen && <LogPanel logs={logs} onClose={() => setLogOpen(false)} />}

      <StatusBar
        status={status}
        error={error}
        running={running}
        onDismissError={() => setError(null)}
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
  const ref = useRef<HTMLIFrameElement | null>(null)

  // Tell the backend the pane's size so anyplotlib lays the figure out to fit.
  useEffect(() => {
    if (!figure) return
    const el = ref.current
    if (!el) return
    const send = () => {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.height > 0) {
        window.groundcrew?.resizeFigure(figure.fig_id, Math.round(r.width), Math.round(r.height))
      }
    }
    send()
    const ro = new ResizeObserver(send)
    ro.observe(el)
    return () => ro.disconnect()
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
      {/*
        srcdoc, not a URL: the backend inlines the figure's ESM bundle, so the
        whole thing is self-contained and needs nothing served off disk. Keyed by
        fig_id so React reuses the SAME iframe across frames — remounting it per
        frame would tear down the WebGPU context and throw away the user's zoom.
      */}
      <iframe
        ref={(el) => { ref.current = el; bridge.register(figure.fig_id, el) }}
        key={figure.fig_id}
        title={figure.title ?? 'Live view'}
        srcDoc={figure.html}
        style={S.iframe}
        data-testid="viewer-frame"
        // Replay whatever arrived before the frame's listener existed. The
        // FIRST pixel frame reliably loses that race, and the backend only ever
        // sends changes — so without this the image can stay on its placeholder
        // indefinitely with nothing to show for it.
        onLoad={() => bridge.replay(figure.fig_id)}
      />
    </div>
  )
}

function StatsStrip({ stats }: { stats: FrameStats | null }) {
  if (!stats) return <div style={S.stats} data-testid="stats-strip" />
  return (
    <div style={S.stats} data-testid="stats-strip">
      <Stat label="Frame" value={String(stats.frame)} testId="stat-frame" />
      <Stat label="Size" value={stats.shape.join(' × ')} />
      <Stat label="dtype" value={stats.dtype} />
      <Stat label="Min" value={stats.min.toFixed(0)} />
      <Stat label="Max" value={stats.max.toFixed(0)} />
      <Stat label="Mean" value={stats.mean.toFixed(1)} />
      <Stat label="Shown" value={String(stats.shown)} />
      {/* Dropped frames are expected (newest-wins painting), so this is
          information, not an error — but it should be visible. */}
      <Stat label="Dropped" value={String(stats.dropped)} />
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
