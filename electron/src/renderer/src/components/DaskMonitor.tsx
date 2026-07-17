/**
 * DaskMonitor.tsx — compact live compute readout in the StatusBar, with a
 * click-to-open per-worker breakdown.
 *
 * Driven by `spyde:dask_stats` CustomEvents (~every 2 s from the backend
 * sampler while the cluster is up). The bar segment shows the hot numbers —
 * "CPU 87% · GPU 96% · 14 tasks" — coloured amber/red as they saturate, so a
 * heavy find-vectors/OM batch is visible at a glance. Clicking opens a
 * popover with one row per worker (CPU bar, memory, running/queued tasks),
 * the GPU row, and the full Dask-dashboard link for deep dives.
 *
 * The segment hides itself when stats stop flowing (no cluster / backend
 * gone) — a stale readout is worse than none.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { DaskStatsMessage } from '../kernel/protocol'
import { Dropdown } from './Dropdown'

const RAM_OPTS = [
  { value: '0.4', label: '40% of RAM' }, { value: '0.5', label: '50% of RAM' },
  { value: '0.65', label: '65% of RAM' }, { value: '0.8', label: '80% of RAM' },
] as const
const CPU_OPTS = [
  { value: '0.25', label: '25% of cores' }, { value: '0.5', label: '50% of cores' },
  { value: '0.75', label: '75% of cores' }, { value: '1.0', label: '100% of cores' },
] as const
const GPU_OPTS = [
  { value: '1', label: '1 worker' }, { value: '2', label: '2 workers' },
  { value: '4', label: '4 workers' }, { value: '6', label: '6 workers' },
  { value: '8', label: '8 workers' }, { value: 'all', label: 'all workers' },
  { value: 'off', label: 'off (CPU only)' },
] as const

// Snap an effective fraction to the nearest dropdown option value.
const snap = (opts: readonly { value: string }[], v: number | string) => {
  const n = Number(v)
  if (!Number.isFinite(n)) return String(v)
  let best = opts[0].value
  for (const o of opts) {
    if (Math.abs(Number(o.value) - n) < Math.abs(Number(best) - n)) best = o.value
  }
  return best
}

const STALE_MS = 7_000          // hide after ~3 missed samples

const pctColor = (p: number) =>
  p >= 95 ? '#f38ba8' : p >= 80 ? '#f9e2af' : '#a6e3a1'

const fmtGB = (bytes: number) => (bytes / 1024 ** 3).toFixed(1)

export function DaskMonitor() {
  const { state, sendAction } = useSpyDE()
  const [stats, setStats] = React.useState<DaskStatsMessage | null>(null)
  const [open, setOpen] = React.useState(false)
  const lastAt = React.useRef(0)
  // Compute-limit drafts (seeded once from the backend's effective config;
  // Apply → compute_configure → cluster restart).
  const [cfgRam, setCfgRam] = React.useState<string | null>(null)
  const [cfgCpu, setCfgCpu] = React.useState<string | null>(null)
  const [cfgGpu, setCfgGpu] = React.useState<string | null>(null)
  const [applying, setApplying] = React.useState(false)
  const [limitsOpen, setLimitsOpen] = React.useState(false)
  const seeded = React.useRef(false)

  React.useEffect(() => {
    const onStats = (e: Event) => {
      lastAt.current = Date.now()
      const s = (e as CustomEvent).detail as DaskStatsMessage
      setStats(s)
      if (!seeded.current && s.config) {
        seeded.current = true
        setCfgRam(snap(RAM_OPTS, s.config.mem_fraction))
        setCfgCpu(snap(CPU_OPTS, s.config.compute_fraction))
        setCfgGpu(GPU_OPTS.some(o => o.value === s.config!.gpu_workers)
          ? s.config.gpu_workers : snap(GPU_OPTS, s.config.gpu_workers))
      }
      setApplying(false)     // fresh stats = the (re)started cluster is alive
    }
    window.addEventListener('spyde:dask_stats', onStats)
    // Staleness sweep: clear the readout when samples stop arriving.
    const sweep = window.setInterval(() => {
      if (lastAt.current && Date.now() - lastAt.current > STALE_MS) {
        lastAt.current = 0
        setStats(null)
        setOpen(false)
      }
    }, 2_000)
    return () => {
      window.removeEventListener('spyde:dask_stats', onStats)
      window.clearInterval(sweep)
    }
  }, [])

  if (!stats) return null

  const nTasks = stats.tasks.executing + stats.tasks.queued
  const cpu = stats.host_cpu ?? Math.max(0, ...stats.workers.map(w => w.cpu))
  const gpu = stats.gpu

  return (
    <div style={S.root}>
      <button
        data-testid="dask-monitor"
        title="Compute activity — click for per-worker detail"
        style={{ ...S.seg, ...(open ? S.segOpen : {}) }}
        onClick={() => setOpen(v => !v)}
      >
        <span style={{ color: pctColor(cpu) }}>CPU {cpu.toFixed(0)}%</span>
        {typeof stats.host_mem === 'number' && (
          <span style={{ color: pctColor(stats.host_mem) }}>
            MEM {stats.host_mem.toFixed(0)}%
          </span>
        )}
        {gpu && <span style={{ color: pctColor(gpu.util) }}>GPU {gpu.util.toFixed(0)}%</span>}
        <span style={{ color: nTasks > 0 ? '#89b4fa' : '#6c7086' }}>
          {nTasks > 0 ? `${nTasks} tasks` : 'idle'}
        </span>
      </button>
      {open && (
        <div style={{ ...S.pop, width: limitsOpen ? 505 : 330 }}
          data-testid="dask-monitor-popover">
        <div style={S.cols}>
        <div style={S.leftCol}>
          <div style={S.popTitle}>
            Compute — {stats.workers.length} workers,{' '}
            {stats.tasks.executing} running / {stats.tasks.queued} queued
          </div>
          {/* Column legend (the bare glyph version read as noise — "what is 0▶?"). */}
          <div style={{ ...S.row, ...S.head }}>
            <span style={S.wname} />
            <div style={{ flex: 1 }}>cpu</div>
            <span style={S.wnum} />
            <span style={S.wmem}>mem (GB)</span>
            <span style={S.wtasks}>tasks</span>
          </div>
          {stats.workers.map((w) => (
            <div key={w.name} style={S.row} data-testid={`dask-worker-${w.name}`}>
              <span style={S.wname}>w{w.name}</span>
              <div style={S.track}>
                <div style={{ ...S.fill, width: `${Math.min(100, w.cpu)}%`,
                              background: pctColor(w.cpu) }} />
              </div>
              <span style={S.wnum}>{w.cpu.toFixed(0)}%</span>
              <span style={S.wmem}>
                {fmtGB(w.mem)}{w.mem_limit > 0 ? `/${fmtGB(w.mem_limit)}` : ''}
              </span>
              <span style={S.wtasks}
                title={`${w.executing} running${w.ready ? `, ${w.ready} queued` : ''}`}>
                {w.executing + w.ready > 0
                  ? `${w.executing}${w.ready > 0 ? `+${w.ready}` : ''}`
                  : '–'}
              </span>
            </div>
          ))}
          {gpu && (
            <div style={S.row} data-testid="dask-gpu-row">
              <span style={S.wname}>GPU</span>
              <div style={S.track}>
                <div style={{ ...S.fill, width: `${Math.min(100, gpu.util)}%`,
                              background: pctColor(gpu.util) }} />
              </div>
              <span style={S.wnum}>{gpu.util.toFixed(0)}%</span>
              <span style={S.wmem}>
                {(gpu.vram_used / 1024).toFixed(1)}/{(gpu.vram_total / 1024).toFixed(1)}
              </span>
              <span style={S.wtasks} />
            </div>
          )}
          <div style={S.footRow}>
            {state.dashboardUrl && (
              <button style={S.dash}
                onClick={() => window.electron.openExternal(state.dashboardUrl!)}>
                Open full Dask dashboard ↗
              </button>
            )}
            {/* Toggles the Compute-limits column open to the right. */}
            <button data-testid="compute-limits-toggle" style={S.limitsToggle}
              onClick={() => setLimitsOpen(v => !v)}>
              ⚙ Limits {limitsOpen ? '▸' : '◂'}
            </button>
          </div>
        </div>
        {limitsOpen && (
          <div style={S.rightCol} data-testid="compute-limits-panel">
            {/* Apply restarts the cluster with the new plan. (The manual
                "Trim memory" button was removed — the backend trims
                automatically after each batch instead.) */}
            <div style={S.cfgTitle}>Compute limits</div>
            <div style={S.cfgRow}>
              <span style={S.cfgLbl}>Max RAM</span>
              <Dropdown testid="compute-ram" value={cfgRam ?? '0.65'}
                options={RAM_OPTS} onChange={setCfgRam} width="100%" />
            </div>
            <div style={S.cfgRow}>
              <span style={S.cfgLbl}>CPU use</span>
              <Dropdown testid="compute-cpu" value={cfgCpu ?? '0.75'}
                options={CPU_OPTS} onChange={setCfgCpu} width="100%" />
            </div>
            <div style={S.cfgRow}>
              <span style={S.cfgLbl}>GPU feeders</span>
              <Dropdown testid="compute-gpu" value={cfgGpu ?? '4'}
                options={GPU_OPTS} onChange={setCfgGpu} width="100%" />
            </div>
            <button
              data-testid="compute-apply"
              style={{ ...S.apply, opacity: applying ? 0.5 : 1 }}
              disabled={applying}
              title="Applies the limits by restarting the compute cluster — in-flight computes are cancelled"
              onClick={() => {
                setApplying(true)
                sendAction('compute_configure', {
                  mem_fraction: Number(cfgRam ?? 0.65),
                  compute_fraction: Number(cfgCpu ?? 0.75),
                  gpu_workers: cfgGpu ?? '4',
                })
              }}
            >
              {applying ? 'Restarting cluster…' : 'Apply (restarts)'}
            </button>
          </div>
        )}
        </div>
        </div>
      )}
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  root: { position: 'relative', flexShrink: 0 },
  seg: {
    display: 'inline-flex', alignItems: 'center', gap: 8,
    background: 'transparent', border: '1px solid #313244', borderRadius: 4,
    color: '#a6adc8', fontSize: 11, cursor: 'pointer', padding: '2px 8px',
    fontVariantNumeric: 'tabular-nums',
  },
  segOpen: { background: '#1e1e2e', borderColor: '#45475a' },
  pop: {
    position: 'absolute', bottom: 26, right: 0, zIndex: 9300,
    background: '#1e1e2e', border: '1px solid #313244',
    borderRadius: 8, padding: 8, boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
  },
  cols: { display: 'flex', gap: 10, alignItems: 'stretch' },
  leftCol: { flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 },
  rightCol: {
    flex: '0 0 160px', display: 'flex', flexDirection: 'column', gap: 6,
    borderLeft: '1px solid #313244', paddingLeft: 10,
  },
  footRow: {
    marginTop: 4, display: 'flex', alignItems: 'center',
    justifyContent: 'space-between', gap: 8,
  },
  limitsToggle: {
    background: 'transparent', color: '#a6adc8', border: '1px solid #45475a',
    borderRadius: 5, padding: '2px 8px', fontSize: 10.5, cursor: 'pointer',
    flexShrink: 0, marginLeft: 'auto',
  },
  popTitle: { fontSize: 10.5, color: '#a6adc8', paddingBottom: 4 },
  head: { color: '#6c7086', fontSize: 9.5, textTransform: 'uppercase', letterSpacing: 0.5 },
  row: {
    display: 'flex', alignItems: 'center', gap: 6,
    fontSize: 10.5, color: '#cdd6f4', fontVariantNumeric: 'tabular-nums',
  },
  wname: { width: 30, color: '#a6adc8', flexShrink: 0 },
  track: { flex: 1, height: 5, borderRadius: 3, background: '#313244', overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 3, transition: 'width 300ms linear' },
  wnum: { width: 32, textAlign: 'right', flexShrink: 0 },
  wmem: { width: 66, textAlign: 'right', color: '#a6adc8', flexShrink: 0 },
  wtasks: { width: 40, textAlign: 'right', color: '#89b4fa', flexShrink: 0 },
  cfgTitle: {
    fontSize: 9.5, color: '#6c7086', textTransform: 'uppercase', letterSpacing: 0.5,
  },
  cfgRow: { display: 'flex', flexDirection: 'column', gap: 2 },
  cfgLbl: { fontSize: 10, color: '#a6adc8', whiteSpace: 'nowrap' },
  apply: {
    marginTop: 2, background: '#313244', color: '#cdd6f4',
    border: '1px solid #45475a', borderRadius: 5, padding: '4px 8px',
    fontSize: 10.5, cursor: 'pointer',
  },
  dash: {
    marginTop: 4, background: 'none', border: 'none', color: '#89b4fa',
    fontSize: 11, cursor: 'pointer', padding: 0, textAlign: 'left',
  },
}
