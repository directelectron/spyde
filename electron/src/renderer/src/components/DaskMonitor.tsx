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

const STALE_MS = 7_000          // hide after ~3 missed samples

const pctColor = (p: number) =>
  p >= 95 ? '#f38ba8' : p >= 80 ? '#f9e2af' : '#a6e3a1'

const fmtGB = (bytes: number) => (bytes / 1024 ** 3).toFixed(1)

export function DaskMonitor() {
  const { state, sendAction } = useSpyDE()
  const [stats, setStats] = React.useState<DaskStatsMessage | null>(null)
  const [open, setOpen] = React.useState(false)
  const lastAt = React.useRef(0)

  React.useEffect(() => {
    const onStats = (e: Event) => {
      lastAt.current = Date.now()
      setStats((e as CustomEvent).detail as DaskStatsMessage)
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
        <div style={S.pop} data-testid="dask-monitor-popover">
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
          <div style={S.footer}>
            {state.dashboardUrl && (
              <button style={S.dash}
                onClick={() => window.electron.openExternal(state.dashboardUrl!)}>
                Open full Dask dashboard ↗
              </button>
            )}
            {/* Reclaim post-batch allocator retention across workers + backend
                (gc + torch cache + Windows EmptyWorkingSet). NB it cannot free
                the ~0.7-1 GB/worker of loaded torch/hyperspy runtime — that
                baseline only goes away with the workers themselves. */}
            <button data-testid="dask-trim" style={S.trim}
              title="Free leftover batch memory in idle workers (the ~1 GB/worker library runtime always remains)"
              onClick={() => sendAction('dask_trim', {})}>
              Trim memory
            </button>
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
    width: 330, background: '#1e1e2e', border: '1px solid #313244',
    borderRadius: 8, padding: 8, boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
    display: 'flex', flexDirection: 'column', gap: 4,
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
  footer: {
    marginTop: 4, display: 'flex', alignItems: 'center',
    justifyContent: 'space-between', gap: 8,
  },
  dash: {
    background: 'none', border: 'none', color: '#89b4fa',
    fontSize: 11, cursor: 'pointer', padding: 0, textAlign: 'left',
  },
  trim: {
    background: 'transparent', color: '#a6adc8', border: '1px solid #45475a',
    borderRadius: 5, padding: '2px 8px', fontSize: 10.5, cursor: 'pointer',
  },
}
