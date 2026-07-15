/**
 * EnvSetupOverlay.tsx — the floating "Setting up SpyDE" card shown on first
 * packaged launch while `uv sync` builds the Python environment (hundreds of MB,
 * incl. PyTorch). Before this, first launch showed a single static "Setting up
 * Python environment…" line with a blank canvas for minutes — indistinguishable
 * from a hang. This overlay is driven by real parsed uv progress (EnvSetupState
 * in SpyDEContext): a spinner, the current phase, a friendly step line, a real
 * download % when uv reports one (indeterminate otherwise), and a live scrolling
 * tail of the raw uv output so it is always visibly moving.
 */
import { useEffect, useRef } from 'react'
import type { EnvSetupState, EnvPhase } from '../kernel/SpyDEContext'

const ACCENT = '#89b4fa'

const PHASES: Array<{ key: EnvPhase | 'torch'; label: string }> = [
  { key: 'resolving', label: 'Resolve' },
  { key: 'downloading', label: 'Download' },
  { key: 'torch', label: 'PyTorch' },
  { key: 'installing', label: 'Install' },
  { key: 'building', label: 'Build' },
]

// Order used to decide which phase pills are "done" vs "upcoming".
const PHASE_ORDER: EnvPhase[] = ['resolving', 'downloading', 'torch', 'installing', 'building']

function phaseIndex(p: EnvPhase): number {
  const i = PHASE_ORDER.indexOf(p)
  return i < 0 ? 0 : i
}

export function EnvSetupOverlay({ setup }: { setup: EnvSetupState }) {
  const tailRef = useRef<HTMLDivElement>(null)
  // Keep the log tail pinned to the newest line.
  useEffect(() => {
    const el = tailRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [setup.lines])

  const active = setup.phase === 'working' ? 'resolving' : setup.phase
  const activeIdx = phaseIndex(active)
  const tail = setup.lines.slice(-14)

  return (
    <div
      data-testid="env-setup-overlay"
      style={{
        position: 'fixed', inset: 0, zIndex: 99998,
        background: 'rgba(17,17,27,0.72)', backdropFilter: 'blur(2px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 24, userSelect: 'text',
      }}
    >
      <div style={{
        maxWidth: 560, width: '100%',
        display: 'flex', flexDirection: 'column',
        padding: '26px 30px', borderRadius: 12,
        background: '#1e1e2e', border: `1px solid ${ACCENT}`,
        boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
        color: '#cdd6f4', fontFamily: 'system-ui, sans-serif',
      }}>
        {/* Header: spinner + title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
          <Spinner />
          <div style={{ fontSize: 18, fontWeight: 600 }}>Setting up SpyDE</div>
        </div>

        <div style={{ fontSize: 13, lineHeight: 1.5, color: '#a6adc8', marginBottom: 16 }}>
          First launch only — downloading the analysis environment (about 1–2&nbsp;GB,
          including PyTorch). This can take a few minutes on a fast connection; every
          later launch is instant.
        </div>

        {/* Phase pills */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 14, flexWrap: 'wrap' }}>
          {PHASES.map((p) => {
            const idx = phaseIndex(p.key as EnvPhase)
            const state = idx < activeIdx ? 'done' : idx === activeIdx ? 'active' : 'todo'
            return (
              <span key={p.key} style={{
                fontSize: 11, padding: '3px 9px', borderRadius: 999,
                border: `1px solid ${state === 'todo' ? '#313244' : ACCENT}`,
                background: state === 'active' ? ACCENT : state === 'done' ? '#313244' : 'transparent',
                color: state === 'active' ? '#11111b' : state === 'done' ? '#cdd6f4' : '#6c7086',
                fontWeight: state === 'active' ? 600 : 400,
              }}>
                {state === 'done' ? '✓ ' : ''}{p.label}
              </span>
            )
          })}
        </div>

        {/* Current step + progress bar */}
        <div
          data-testid="env-setup-step"
          style={{ fontSize: 14, fontWeight: 500, marginBottom: 8 }}
        >
          {setup.step}{setup.percent != null ? ` — ${setup.percent}%` : '…'}
        </div>
        <ProgressBar percent={setup.percent} />

        {/* Live raw log tail */}
        <div style={{
          marginTop: 16, fontSize: 10.5, color: '#6c7086',
          textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 5,
        }}>
          Setup log
        </div>
        <div
          ref={tailRef}
          data-testid="env-setup-log"
          style={{
            height: 128, overflow: 'auto',
            background: '#11111b', border: '1px solid #313244', borderRadius: 6,
            padding: '7px 10px',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 11, lineHeight: 1.55, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}
        >
          {tail.length === 0
            ? <span style={{ color: '#6c7086', fontStyle: 'italic' }}>Starting…</span>
            : tail.map((l, i) => (
                <div key={i} style={{ color: '#a6adc8' }}>{l.replace(/\n$/, '')}</div>
              ))}
        </div>
      </div>
    </div>
  )
}

/** Indeterminate when percent is null; a filled bar when we have a %. */
function ProgressBar({ percent }: { percent: number | null }) {
  const indeterminate = percent == null
  return (
    <div style={{
      position: 'relative', height: 6, borderRadius: 3, overflow: 'hidden',
      background: '#313244',
    }}>
      <div style={indeterminate
        ? {
            position: 'absolute', top: 0, bottom: 0, width: '35%',
            background: ACCENT, borderRadius: 3,
            animation: 'spyde-env-indeterminate 1.15s ease-in-out infinite',
          }
        : {
            position: 'absolute', top: 0, bottom: 0, left: 0,
            width: `${Math.max(2, percent)}%`,
            background: ACCENT, borderRadius: 3,
            transition: 'width 0.25s ease',
          }
      } />
      <style>{`
        @keyframes spyde-env-indeterminate {
          0%   { left: -35%; }
          100% { left: 100%; }
        }
        @keyframes spyde-env-spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

function Spinner() {
  return (
    <div style={{
      width: 18, height: 18, borderRadius: '50%',
      border: `2px solid #313244`, borderTopColor: ACCENT,
      animation: 'spyde-env-spin 0.8s linear infinite',
    }} />
  )
}
