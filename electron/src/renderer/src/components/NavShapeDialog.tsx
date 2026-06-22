/**
 * NavShapeDialog.tsx — confirm the navigation (scan) shape + step size when
 * opening a navigated dataset (e.g. a 4D-STEM MRC scan or a flat stack of
 * diffraction patterns).
 *
 * The backend emits `nav_shape_prompt` with the inferred shape + current step
 * size and waits; this modal lets the user confirm/override and replies with
 * `confirm_nav_shape`. Pre-filled with the inferred values, so the common case
 * is just pressing Open.
 */
import React, { useMemo, useState } from 'react'

export interface NavShapePrompt {
  nav_shape: number[]        // inferred (display x, y)
  n_patterns: number         // total frames (for stack factor hints)
  signal_shape: number[]
  scale: number
  units: string
  filename: string
}

const ACCENT = '#89b4fa'

// Integer factor pairs (x, y) of n that tile the stack — handy presets for a
// flat stack the user wants to fold into a 2-D grid.
function factorPairs(n: number): Array<[number, number]> {
  const out: Array<[number, number]> = []
  for (let x = 1; x <= Math.sqrt(n); x++) {
    if (n % x === 0) out.push([n / x, x])   // (wider, shorter) → display (x, y)
  }
  return out.reverse()
}

export function NavShapeDialog({
  prompt, onConfirm, onCancel,
}: {
  prompt: NavShapePrompt
  onConfirm: (navShape: number[], step: number, units: string) => void
  onCancel: () => void
}) {
  const [nx, setNx] = useState(prompt.nav_shape[0] ?? prompt.n_patterns)
  const [ny, setNy] = useState(prompt.nav_shape[1] ?? 1)
  const [step, setStep] = useState(prompt.scale || 1)
  const [units, setUnits] = useState(prompt.units || 'nm')

  const product = nx * ny
  const matches = product === prompt.n_patterns
  const presets = useMemo(() => factorPairs(prompt.n_patterns).slice(0, 6), [prompt.n_patterns])

  return (
    <div style={styles.overlay} data-testid="nav-shape-dialog">
      <div style={styles.dialog} onClick={(e) => e.stopPropagation()}>
        <h3 style={styles.title}>Scan shape — {prompt.filename}</h3>
        <p style={styles.sub}>
          {prompt.n_patterns.toLocaleString()} diffraction patterns of{' '}
          {prompt.signal_shape.join('×')}. Confirm the scan grid and step size.
        </p>

        <div style={styles.row}>
          <label style={styles.label}>Scan grid (X × Y)</label>
          <div style={styles.inputs}>
            <input data-testid="nav-shape-x" type="number" min={1} value={nx}
              onChange={(e) => setNx(Math.max(1, parseInt(e.target.value) || 1))}
              style={styles.num} />
            <span style={styles.times}>×</span>
            <input data-testid="nav-shape-y" type="number" min={1} value={ny}
              onChange={(e) => setNy(Math.max(1, parseInt(e.target.value) || 1))}
              style={styles.num} />
          </div>
        </div>

        {presets.length > 1 && (
          <div style={styles.presets}>
            {presets.map(([px, py]) => (
              <button key={`${px}x${py}`} data-testid={`nav-preset-${px}x${py}`}
                onClick={() => { setNx(px); setNy(py) }}
                style={(px === nx && py === ny) ? styles.presetActive : styles.preset}>
                {px}×{py}
              </button>
            ))}
          </div>
        )}

        <div style={styles.row}>
          <label style={styles.label}>Step size</label>
          <div style={styles.inputs}>
            <input data-testid="nav-step" type="number" min={0} step="any" value={step}
              onChange={(e) => setStep(parseFloat(e.target.value) || 0)}
              style={styles.num} />
            <input data-testid="nav-units" type="text" value={units}
              onChange={(e) => setUnits(e.target.value)}
              style={{ ...styles.num, width: 56 }} />
            <span style={styles.perpx}>/ px</span>
          </div>
        </div>

        {!matches && (
          <div data-testid="nav-shape-warn" style={styles.warn}>
            {nx} × {ny} = {product.toLocaleString()} ≠ {prompt.n_patterns.toLocaleString()} frames
          </div>
        )}

        <div style={styles.footer}>
          <button data-testid="nav-cancel" style={styles.cancel} onClick={onCancel}>Cancel</button>
          <button data-testid="nav-open" style={{ ...styles.open, opacity: matches ? 1 : 0.5 }}
            disabled={!matches}
            onClick={() => onConfirm([nx, ny], step, units)}>
            Open
          </button>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9500,
    background: 'rgba(17,17,27,0.6)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  dialog: {
    width: 380, background: '#1e1e2e', border: '1px solid #313244',
    borderRadius: 10, padding: 18, color: '#cdd6f4',
    boxShadow: '0 16px 40px rgba(0,0,0,0.55)', fontSize: 13,
  },
  title: { margin: '0 0 4px', fontSize: 16, fontWeight: 600 },
  sub: { margin: '0 0 16px', fontSize: 12, color: '#a6adc8', lineHeight: 1.4 },
  row: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 },
  label: { fontSize: 12.5, color: '#bac2de' },
  inputs: { display: 'flex', alignItems: 'center', gap: 6 },
  num: {
    width: 70, background: '#11111b', border: '1px solid #313244', borderRadius: 6,
    color: '#cdd6f4', padding: '5px 8px', fontSize: 13,
  },
  times: { color: '#6c7086' },
  perpx: { color: '#6c7086', fontSize: 11 },
  presets: { display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 14 },
  preset: {
    background: '#11111b', border: '1px solid #313244', color: '#a6adc8',
    borderRadius: 6, padding: '3px 9px', cursor: 'pointer', fontSize: 11,
  },
  presetActive: {
    background: ACCENT, border: `1px solid ${ACCENT}`, color: '#11111b',
    borderRadius: 6, padding: '3px 9px', cursor: 'pointer', fontSize: 11, fontWeight: 600,
  },
  warn: {
    background: 'rgba(243,139,168,0.12)', borderLeft: '3px solid #f38ba8',
    color: '#f38ba8', padding: '7px 10px', borderRadius: 6, fontSize: 12, marginBottom: 12,
  },
  footer: { display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 4 },
  cancel: {
    background: 'transparent', border: '1px solid #313244', color: '#cdd6f4',
    borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 12,
  },
  open: {
    background: ACCENT, border: 'none', color: '#11111b', fontWeight: 600,
    borderRadius: 6, padding: '6px 18px', cursor: 'pointer', fontSize: 12,
  },
}
