/**
 * WizardShell.tsx — shared chrome + field primitives for the staged action
 * carets (Orientation / Vector-Orientation / Find-Vectors / Center-Zero-Beam).
 *
 * Every wizard is the same box: a header (title + ✕), an optional tab row, the
 * step content, and a status footer — only the steps differ. This module owns
 * that chrome and the common form controls so each wizard is just its content.
 */
import React from 'react'

interface ShellProps {
  testid: string
  title: string
  /** Placement style computed by FloatingToolbar (below the window, or floated
   *  to the window's right/left when below would run off the MDI area). */
  posStyle: React.CSSProperties
  onClose: () => void
  closeTestid: string
  status: string
  statusTestid: string
  children: React.ReactNode
  width?: number          // override the default box width (e.g. 2-column wizards)
}

export function WizardShell({
  testid, title, posStyle, onClose, closeTestid, status, statusTestid, children,
  width,
}: ShellProps) {
  return (
    <div data-testid={testid}
      style={{ ...posStyle, ...S.box, ...(width ? { width } : {}) }}>
      <div style={S.head}>
        <span style={S.title}>{title}</span>
        <button data-testid={closeTestid} style={S.close} onClick={onClose}>✕</button>
      </div>
      {children}
      <div data-testid={statusTestid} style={S.status}>{status}</div>
    </div>
  )
}

export function TabRow<T extends string>({ tabs, active, onSelect, locked, testid }: {
  tabs: readonly T[]
  active: T
  onSelect: (t: T) => void
  locked?: (t: T) => boolean
  testid: (t: T) => string
}) {
  return (
    <div style={S.tabRow}>
      {tabs.map(t => {
        const isLocked = locked?.(t) ?? false
        return (
          <button key={t} data-testid={testid(t)} disabled={isLocked}
            style={t === active ? S.tabActive : (isLocked ? S.tabLocked : S.tab)}
            onClick={() => !isLocked && onSelect(t)}>{t}</button>
        )
      })}
    </div>
  )
}

/** A label + control field row. */
export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div style={S.fieldRow}><label style={S.lbl}>{label}</label>{children}</div>
}

export function NumInput({ value, onChange, step = 'any', width = 64, testid }: {
  value: number; onChange: (n: number) => void; step?: string; width?: number; testid?: string
}) {
  // A bare `Number(e.target.value)` propagates NaN upward for a mid-edit or
  // malformed string (empty, "-", "1.", "e", …), which then flows into every
  // consumer's state (and often straight into a backend payload) as NaN. Keep
  // typing responsive by tracking the raw text locally, but only ever call
  // `onChange` with a finite number — an invalid/incomplete value is ignored
  // (the last valid `value` from the parent stays authoritative) rather than
  // clobbering state with NaN. Valid input's behavior is unchanged.
  const [draft, setDraft] = React.useState<string | null>(null)
  return (
    <input
      data-testid={testid} type="number" step={step}
      value={draft ?? value}
      style={{ ...S.num, width }}
      onChange={(e) => {
        const text = e.target.value
        setDraft(text)
        const n = Number(text)
        if (text !== '' && Number.isFinite(n)) onChange(n)
      }}
      onBlur={() => setDraft(null)}
    />
  )
}

export function Slider({ value, min, max, step, onChange, fmt, testid }: {
  value: number; min: number; max: number; step: number
  onChange: (n: number) => void; fmt?: (n: number) => string; testid: string
}) {
  return (
    <div style={S.sliderRow}>
      <input data-testid={testid} type="range" min={min} max={max} step={step} value={value}
        style={{ flex: 1, minWidth: 40 }} onChange={(e) => onChange(Number(e.target.value))} />
      <span style={S.sliderVal}>{fmt ? fmt(value) : value}</span>
    </div>
  )
}

export function Select<T extends string>({ value, options, onChange, testid }: {
  value: T
  options: readonly { value: T; label: string }[]
  onChange: (v: T) => void
  testid: string
}) {
  return (
    <select data-testid={testid} value={value} style={S.sel}
      onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  )
}

export function Check({ checked, onChange, label, testid }: {
  checked: boolean; onChange: (b: boolean) => void; label: string; testid: string
}) {
  return (
    <label style={S.check}>
      <input data-testid={testid} type="checkbox" checked={checked}
        onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}

const POP_BG = '#1e1e2e'
export const S: Record<string, React.CSSProperties> = {
  box: {
    background: POP_BG, border: '1px solid #313244', borderRadius: 8, padding: 8,
    width: 240, zIndex: 14, color: '#cdd6f4', boxShadow: '0 8px 24px rgba(0,0,0,0.55)',
    display: 'flex', flexDirection: 'column', gap: 6,
  },
  head: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  title: { fontSize: 11, fontWeight: 600, color: '#cdd6f4' },
  close: { background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer', fontSize: 12 },
  tabRow: { display: 'flex', gap: 2, borderBottom: '1px solid #313244', paddingBottom: 4 },
  tab: { background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer', fontSize: 11, padding: '2px 7px', borderRadius: 4 },
  tabActive: { background: '#313244', border: 'none', color: '#cdd6f4', cursor: 'pointer', fontSize: 11, padding: '2px 7px', borderRadius: 4, fontWeight: 600 },
  tabLocked: { background: 'none', border: 'none', color: '#494d64', cursor: 'not-allowed', fontSize: 11, padding: '2px 7px', borderRadius: 4 },
  page: { display: 'flex', flexDirection: 'column', gap: 6, paddingTop: 2 },
  fieldRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 },
  lbl: { fontSize: 10, color: '#a6adc8', whiteSpace: 'nowrap' },
  num: { background: '#11111b', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 4, padding: '3px 5px', fontSize: 11 },
  sel: { background: '#11111b', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 4, padding: '3px 5px', fontSize: 11 },
  sliderRow: { display: 'flex', alignItems: 'center', gap: 4 },
  sliderVal: { fontSize: 10, color: '#cdd6f4', minWidth: 28, textAlign: 'right' },
  check: { fontSize: 11, color: '#cdd6f4', display: 'flex', alignItems: 'center', gap: 6 },
  hint: { fontSize: 10, color: '#6c7086', fontStyle: 'italic' },
  fileBtn: { background: '#313244', color: '#cdd6f4', border: '1px solid #45475a', borderRadius: 4, padding: '4px 8px', fontSize: 11, cursor: 'pointer', maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', alignSelf: 'flex-start' },
  primary: { background: '#89b4fa', color: '#11111b', border: 'none', borderRadius: 5, padding: '6px 10px', fontSize: 12, fontWeight: 600, cursor: 'pointer', alignSelf: 'flex-start' },
  status: { fontSize: 10, color: '#a6adc8', borderTop: '1px solid #313244', paddingTop: 4 },
  cifList: { display: 'flex', flexDirection: 'column', gap: 2 },
  cifRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, background: '#11111b', borderRadius: 4, padding: '2px 6px' },
  cifName: { fontSize: 10, color: '#cdd6f4', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
}
