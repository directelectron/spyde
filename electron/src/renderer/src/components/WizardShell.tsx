/**
 * WizardShell.tsx — shared chrome + field primitives for the staged action
 * carets (Orientation / Vector-Orientation / Find-Vectors / Center-Zero-Beam).
 *
 * Every wizard is the same box: a header (title + ✕), an optional tab row, the
 * step content, and a status footer — only the steps differ. This module owns
 * that chrome and the common form controls so each wizard is just its content.
 */
import React from 'react'
import { Dropdown } from './Dropdown'

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

/** Whether the caret currently open belongs to a `beta:` action.
 *
 *  FloatingToolbar provides this for whichever action it opened, so a wizard
 *  never declares its own beta status — that would be a second copy of a fact
 *  the toolbar schema already states, free to drift from it. Every caret built
 *  on WizardShell therefore gets the ribbon with no change of its own.
 */
export const BetaContext = React.createContext(false)

export function WizardShell({
  testid, title, posStyle, onClose, closeTestid, status, statusTestid, children,
  width,
}: ShellProps) {
  const beta = React.useContext(BetaContext)
  return (
    <div data-testid={testid}
      style={{ ...posStyle, ...S.box, ...(width ? { width } : {}) }}>
      <div style={S.head}>
        <span style={S.title}>{title}</span>
        <button data-testid={closeTestid} style={S.close} onClick={onClose}>✕</button>
      </div>
      {beta && (
        <div data-testid={`${testid}-beta`} style={S.betaRibbon}>
          BETA · still under development, may change
        </div>
      )}
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

/** A label + control field row.
 *
 *  `label` takes a node, not just a string, so a caret can hang an affordance
 *  off it — an ⓘ disclosure, a unit badge — without a second row. The row wraps
 *  so an expanded disclosure flows underneath instead of squeezing the control.
 */
export function Field({ label, children }: {
  label: React.ReactNode; children: React.ReactNode
}) {
  return (
    <div style={S.fieldRow}>
      <span style={{ ...S.lbl, display: 'flex', alignItems: 'center', gap: 4 }}>
        {label}
      </span>
      {children}
    </div>
  )
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

/** Themed dropdown (menubar look) — NOT a native <select>; see Dropdown.tsx
 *  for the Playwright interaction pattern (`selectOption` does not apply). */
export function Select<T extends string>({ value, options, onChange, testid }: {
  value: T
  options: readonly { value: T; label: string }[]
  onChange: (v: T) => void
  testid: string
}) {
  return <Dropdown value={value} options={options} onChange={onChange} testid={testid} />
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
  betaRibbon: {
    background: '#fab387', color: '#11111b', textAlign: 'center' as const,
    fontSize: 9, fontWeight: 700, letterSpacing: 0.3,
    padding: '2px 6px', borderRadius: 4,
  },
  title: { fontSize: 11, fontWeight: 600, color: '#cdd6f4' },
  close: { background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer', fontSize: 12 },
  tabRow: { display: 'flex', gap: 2, borderBottom: '1px solid #313244', paddingBottom: 4 },
  tab: { background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer', fontSize: 11, padding: '2px 7px', borderRadius: 4 },
  tabActive: { background: '#313244', border: 'none', color: '#cdd6f4', cursor: 'pointer', fontSize: 11, padding: '2px 7px', borderRadius: 4, fontWeight: 600 },
  tabLocked: { background: 'none', border: 'none', color: '#494d64', cursor: 'not-allowed', fontSize: 11, padding: '2px 7px', borderRadius: 4 },
  page: { display: 'flex', flexDirection: 'column', gap: 6, paddingTop: 2 },
  groupLabel: { fontSize: 10.5, color: '#6c7086', marginTop: 4, borderTop: '1px solid #313244', paddingTop: 5 },
  // `flexWrap` so an expanded ⓘ disclosure (width:100%) drops to its own line
  // rather than crushing the control beside it.
  fieldRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, flexWrap: 'wrap' },
  lbl: { fontSize: 10, color: '#a6adc8', whiteSpace: 'nowrap' },
  num: { background: '#11111b', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 4, padding: '3px 5px', fontSize: 11 },
  sel: { background: '#11111b', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 4, padding: '3px 5px', fontSize: 11 },
  sliderRow: { display: 'flex', alignItems: 'center', gap: 4 },
  sliderVal: { fontSize: 10, color: '#cdd6f4', minWidth: 28, textAlign: 'right' },
  check: { fontSize: 11, color: '#cdd6f4', display: 'flex', alignItems: 'center', gap: 6 },
  hint: { fontSize: 10, color: '#6c7086', fontStyle: 'italic' },
  fileBtn: { background: '#313244', color: '#cdd6f4', border: '1px solid #45475a', borderRadius: 4, padding: '4px 8px', fontSize: 11, cursor: 'pointer', maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', alignSelf: 'flex-start' },
  primary: { background: '#89b4fa', color: '#11111b', border: 'none', borderRadius: 5, padding: '6px 10px', fontSize: 12, fontWeight: 600, cursor: 'pointer', alignSelf: 'flex-start' },
  primaryBusy: { background: '#45475a', color: '#a6adc8', cursor: 'progress' },
  status: { fontSize: 10, color: '#a6adc8', borderTop: '1px solid #313244', paddingTop: 4 },
  cifList: { display: 'flex', flexDirection: 'column', gap: 2 },
  cifRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, background: '#11111b', borderRadius: 4, padding: '2px 6px' },
  cifName: { fontSize: 10, color: '#cdd6f4', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
}

/** The primary action with a busy state: disabled and saying so while the
 *  backend works, so a second click cannot queue a second run. */
export function PrimaryButton({ busy, label, onClick, testid }: {
  busy: boolean; label: string; onClick: () => void; testid: string
}) {
  return (
    <button data-testid={testid} disabled={busy} onClick={onClick}
      style={{ ...S.primary, ...(busy ? S.primaryBusy : null) }}>
      {busy ? 'Running…' : label}
    </button>
  )
}

/**
 * ⓘ that opens a short paragraph of help under a control's label. A popover
 * rather than inline text because labels are `white-space: nowrap` so a
 * control label never breaks mid-word; inline help inherited that and ran off
 * the edge of the caret. Escape or a click on the text closes it.
 */
export function Info({ text, testid, width = 216 }: {
  text: string; testid: string; width?: number
}) {
  const [open, setOpen] = React.useState(false)
  React.useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])
  return (
    <span style={{ position: 'relative', display: 'inline-flex' }}>
      <button data-testid={testid} aria-expanded={open} title="More information"
        style={infoBtn} onClick={() => setOpen(v => !v)}>ⓘ</button>
      {open && (
        <div data-testid={`${testid}-text`} style={{ ...infoText, width }}
          onClick={() => setOpen(false)}>{text}</div>
      )}
    </span>
  )
}

const infoBtn: React.CSSProperties = {
  background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
  fontSize: 11, padding: 0, lineHeight: 1, flex: '0 0 auto',
}
const infoText: React.CSSProperties = {
  position: 'absolute', top: 'calc(100% + 4px)', left: -8, zIndex: 20,
  fontSize: 10.5, color: '#cdd6f4', background: '#1e1e2e',
  border: '1px solid #45475a', borderRadius: 5, padding: '6px 8px',
  lineHeight: 1.45, boxShadow: '0 8px 20px rgba(0,0,0,0.55)',
  // `S.lbl` is nowrap so control labels never break mid-word; this is a
  // paragraph and must opt back out of that.
  whiteSpace: 'normal', cursor: 'pointer',
}
