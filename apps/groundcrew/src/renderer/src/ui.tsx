/**
 * ui.tsx — the shared primitives.
 *
 * `Field` is the important one. The previous design put every setting behind a
 * card that had to be clicked to reveal what it did, which made a value both
 * hard to read and hard to change. These are labelled input boxes: the current
 * value is always legible, and editing is typing.
 *
 * A `Field` is COMMITTED on blur or Enter, not on every keystroke. Each commit
 * is a property write over the one connection to the camera, and writing on
 * keystroke would send "2", "25", "250" while someone typed 250.
 */
import React, { useEffect, useRef, useState } from 'react'
import { C, FONT_MONO, stateOf } from './theme'

// ── Layout ────────────────────────────────────────────────────────────────────

export function Section({ title, right, children }: {
  title: string; right?: React.ReactNode; children: React.ReactNode
}) {
  return (
    <section style={{ marginBottom: 18 }}>
      <header style={{
        display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <h2 style={{
          margin: 0, fontSize: 10.5, fontWeight: 700, letterSpacing: '.09em',
          textTransform: 'uppercase', color: C.textMuted,
        }}>{title}</h2>
        {right}
      </header>
      {children}
    </section>
  )
}

// ── Status ────────────────────────────────────────────────────────────────────

export function Pill({ state, children }: { state: string; children?: React.ReactNode }) {
  const s = stateOf(state)
  return (
    <span
      data-state={state}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 5,
        padding: '2px 8px', borderRadius: 999, background: s.bg, color: s.fg,
        fontSize: 10.5, fontWeight: 700, letterSpacing: '.04em',
        whiteSpace: 'nowrap',
      }}
    >
      <span style={{
        width: 6, height: 6, background: s.fg,
        // A shape difference as well as a colour one — a disc for pass, a
        // diamond for everything else — so the board is readable without
        // colour vision.
        transform: state === 'ok' ? 'none' : 'rotate(45deg)',
        borderRadius: state === 'ok' ? 999 : 1,
      }} />
      {children ?? s.label}
    </span>
  )
}

// ── Inputs ────────────────────────────────────────────────────────────────────

/**
 * Render a value for a field, without inventing precision or losing it.
 *
 * The server returns float32 widened to double, so a frame rate of 340.82
 * arrives as 340.82000732421875 and overflows the box. Six significant digits
 * is past any real instrument precision and short enough to read. Integers and
 * strings are left exactly as they came — a serial number or a path must never
 * be rounded.
 */
export function format(value: string | number | null | undefined): string {
  if (value == null || value === '') return ''
  if (typeof value !== 'number') return String(value)
  if (!Number.isFinite(value)) return '—'
  if (Number.isInteger(value)) return String(value)
  return String(Number(value.toPrecision(6)))
}

/** A labelled value that can be edited in place. `onCommit` fires on blur/Enter. */
export function Field({ label, value, unit, onCommit, disabled, testId, hint }: {
  label: string
  value: string | number | null | undefined
  unit?: string
  onCommit?: (raw: string) => void
  disabled?: boolean
  testId?: string
  hint?: string
}) {
  const shown = format(value)
  const [draft, setDraft] = useState(shown)
  const [editing, setEditing] = useState(false)
  const ref = useRef<HTMLInputElement>(null)

  // Track the backend while NOT editing. Without the guard, a status poll
  // landing mid-keystroke would overwrite what is being typed.
  useEffect(() => { if (!editing) setDraft(shown) }, [shown, editing])

  const readOnly = !onCommit || disabled
  const commit = () => {
    setEditing(false)
    if (onCommit && draft !== shown) onCommit(draft)
  }

  return (
    <label style={{
      display: 'grid', gridTemplateColumns: '1fr auto', alignItems: 'center',
      gap: 8, marginBottom: 6,
    }}>
      <span style={{ fontSize: 12, color: C.textMuted }} title={hint}>{label}</span>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
        <input
          ref={ref}
          value={draft}
          data-testid={testId}
          readOnly={readOnly}
          disabled={disabled}
          onChange={(e) => { setEditing(true); setDraft(e.target.value) }}
          onFocus={() => setEditing(true)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { commit(); ref.current?.blur() }
            if (e.key === 'Escape') { setEditing(false); setDraft(shown); ref.current?.blur() }
          }}
          placeholder={value == null ? '—' : ''}
          style={{
            width: 92, textAlign: 'right', padding: '4px 7px',
            background: readOnly ? 'transparent' : C.bgSunken,
            border: `1px solid ${readOnly ? 'transparent' : C.borderStrong}`,
            borderRadius: 5, color: value == null ? C.textMuted : C.text,
            fontSize: 12.5, fontFamily: FONT_MONO,
            fontVariantNumeric: 'tabular-nums',
            outline: 'none', cursor: readOnly ? 'default' : 'text',
          }}
        />
        <span style={{ fontSize: 11, color: C.textMuted, width: 30 }}>{unit ?? ''}</span>
      </span>
    </label>
  )
}

export function Btn({ children, onClick, disabled, active, tone, testId, title, wide }: {
  children: React.ReactNode
  onClick?: () => void
  disabled?: boolean
  active?: boolean
  tone?: 'default' | 'danger' | 'go'
  testId?: string
  title?: string
  wide?: boolean
}) {
  const fg = tone === 'danger' ? '#ef6b6b' : tone === 'go' ? '#41d18a' : C.textDim
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      data-testid={testId}
      style={{
        flex: wide ? 1 : undefined,
        padding: '7px 12px',
        background: active ? C.accentSunken : C.panelRaised,
        border: `1px solid ${active ? C.accent : C.border}`,
        borderRadius: 6, color: disabled ? C.textMuted : fg,
        fontSize: 12.5, cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.55 : 1, whiteSpace: 'nowrap',
      }}
    >{children}</button>
  )
}

export function Select({ label, value, options, onChange, testId }: {
  label: string; value: string; options: readonly string[]
  onChange: (v: string) => void; testId?: string
}) {
  return (
    <label style={{
      display: 'grid', gridTemplateColumns: '1fr auto', alignItems: 'center',
      gap: 8, marginBottom: 6,
    }}>
      <span style={{ fontSize: 12, color: C.textMuted }}>{label}</span>
      <select
        value={value}
        data-testid={testId}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: 129, padding: '4px 6px', background: C.bgSunken,
          border: `1px solid ${C.borderStrong}`, borderRadius: 5,
          color: C.text, fontSize: 12.5, outline: 'none',
        }}
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  )
}

/** A pane that exists but has nothing wired behind it yet. Says so plainly. */
export function NotBuilt({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '100%', padding: 40,
    }}>
      <div style={{ maxWidth: 460, textAlign: 'center' }}>
        <div style={{ fontSize: 15, fontWeight: 600, color: C.textDim, marginBottom: 10 }}>
          {title}
        </div>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: C.textMuted }}>
          {children}
        </p>
      </div>
    </div>
  )
}
