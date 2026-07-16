/**
 * Dropdown.tsx — the app's themed replacement for native <select>.
 *
 * Native select POPUPS ignore CSS and render as OS-default lists — the one
 * visual element that broke the dark theme (user feedback 2026-07-16: "every
 * pull down selection should be like the File / Examples / Help theming").
 * This reproduces the MenuBar dropdown styling exactly: #1e1e2e panel,
 * #313244 border + hover, #cdd6f4 text, rounded, shadowed.
 *
 * API mirrors the old WizardShell `Select` so call sites don't change:
 *     <Dropdown value={v} options={[{value,label}]} onChange={f} testid="x" />
 *
 * Testing: this is NOT a <select>, so Playwright's `selectOption()` does not
 * work — click the trigger (`data-testid={testid}`, which also carries
 * `data-value` = current value) then the option
 * (`data-testid={`${testid}-opt-${value}`}`).
 */
import React from 'react'

export function Dropdown<T extends string>({ value, options, onChange, testid, width }: {
  value: T
  options: readonly { value: T; label: string }[]
  onChange: (v: T) => void
  testid: string
  width?: number | string
}) {
  const [open, setOpen] = React.useState(false)
  const rootRef = React.useRef<HTMLDivElement>(null)

  // Close on outside click / Escape (same behaviour as MenuBar).
  React.useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const current = options.find((o) => o.value === value)
  return (
    <div ref={rootRef} style={{ ...S.root, ...(width !== undefined ? { width } : {}) }}>
      <button
        type="button" data-testid={testid} data-value={value}
        aria-haspopup="listbox" aria-expanded={open}
        style={{ ...S.trigger, ...(open ? S.triggerOpen : {}) }}
        onClick={() => setOpen(!open)}
      >
        <span style={S.triggerLabel}>{current?.label ?? String(value)}</span>
        <span style={S.caret}>▾</span>
      </button>
      {open && (
        <div role="listbox" style={S.menu}>
          {options.map((o) => (
            <button
              key={o.value} type="button" role="option"
              aria-selected={o.value === value}
              data-testid={`${testid}-opt-${o.value}`}
              style={{ ...S.item, ...(o.value === value ? S.itemSelected : {}) }}
              onClick={() => { setOpen(false); if (o.value !== value) onChange(o.value) }}
              onMouseEnter={(e) => { e.currentTarget.style.background = '#313244' }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = o.value === value ? '#2a2a3c' : 'transparent'
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  root: { position: 'relative', minWidth: 0 },
  trigger: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    gap: 6, width: '100%',
    background: '#11111b', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 4, padding: '3px 7px', fontSize: 11, cursor: 'pointer',
    textAlign: 'left',
  },
  triggerOpen: { borderColor: '#45475a', background: '#181825' },
  triggerLabel: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  caret: { fontSize: 9, color: '#6c7086', flex: '0 0 auto' },
  // The panel is a copy of MenuBar's `styles.dropdown` / `styles.item`.
  menu: {
    position: 'absolute', top: 'calc(100% + 3px)', left: 0, zIndex: 9500,
    minWidth: '100%', maxHeight: 260, overflowY: 'auto',
    background: '#1e1e2e', border: '1px solid #313244',
    borderRadius: 8, padding: 5, boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
  },
  item: {
    display: 'block', width: '100%', textAlign: 'left',
    border: 'none', background: 'transparent', color: '#cdd6f4',
    borderRadius: 5, padding: '5px 9px', fontSize: 11.5, cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  itemSelected: { background: '#2a2a3c', color: '#89b4fa', fontWeight: 600 },
}
