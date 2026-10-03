/**
 * ModelMenu.tsx — the detector model dropdown.
 *
 * A themed dropdown (the same panel and rows as Dropdown.tsx) whose rows carry
 * the model's icon, its short name and a one-line subtitle. Vendored models
 * (bundled or downloaded) come first, then a divider and the models the user
 * taught. A row's icon is the model's preferred input — the disk it would most
 * like to see (spyde/models/adapt.py write_icon); a model without one shows its
 * initials. Hovering a row, or moving to it with the keys, shows its card: the
 * full description, the chain it came from, what it was taught on, when, how
 * many marks, how general it still is. A taught model's row has a ⋯ with Rename
 * and Delete.
 *
 * Keyboard: Enter / Space / ↓ open; ↑ ↓ move; Enter picks; Esc closes.
 * Testing (the themed-dropdown pattern — not a <select>): click the trigger
 * (`testid`, carrying `data-value`), then `${testid}-opt-${id}`. The divider is
 * `${testid}-divider`, the card `${testid}-card`, a row's ⋯ `${testid}-more-${id}`,
 * then `${testid}-rename` / `${testid}-rename-input` / `${testid}-delete`.
 */
import React from 'react'

export interface ModelInfo {
  id: string
  label: string
  description?: string | null
  group: 'vendored' | 'local'
  icon?: string | null
  notes?: string | null
  version?: number | null
  taught?: boolean
  unsaved?: boolean
  name?: string | null
  created?: string | null
  trained_on?: string | null
  chain?: string[]
  marks?: number
  original_f1?: number | null
  original_f1_base?: number | null
}

export function initials(label: string): string {
  const words = label.replace(/^Unsaved:\s*/, '').split(/[\s_\-()+]+/).filter(Boolean)
  return words.slice(0, 2).map((w) => w[0]!.toUpperCase()).join('') || '?'
}

export function subtitle(m: ModelInfo): string {
  if (!m.taught) return 'built-in'
  if (m.unsaved) return 'not saved yet — name it below'
  const f1 = typeof m.original_f1 === 'number' ? ` · F1 ${m.original_f1.toFixed(2)}` : ''
  return `taught on ${m.trained_on ?? 'a dataset'}${f1}`
}

function Icon({ model, size }: { model: ModelInfo | undefined; size: number }) {
  const style = { ...iconBox, width: size, height: size, ...(model?.unsaved ? iconUnsaved : null) }
  if (model?.icon) return <img src={model.icon} alt="" style={{ ...style, objectFit: 'cover' }} draggable={false} />
  return <span style={{ ...style, fontSize: Math.round(size * 0.36) }}>{model ? initials(model.label) : '?'}</span>
}

export function ModelMenu({ models, value, onChange, onRename, onDelete, testid }: {
  models: readonly ModelInfo[]
  value: string
  onChange: (id: string) => void
  onRename: (id: string, name: string) => void
  onDelete: (id: string) => void
  testid: string
}) {
  const vendored = models.filter((m) => m.group !== 'local')
  const local = models.filter((m) => m.group === 'local')
  const order = [...vendored, ...local]
  const current = order.find((m) => m.id === value)
  const [open, setOpen] = React.useState(false)
  const [active, setActive] = React.useState(-1)
  const [card, setCard] = React.useState<{ model: ModelInfo; top: number } | null>(null)
  const [actionsFor, setActionsFor] = React.useState<string | null>(null)
  const [renaming, setRenaming] = React.useState<string | null>(null)
  const renamingRef = React.useRef<string | null>(null)
  renamingRef.current = renaming
  const [renameText, setRenameText] = React.useState('')
  const [confirmDelete, setConfirmDelete] = React.useState<string | null>(null)
  const rootRef = React.useRef<HTMLDivElement>(null)
  const listRef = React.useRef<HTMLDivElement>(null)
  const triggerRef = React.useRef<HTMLButtonElement>(null)

  const close = () => {
    setOpen(false); setCard(null); setActionsFor(null); setRenaming(null); setConfirmDelete(null)
  }
  const openMenu = () => {
    setOpen(true)
    setActive(Math.max(0, order.findIndex((m) => m.id === value)))
  }
  const pick = (id: string) => {
    close()
    if (id !== value) onChange(id)
    triggerRef.current?.focus()
  }

  React.useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) close()
    }
    // Esc closes from anywhere, as in Dropdown.tsx — focus can sit outside the
    // menu (on the body, after a rename input goes away).
    const onEsc = (e: KeyboardEvent) => { if (e.key === 'Escape' && !renamingRef.current) close() }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onEsc)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onEsc)
    }
  }, [open])

  const rowEl = (id: string) =>
    listRef.current?.querySelector<HTMLElement>(`[data-model-id="${CSS.escape(id)}"]`) ?? null
  const showCard = (m: ModelInfo) => {
    const el = rowEl(m.id)
    const list = listRef.current
    if (!el || !list) return
    setCard({ model: m, top: el.offsetTop - list.scrollTop })
  }
  // Keyboard focus follows the active row and shows its card.
  React.useEffect(() => {
    if (!open || active < 0 || !order[active]) return
    rowEl(order[active]!.id)?.scrollIntoView({ block: 'nearest' })
    showCard(order[active]!)
  }, [open, active])  // eslint-disable-line react-hooks/exhaustive-deps

  const onKey = (e: React.KeyboardEvent) => {
    if (renaming) return
    if (!open) {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') { e.preventDefault(); openMenu() }
      return
    }
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(order.length - 1, a + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(0, a - 1)) }
    else if (e.key === 'Home') { e.preventDefault(); setActive(0) }
    else if (e.key === 'End') { e.preventDefault(); setActive(order.length - 1) }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (order[active]) pick(order[active]!.id) }
    else if (e.key === 'Escape') { e.preventDefault(); close() }
  }

  const saveRename = (id: string) => {
    const name = renameText.trim()
    if (name) onRename(id, name)
    setRenaming(null); setActionsFor(null)
    // After the Enter that saved, not during it: a trigger focused mid-keydown
    // receives that Enter's activation and toggles the menu shut.
    setTimeout(() => triggerRef.current?.focus(), 0)
  }

  const row = (m: ModelInfo) => {
    const index = order.indexOf(m)
    const selected = m.id === value
    const showActions = actionsFor === m.id && m.taught && !m.unsaved
    return (
      <div key={m.id} role="option" aria-selected={selected} data-model-id={m.id}
        data-testid={`${testid}-opt-${m.id}`}
        style={{ ...item, ...(selected ? itemSelected : null), ...(index === active ? itemActive : null),
                 ...(m.unsaved ? itemUnsaved : null) }}
        onMouseEnter={() => { setActive(index); showCard(m) }}
        // Keep focus on the trigger, so the keys keep working after a mouse pick.
        onMouseDown={(e) => { if (renaming !== m.id) e.preventDefault() }}
        onClick={() => { if (renaming !== m.id) pick(m.id) }}>
        <Icon model={m} size={28} />
        <span style={itemText}>
          {renaming === m.id
            ? <input data-testid={`${testid}-rename-input`} style={renameInput} value={renameText} autoFocus
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => setRenameText(e.target.value)}
                onKeyDown={(e) => {
                  e.stopPropagation()
                  if (e.key === 'Enter') saveRename(m.id)
                  if (e.key === 'Escape') setRenaming(null)
                }} />
            : <span style={itemName}>{m.label}</span>}
          <span style={itemSub}>{subtitle(m)}</span>
        </span>
        {m.taught && !m.unsaved && (
          showActions
            ? (
              <span style={actionsBox} onClick={(e) => e.stopPropagation()}>
                {renaming === m.id
                  ? <button type="button" data-testid={`${testid}-rename-save`} style={rowBtn}
                      onClick={() => saveRename(m.id)}>Save</button>
                  : <button type="button" data-testid={`${testid}-rename`} style={rowBtn}
                      onClick={() => { setRenaming(m.id); setRenameText(m.name ?? m.label) }}>Rename</button>}
                <button type="button" data-testid={`${testid}-delete`}
                  style={{ ...rowBtn, ...(confirmDelete === m.id ? { color: '#f38ba8', borderColor: '#f38ba8' } : null) }}
                  onClick={() => {
                    if (confirmDelete !== m.id) { setConfirmDelete(m.id); return }
                    setConfirmDelete(null); setActionsFor(null); onDelete(m.id)
                  }}>{confirmDelete === m.id ? 'Delete?' : 'Delete'}</button>
              </span>
            )
            : <button type="button" data-testid={`${testid}-more-${m.id}`} style={moreBtn} title="Rename or delete"
                onClick={(e) => { e.stopPropagation(); setActionsFor(m.id); setConfirmDelete(null) }}>⋯</button>
        )}
      </div>
    )
  }

  return (
    // A press inside the menu must not reach the SubWindow: its mousedown focuses
    // (raises) the window, the caret is laid out again between mousedown and
    // mouseup, and the click on a row never arrives. BackgroundWizard guards the
    // same way.
    <div ref={rootRef} style={{ position: 'relative', minWidth: 0 }} onKeyDown={onKey}
      onMouseDown={(e) => e.stopPropagation()}>
      <button ref={triggerRef} type="button" data-testid={testid} data-value={value}
        aria-haspopup="listbox" aria-expanded={open}
        style={{ ...trigger, ...(open ? triggerOpen : null) }}
        onClick={() => (open ? close() : openMenu())}>
        <Icon model={current} size={18} />
        <span style={triggerLabel}>{current ? current.label : 'Default model'}</span>
        <span style={caret}>▾</span>
      </button>
      {open && (
        <div style={menuFrame}>
          <div ref={listRef} role="listbox" aria-label="Model" style={menuList}
            onScroll={() => { if (card) showCard(card.model) }}>
            {vendored.map(row)}
            {local.length > 0 && (
              <div data-testid={`${testid}-divider`} style={divider} aria-hidden>
                <span style={dividerLabel}>your models</span>
              </div>
            )}
            {local.map(row)}
          </div>
          {card && <ModelCard model={card.model} top={card.top} testid={`${testid}-card`} />}
        </div>
      )}
    </div>
  )
}

function ModelCard({ model: m, top, testid }: { model: ModelInfo; top: number; testid: string }) {
  const rows: [string, string][] = []
  if (m.taught) {
    rows.push(['from', [...(m.chain ?? []), m.unsaved ? 'this fit' : (m.name ?? m.label)].join(' → ')])
    if (m.trained_on) rows.push(['taught on', m.trained_on])
    if (m.created) rows.push(['when', m.created.replace('T', ' ').slice(0, 16)])
    if (m.marks !== undefined) rows.push(['marks', String(m.marks)])
    if (typeof m.original_f1 === 'number') {
      rows.push(['general F1', m.original_f1.toFixed(2)
        + (typeof m.original_f1_base === 'number' ? ` (parent ${m.original_f1_base.toFixed(2)})` : '')])
    }
  } else if (m.version) {
    rows.push(['version', String(m.version)])
  }
  return (
    <div data-testid={testid} role="tooltip" style={{ ...cardStyle, top: Math.max(0, top) }}>
      <div style={{ fontWeight: 600, color: '#cdd6f4' }}>{m.taught ? m.label : (m.description ?? m.label)}</div>
      {rows.map(([k, v]) => (
        <div key={k} style={cardRow}><span style={cardKey}>{k}</span><span>{v}</span></div>
      ))}
      {!m.taught && m.notes && <div style={cardNotes}>{m.notes}</div>}
      {m.unsaved && <div style={{ color: '#f9e2af' }}>Not saved — name it to keep it.</div>}
    </div>
  )
}

const CARD_W = 230

// Trigger, panel and rows follow Dropdown.tsx (itself a copy of MenuBar's).
const trigger: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 6, width: '100%', background: '#11111b', color: '#cdd6f4',
  border: '1px solid #313244', borderRadius: 4, padding: '2px 7px 2px 3px', fontSize: 11, cursor: 'pointer',
  textAlign: 'left', minWidth: 0,
}
const triggerOpen: React.CSSProperties = { borderColor: '#45475a', background: '#181825' }
const triggerLabel: React.CSSProperties = {
  flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
}
const caret: React.CSSProperties = { fontSize: 9, color: '#6c7086', flex: '0 0 auto' }
const menuFrame: React.CSSProperties = {
  position: 'absolute', left: 0, top: 'calc(100% + 3px)', zIndex: 9500, width: '100%', minWidth: 250,
}
const menuList: React.CSSProperties = {
  maxHeight: 300, overflowY: 'auto', background: '#1e1e2e', border: '1px solid #313244', borderRadius: 8,
  padding: 5, boxShadow: '0 10px 28px rgba(0,0,0,0.5)', boxSizing: 'border-box',
}
const item: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 8, width: '100%', boxSizing: 'border-box', textAlign: 'left',
  background: 'transparent', color: '#cdd6f4', borderRadius: 5, padding: '4px 6px', fontSize: 11.5,
  cursor: 'pointer', border: '1px solid transparent',
}
const itemSelected: React.CSSProperties = { background: '#2a2a3c', color: '#89b4fa', fontWeight: 600 }
const itemActive: React.CSSProperties = { background: '#313244' }
const itemUnsaved: React.CSSProperties = { border: '1px dashed #f9e2af' }
const itemText: React.CSSProperties = { display: 'flex', flexDirection: 'column', minWidth: 0, flex: 1 }
const itemName: React.CSSProperties = { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }
const itemSub: React.CSSProperties = {
  fontSize: 10, color: '#7f849c', fontWeight: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
}
const iconBox: React.CSSProperties = {
  flex: '0 0 auto', borderRadius: 4, border: '1px solid #313244', background: '#11111b', display: 'flex',
  alignItems: 'center', justifyContent: 'center', color: '#a6adc8', fontWeight: 700, boxSizing: 'border-box',
}
const iconUnsaved: React.CSSProperties = { borderStyle: 'dashed', borderColor: '#f9e2af' }
const divider: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 6, margin: '5px 4px 3px', borderTop: '1px solid #45475a',
  paddingTop: 3,
}
const dividerLabel: React.CSSProperties = { fontSize: 9.5, color: '#6c7086', textTransform: 'uppercase', letterSpacing: 0.5 }
const moreBtn: React.CSSProperties = {
  flex: '0 0 auto', background: 'transparent', border: 'none', color: '#a6adc8', cursor: 'pointer',
  fontSize: 14, padding: '0 4px', borderRadius: 4,
}
const actionsBox: React.CSSProperties = { display: 'flex', gap: 4, flex: '0 0 auto' }
const rowBtn: React.CSSProperties = {
  background: 'transparent', color: '#cdd6f4', border: '1px solid #555', borderRadius: 4, padding: '1px 6px',
  fontSize: 10.5, cursor: 'pointer',
}
const renameInput: React.CSSProperties = {
  background: '#11111b', color: '#cdd6f4', border: '1px solid #45475a', borderRadius: 4, padding: '1px 4px',
  fontSize: 11, minWidth: 0,
}
const cardStyle: React.CSSProperties = {
  position: 'absolute', left: 'calc(100% + 6px)', width: CARD_W, boxSizing: 'border-box', padding: '6px 8px',
  background: '#1e1e2e', border: '1px solid #45475a', borderRadius: 6, fontSize: 10.5, color: '#bac2de',
  boxShadow: '0 8px 20px rgba(0,0,0,0.55)', display: 'flex', flexDirection: 'column', gap: 2,
  pointerEvents: 'none', zIndex: 9501,
}
const cardRow: React.CSSProperties = { display: 'flex', gap: 6 }
const cardKey: React.CSSProperties = { color: '#6c7086', minWidth: 58 }
const cardNotes: React.CSSProperties = { color: '#7f849c', fontStyle: 'italic' }
