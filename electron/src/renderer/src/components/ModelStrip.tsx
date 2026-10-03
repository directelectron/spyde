/**
 * ModelStrip.tsx — pick a detector model from a strip of icon tiles.
 *
 * Vendored models (bundled with SpyDE or downloaded) come first, then a divider,
 * then the models the user taught. A tile's picture is the model's preferred
 * input — the disk it would most like to see — computed by the backend
 * (spyde/models/adapt.py write_icon); a model without one shows its initials.
 * Hovering or focusing a tile shows its card (name, parent chain, the dataset
 * it was taught on, when, how many marks, how general it still is).
 *
 * Keyboard: Tab into the strip, ←/→ (Home/End) move the selection.
 * Testing: tiles are `${testid}-tile-${id}`, the divider `${testid}-divider`,
 * the card `${testid}-card`.
 */
import React from 'react'

export interface ModelInfo {
  id: string
  label: string
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
  return (words.slice(0, 2).map((w) => w[0]!.toUpperCase()).join('') || '?')
}

const TILE = 40
const CARD_W = 220

export function ModelStrip({ models, value, onChange, testid }: {
  models: readonly ModelInfo[]
  value: string
  onChange: (id: string) => void
  testid: string
}) {
  const [card, setCard] = React.useState<{ model: ModelInfo; left: number; width: number } | null>(null)
  const rootRef = React.useRef<HTMLDivElement>(null)
  const stripRef = React.useRef<HTMLDivElement>(null)
  const vendored = models.filter((m) => m.group !== 'local')
  const local = models.filter((m) => m.group === 'local')
  const order = [...vendored, ...local]

  const showCard = (model: ModelInfo, el: HTMLElement) => {
    const root = rootRef.current?.getBoundingClientRect()
    const tile = el.getBoundingClientRect()
    setCard({ model, left: root ? tile.left - root.left : 0, width: root ? root.width : CARD_W })
  }

  // Keep the selected tile in view (a newly taught model lands at the end).
  React.useEffect(() => {
    const el = stripRef.current?.querySelector<HTMLElement>(`[data-model-id="${CSS.escape(value)}"]`)
    el?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [value, models])

  const onKey = (e: React.KeyboardEvent) => {
    const at = order.findIndex((m) => m.id === value)
    let next = at
    if (e.key === 'ArrowRight') next = Math.min(order.length - 1, at + 1)
    else if (e.key === 'ArrowLeft') next = Math.max(0, at - 1)
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = order.length - 1
    else return
    e.preventDefault()
    if (next >= 0 && next !== at) {
      onChange(order[next]!.id)
      const el = stripRef.current?.querySelector<HTMLElement>(`[data-model-id="${CSS.escape(order[next]!.id)}"]`)
      el?.focus()
    }
  }

  const tile = (m: ModelInfo) => {
    const selected = m.id === value
    return (
      <button key={m.id} type="button" role="option" aria-selected={selected}
        aria-label={m.label} data-model-id={m.id} data-testid={`${testid}-tile-${m.id}`}
        tabIndex={selected ? 0 : -1}
        style={{ ...tileStyle, ...(selected ? tileSelected : null), ...(m.unsaved ? tileUnsaved : null) }}
        onClick={() => onChange(m.id)}
        onMouseEnter={(e) => showCard(m, e.currentTarget)}
        onMouseLeave={() => setCard(null)}
        onFocus={(e) => showCard(m, e.currentTarget)}
        onBlur={() => setCard(null)}>
        {m.icon
          ? <img src={m.icon} alt="" style={iconStyle} draggable={false} />
          : <span style={glyphStyle}>{initials(m.label)}</span>}
      </button>
    )
  }

  const selected = order.find((m) => m.id === value)
  return (
    <div ref={rootRef} style={{ position: 'relative', minWidth: 0 }}>
      <div ref={stripRef} role="listbox" aria-label="Model" data-testid={testid}
        data-value={value} onKeyDown={onKey} style={stripStyle}>
        {vendored.map(tile)}
        {local.length > 0 && <div data-testid={`${testid}-divider`} style={dividerStyle} aria-hidden />}
        {local.map(tile)}
      </div>
      <div data-testid={`${testid}-selected`} style={captionStyle} title={selected?.label}>
        {selected ? selected.label : 'Default model'}
      </div>
      {card && <ModelCard model={card.model} left={Math.max(0, Math.min(card.left - 40, card.width - CARD_W))}
        testid={`${testid}-card`} />}
    </div>
  )
}

function ModelCard({ model: m, left, testid }: { model: ModelInfo; left: number; testid: string }) {
  const rows: [string, string][] = []
  if (m.taught) {
    rows.push(['from', [...(m.chain ?? []), m.unsaved ? 'this fit' : (m.name ?? m.label)].join(' → ')])
    if (m.trained_on) rows.push(['taught on', m.trained_on])
    if (m.created) rows.push(['when', m.created.replace('T', ' ').slice(0, 16)])
    if (m.marks !== undefined) rows.push(['marks', String(m.marks)])
    if (typeof m.original_f1 === 'number') {
      rows.push(['general F1', `${m.original_f1.toFixed(2)}`
        + (typeof m.original_f1_base === 'number' ? ` (parent ${m.original_f1_base.toFixed(2)})` : '')])
    }
  } else if (m.version) {
    rows.push(['version', String(m.version)])
  }
  return (
    <div data-testid={testid} role="tooltip"
      style={{ ...cardStyle, left }}>
      <div style={{ fontWeight: 600, color: '#cdd6f4' }}>
        {m.label}{m.taught ? '' : '  · vendored'}
      </div>
      {rows.map(([k, v]) => (
        <div key={k} style={cardRow}><span style={cardKey}>{k}</span><span>{v}</span></div>
      ))}
      {!m.taught && m.notes && <div style={cardNotes}>{m.notes}</div>}
      {m.unsaved && <div style={{ color: '#f9e2af' }}>Not saved — name it to keep it.</div>}
    </div>
  )
}

const stripStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 4, overflowX: 'auto', overflowY: 'hidden',
  padding: '4px 4px 6px', scrollbarWidth: 'thin', minWidth: 0,
}
const tileStyle: React.CSSProperties = {
  flex: '0 0 auto', width: TILE, height: TILE, padding: 0, borderRadius: 6, cursor: 'pointer',
  background: '#11111b', border: '1px solid #313244', overflow: 'hidden', position: 'relative',
  display: 'flex', alignItems: 'center', justifyContent: 'center', outlineOffset: 1,
}
// An outline, not the border: an unsaved tile's dashed border must stay visible
// while it is the selected one.
const tileSelected: React.CSSProperties = { outline: '2px solid #89b4fa', outlineOffset: 1 }
const tileUnsaved: React.CSSProperties = { borderStyle: 'dashed', borderColor: '#f9e2af' }
const iconStyle: React.CSSProperties = { width: '100%', height: '100%', objectFit: 'cover', display: 'block' }
const glyphStyle: React.CSSProperties = { fontSize: 13, fontWeight: 700, color: '#a6adc8', letterSpacing: 0.5 }
const dividerStyle: React.CSSProperties = {
  flex: '0 0 auto', width: 1, alignSelf: 'stretch', margin: '2px 4px', background: '#585b70',
}
const captionStyle: React.CSSProperties = {
  fontSize: 10, color: '#a6adc8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
}
const cardStyle: React.CSSProperties = {
  position: 'absolute', top: TILE + 12, zIndex: 30, width: CARD_W, boxSizing: 'border-box', padding: '6px 8px',
  background: '#1e1e2e', border: '1px solid #45475a', borderRadius: 6, fontSize: 10.5,
  color: '#bac2de', boxShadow: '0 8px 20px rgba(0,0,0,0.55)', display: 'flex',
  flexDirection: 'column', gap: 2, pointerEvents: 'none',
}
const cardRow: React.CSSProperties = { display: 'flex', gap: 6 }
const cardKey: React.CSSProperties = { color: '#6c7086', minWidth: 58 }
const cardNotes: React.CSSProperties = { color: '#7f849c', fontStyle: 'italic' }
