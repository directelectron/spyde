/**
 * CifRecents.tsx — remember picked .cif crystals and offer them as quick-select
 * chips, so you don't have to re-browse the file dialog every time.
 *
 * The list lives in the renderer's localStorage (most-recent-first, capped), so
 * it persists across sessions and needs no backend. Used by both the Orientation
 * and Vector-Orientation wizards.
 */
import React from 'react'

const KEY = 'spyde:cif-recents'
const MAX = 8
const base = (p: string) => p.split(/[/\\]/).pop() || p

function load(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || '[]')
    return Array.isArray(v) ? v.filter((p) => typeof p === 'string') : []
  } catch {
    return []
  }
}

export function useCifRecents() {
  const [recents, setRecents] = React.useState<string[]>(load)
  const remember = React.useCallback((path: string) => {
    setRecents((prev) => {
      const next = [path, ...prev.filter((p) => p !== path)].slice(0, MAX)
      try { localStorage.setItem(KEY, JSON.stringify(next)) } catch { /* private mode */ }
      return next
    })
  }, [])
  return { recents, remember }
}

/** Clickable "Recent" .cif chips. Chips already in ``exclude`` are hidden. */
export function RecentCifs({ recents, exclude = [], onPick }: {
  recents: string[]
  exclude?: string[]
  onPick: (path: string) => void
}) {
  const shown = recents.filter((p) => !exclude.includes(p))
  if (shown.length === 0) return null
  return (
    <div data-testid="cif-recents" style={S.wrap}>
      <span style={S.label}>Recent</span>
      <div style={S.chips}>
        {shown.map((p) => (
          <button key={p} data-testid={`cif-recent-${base(p)}`} title={p}
            style={S.chip} onClick={() => onPick(p)}>{base(p)}</button>
        ))}
      </div>
    </div>
  )
}

const S: Record<string, React.CSSProperties> = {
  wrap: { display: 'flex', flexDirection: 'column', gap: 2 },
  label: { fontSize: 9, color: '#6c7086', textTransform: 'uppercase', letterSpacing: 0.4 },
  chips: { display: 'flex', flexWrap: 'wrap', gap: 4 },
  chip: {
    background: '#181825', color: '#a6adc8', border: '1px solid #313244',
    borderRadius: 10, padding: '2px 8px', fontSize: 10, cursor: 'pointer',
    maxWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
}
