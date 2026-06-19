/**
 * CodPicker.tsx — the "easy CIF" picker: search the Crystallography Open
 * Database for structures matching the sample composition (set in the dock), and
 * pick one from a POPOUT list showing formula / phase / space group / a,b,c,α,β,γ.
 * Choosing downloads the .cif and hands its path to the wizard as a phase.
 *
 * Renders a compact button (sits in a row next to the file picker) + a modal
 * popout for the results. Shared by the Orientation and Vector-Orientation wizards.
 */
import React from 'react'

interface CodResult {
  id: string; formula: string; phase: string; sg: string
  a: number; b: number; c: number
  alpha: number | null; beta: number | null; gamma: number | null
  volume: number | null
}

interface Props {
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onCif: (path: string) => void
}

const fmt = (v: number | null) => (v == null ? '–' : (Math.round(v * 1000) / 1000).toString())

export function CodPicker({ windowId, sendAction, onCif }: Props) {
  const [results, setResults] = React.useState<CodResult[]>([])
  const [busy, setBusy] = React.useState(false)
  const [note, setNote] = React.useState('')
  const [open, setOpen] = React.useState(false)

  React.useEffect(() => {
    const onResults = (e: Event) => {
      const d = (e as CustomEvent).detail as { window_id?: number; results?: CodResult[]; error?: string; elements?: string[] }
      if (d.window_id != null && d.window_id !== windowId) return
      setBusy(false)
      setResults(d.results ?? [])
      setNote(d.error
        ? d.error
        : (d.results?.length ? `${d.results.length} match for ${(d.elements ?? []).join('-')}`
                             : 'No structures found — set a composition in the dock?'))
    }
    const onCifReady = (e: Event) => {
      const d = (e as CustomEvent).detail as { window_id?: number; path?: string }
      if (d.window_id != null && d.window_id !== windowId) return
      if (d.path) onCif(d.path)
    }
    window.addEventListener('spyde:cod_results', onResults)
    window.addEventListener('spyde:cod_cif_ready', onCifReady)
    return () => {
      window.removeEventListener('spyde:cod_results', onResults)
      window.removeEventListener('spyde:cod_cif_ready', onCifReady)
    }
  }, [windowId, onCif])

  const search = () => {
    setBusy(true); setOpen(true); setResults([]); setNote('Searching COD…')
    sendAction('cod_search', {}, windowId)
  }
  const pick = (r: CodResult) => {
    sendAction('cod_pick', { cod_id: r.id, label: `${r.formula} (${r.phase || r.sg})` }, windowId)
    setOpen(false)
  }

  return (
    <>
      <button data-testid="cod-search" style={S.searchBtn} onClick={search}
        title="Search the Crystallography Open Database by the sample composition">
        🔎 Search
      </button>

      {open && (
        <div style={S.backdrop} data-testid="cod-popout" onClick={() => setOpen(false)}>
          <div style={S.modal} onClick={e => e.stopPropagation()}>
            <div style={S.header}>
              <span style={S.title}>Structures from composition</span>
              <button data-testid="cod-close" style={S.x} onClick={() => setOpen(false)}>✕</button>
            </div>
            {(busy || note) && <div style={S.note} data-testid="cod-note">{busy ? 'Searching COD…' : note}</div>}
            {results.length > 0 && (
              <div style={S.list} data-testid="cod-list">
                {results.map(r => (
                  <button key={r.id} data-testid={`cod-row-${r.id}`} style={S.row}
                    title={`COD ${r.id}`} onClick={() => pick(r)}>
                    <div style={S.rowTop}>
                      <span style={S.formula}>{r.formula}</span>
                      {r.phase && <span style={S.phase}>{r.phase}</span>}
                      <span style={S.sg}>{r.sg}</span>
                    </div>
                    <div style={S.cell}>
                      a {fmt(r.a)} · b {fmt(r.b)} · c {fmt(r.c)} Å &nbsp; α {fmt(r.alpha)} · β {fmt(r.beta)} · γ {fmt(r.gamma)}°
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}

const S: Record<string, React.CSSProperties> = {
  searchBtn: {
    flex: 1, background: '#1e1e2e', border: '1px dashed #585b70', color: '#cba6f7',
    cursor: 'pointer', fontSize: 11, fontWeight: 600, padding: '4px 8px', borderRadius: 4,
    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
  },
  backdrop: {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  modal: {
    background: '#181825', border: '1px solid #313244', borderRadius: 10, padding: 14,
    boxShadow: '0 12px 48px rgba(0,0,0,0.6)', width: 380, maxWidth: '92vw',
  },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  title: { color: '#cdd6f4', fontSize: 13, fontWeight: 600 },
  x: { background: 'none', border: 'none', color: '#f38ba8', cursor: 'pointer', fontSize: 14 },
  note: { fontSize: 10, color: '#a6adc8', marginBottom: 6 },
  list: {
    display: 'flex', flexDirection: 'column', gap: 4, maxHeight: '60vh', overflowY: 'auto',
  },
  row: {
    display: 'flex', flexDirection: 'column', gap: 2, alignItems: 'flex-start',
    background: '#1e1e2e', border: '1px solid #313244', borderRadius: 6,
    padding: '6px 9px', cursor: 'pointer', textAlign: 'left', width: '100%',
  },
  rowTop: { display: 'flex', alignItems: 'baseline', gap: 6, flexWrap: 'wrap', width: '100%' },
  formula: { fontSize: 12, fontWeight: 700, color: '#cdd6f4' },
  phase: { fontSize: 10, color: '#a6e3a1' },
  sg: { fontSize: 10, color: '#89b4fa', marginLeft: 'auto' },
  cell: { fontSize: 10, color: '#9399b2', fontFamily: 'monospace' },
}
