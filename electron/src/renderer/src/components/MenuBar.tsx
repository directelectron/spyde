/**
 * MenuBar.tsx — File / Examples / Help dropdown menus drawn IN the custom title
 * bar. The native OS menu bar is hidden on Windows/Linux by
 * titleBarStyle:'hidden' (so we don't get a second bar), so these HTML dropdowns
 * are the only menus there; on macOS they duplicate the system menu, which is
 * harmless and keeps one consistent UI.
 *
 * Each item dispatches a renderer-side action (the same ones the old native menu
 * fired): file dialogs via window.electron.*, examples/actions via sendAction,
 * the Load-Stack dialog + guided tours via the SpyDE context / a callback.
 */
import React, { useEffect, useRef, useState } from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { GUIDES, type Guide } from '@guides/index'

const EXAMPLES = [
  'mgo_nanocrystals',
  'small_ptychography',
  'zrnb_precipitate',
  'pdcusi_insitu',
  'sped_ag',
  'fe_multi_phase_grains',
]

type Item =
  | { label: string; onClick: () => void; disabled?: boolean }
  | { separator: true }

export function MenuBar({ onStartGuide }: { onStartGuide: (g: Guide) => void }) {
  const { sendAction, openStackDialog, openUpdateDialog, openGpuStatusDialog, openGpuHelpDialog, state } = useSpyDE()
  const [open, setOpen] = useState<string | null>(null)
  const barRef = useRef<HTMLDivElement>(null)

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) setOpen(null)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(null) }
    window.addEventListener('mousedown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  const menus: Record<string, Item[]> = {
    File: [
      { label: 'Open…', onClick: () => window.electron.openFile() },
      { label: 'Open Zarr Folder (.zspy)…', onClick: () => window.electron.openZarrFolder() },
      { label: 'Load Stack…', onClick: () => openStackDialog() },
      { separator: true },
      { label: 'Save Signal…', onClick: () => window.electron.saveDialog() },
      { separator: true },
      { label: 'Quit', onClick: () => window.electron.quit() },
    ],
    Examples: EXAMPLES.map((name) => ({
      label: name,
      onClick: () => sendAction('load_example', { name }),
    })),
    Help: [
      ...GUIDES.map((g) => ({
        label: `Guided Tour: ${g.title}`,
        onClick: () => onStartGuide(g),
      })),
      { separator: true },
      {
        label: 'Dask Dashboard ↗',
        disabled: !state.dashboardUrl,
        onClick: () => state.dashboardUrl && window.electron.openExternal(state.dashboardUrl),
      },
      { label: 'GitHub ↗', onClick: () => window.electron.openExternal('https://github.com/cssfrancis/spyde') },
      { separator: true },
      { label: 'Check for Updates…', onClick: () => openUpdateDialog() },
      { label: 'GPU & CUDA', onClick: () => openGpuHelpDialog() },
      { label: 'GPU Status…', onClick: () => openGpuStatusDialog() },
    ],
  }

  return (
    <div ref={barRef} style={styles.bar} data-testid="menu-bar">
      {Object.keys(menus).map((name) => (
        <div key={name} style={{ position: 'relative' }}>
          <button
            data-testid={`menu-${name.toLowerCase()}`}
            style={{
              ...styles.top,
              background: open === name ? '#2a2a3c' : 'transparent',
              color: open === name ? '#cdd6f4' : '#bac2de',
            }}
            onClick={(e) => { e.stopPropagation(); setOpen(open === name ? null : name) }}
            onMouseEnter={() => { if (open) setOpen(name) }}   // hover-switch once a menu is open
          >
            {name}
          </button>
          {open === name && (
            <div style={styles.dropdown} data-testid={`menu-${name.toLowerCase()}-items`}
                 onClick={(e) => e.stopPropagation()}>
              {menus[name].map((it, i) =>
                'separator' in it ? (
                  <div key={`sep${i}`} style={styles.sep} />
                ) : (
                  <button
                    key={it.label}
                    data-testid={`menu-item-${it.label.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`}
                    disabled={it.disabled}
                    style={{ ...styles.item, opacity: it.disabled ? 0.4 : 1 }}
                    onClick={() => { setOpen(null); if (!it.disabled) it.onClick() }}
                    onMouseEnter={(e) => { if (!it.disabled) (e.currentTarget.style.background = '#313244') }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
                  >
                    {it.label}
                  </button>
                ),
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

const noDrag = { WebkitAppRegion: 'no-drag' } as React.CSSProperties

const styles: Record<string, React.CSSProperties> = {
  bar: { display: 'flex', alignItems: 'center', gap: 1, ...noDrag },
  top: {
    border: 'none', borderRadius: 5, cursor: 'pointer',
    padding: '3px 9px', fontSize: 12.5, fontWeight: 500,
    transition: 'background 100ms ease, color 100ms ease',
  },
  dropdown: {
    position: 'absolute', top: 26, left: 0, zIndex: 9200,
    minWidth: 210, background: '#1e1e2e', border: '1px solid #313244',
    borderRadius: 8, padding: 5, boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
  },
  item: {
    display: 'block', width: '100%', textAlign: 'left',
    border: 'none', background: 'transparent', color: '#cdd6f4',
    borderRadius: 5, padding: '6px 10px', fontSize: 12.5, cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  sep: { height: 1, background: '#313244', margin: '5px 4px' },
}
