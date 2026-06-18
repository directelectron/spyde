import React, { useState } from 'react'
import { SpyDEProvider } from './kernel/SpyDEContext'
import { MDIArea } from './components/MDIArea'
import { PlotControlDock } from './components/PlotControlDock'
import { StatusBar } from './components/StatusBar'

export function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  return (
    <SpyDEProvider>
      <div style={styles.root}>
        <AppBar sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(v => !v)} />
        <div style={styles.body}>
          <MDIArea />
          {sidebarOpen && <PlotControlDock />}
        </div>
        <StatusBar />
      </div>
    </SpyDEProvider>
  )
}

// Right-panel toggle glyph: a framed rect with the side column emphasised when
// the panel is open. Uses currentColor so it inherits the button's hover colour.
function PanelIcon({ open }: { open: boolean }) {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none"
         stroke="currentColor" strokeWidth="1.3">
      <rect x="1.75" y="3" width="12.5" height="10" rx="2" />
      <line x1="10" y1="3" x2="10" y2="13" />
      {open && (
        <rect x="10" y="3" width="4.25" height="10" rx="0"
              fill="currentColor" stroke="none" opacity="0.5" />
      )}
    </svg>
  )
}

// Frameless-window top bar. The whole bar is a drag region (so the OS window can
// be moved); interactive controls opt out with -webkit-app-region: no-drag. Left
// padding clears the macOS traffic-light buttons (titleBarStyle: hiddenInset).
function AppBar({ sidebarOpen, onToggleSidebar }: {
  sidebarOpen: boolean
  onToggleSidebar: () => void
}) {
  const [hover, setHover] = useState(false)
  return (
    <div data-testid="app-bar" style={styles.appBar}>
      <div style={styles.brand}>
        <span style={styles.logoDot} />
        <span style={styles.appTitle}>SpyDE</span>
      </div>
      <button
        data-testid="toggle-sidebar"
        aria-label={sidebarOpen ? 'Hide control panel' : 'Show control panel'}
        title={sidebarOpen ? 'Hide control panel' : 'Show control panel'}
        onClick={onToggleSidebar}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          ...styles.iconBtn,
          color: sidebarOpen ? '#cdd6f4' : '#7f849c',
          background: hover ? '#2a2a3c' : 'transparent',
        }}
      >
        <PanelIcon open={sidebarOpen} />
      </button>
    </div>
  )
}

const drag = { WebkitAppRegion: 'drag' } as React.CSSProperties
const noDrag = { WebkitAppRegion: 'no-drag' } as React.CSSProperties

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
    height: '100%',
    overflow: 'hidden',
  },
  appBar: {
    height: 38,
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingLeft: 84,   // clear the macOS traffic lights
    paddingRight: 8,
    background: 'linear-gradient(#1c1c2b, #181825)',
    borderBottom: '1px solid #2a2a3c',
    ...drag,
  },
  brand: { display: 'flex', alignItems: 'center', gap: 7, ...noDrag },
  logoDot: {
    width: 8, height: 8, borderRadius: '50%',
    background: 'linear-gradient(135deg, #89b4fa, #cba6f7)',
    boxShadow: '0 0 6px rgba(137,180,250,0.6)',
  },
  appTitle: {
    fontSize: 12.5, color: '#cdd6f4', fontWeight: 600, letterSpacing: 0.4,
  },
  iconBtn: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: 28, height: 26, padding: 0,
    border: 'none', borderRadius: 6, cursor: 'pointer',
    transition: 'background 120ms ease, color 120ms ease',
    ...noDrag,
  },
  body: {
    flex: 1,
    display: 'flex',
    flexDirection: 'row',
    minHeight: 0,
  },
}
