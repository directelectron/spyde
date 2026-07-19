import React, { useEffect, useState } from 'react'
import { SpyDEProvider, useSpyDE } from './kernel/SpyDEContext'
import { MDIArea } from './components/MDIArea'
import { PlotControlDock } from './components/PlotControlDock'
import { ReportSidebar } from './components/ReportSidebar'
import { ConsoleBar } from './components/ConsoleBar'
import { StatusBar } from './components/StatusBar'
import { LogPanel } from './components/LogPanel'
import { Tour } from './components/Tour'
import { NavShapeGate } from './components/NavShapeGate'
import { StackGate } from './components/StackGate'
import { UpdateGate } from './components/UpdateGate'
import { GpuStatusGate } from './components/GpuStatusGate'
import { GpuHelpGate } from './components/GpuHelpGate'
import { UpdateCard } from './components/UpdateCard'
import { MenuBar } from './components/MenuBar'
import { DownloadToasts } from './components/DownloadToasts'
import { PresentGate } from './components/PresentGate'
import { FirstRunGate } from './components/FirstRunGate'
import { GUIDES, getGuide, type Guide } from '@guides/index'

export function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [reportOpen, setReportOpen] = useState(false)
  const [logOpen, setLogOpen] = useState(false)
  const [tour, setTour] = useState<Guide | null>(null)

  // The Help menu (main process) can launch a guide by id via spyde:start-guide.
  useEffect(() => {
    const dispose = window.electron?.onStartGuide?.((id: string) => {
      const g = getGuide(id)
      if (g) setTour(g)
    })
    return () => dispose?.()
  }, [])

  // "Capture to presentation" (per-window camera button, WindowContent/SubWindow)
  // fires this so the report dock becomes visible even if it was closed — the
  // user should SEE the slide they just captured, not wonder where it went.
  useEffect(() => {
    const onOpen = () => setReportOpen(true)
    window.addEventListener('spyde:report_open_dock', onOpen)
    return () => window.removeEventListener('spyde:report_open_dock', onOpen)
  }, [])

  return (
    <SpyDEProvider>
      <div style={styles.root}>
        <AppBar
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen(v => !v)}
          reportOpen={reportOpen}
          onToggleReport={() => setReportOpen(v => !v)}
          onStartGuide={(g) => setTour(g)}
        />
        <div style={styles.body}>
          <MDIArea />
          {sidebarOpen && <PlotControlDock />}
          {reportOpen && <ReportSidebar />}
        </div>
        <LogPanel open={logOpen} onClose={() => setLogOpen(false)} />
        <ConsoleBar />
        <StatusBar logOpen={logOpen} onToggleLog={() => setLogOpen(v => !v)} />
      </div>
      {/* Examples-menu download progress cards (bottom-right, above the bar). */}
      <DownloadToasts />
      {/* "New version available" card (bottom-left, above the bar) — mirrors the
          DownloadToasts corner. Shows the whole updater lifecycle incl. errors. */}
      <UpdateCard />
      {tour && <Tour guide={tour} onClose={() => setTour(null)} />}
      {/* Present mode (Phase 6): renders the open Report as full-screen slides,
          launched from the Report sidebar's "Present" button (via the
          spyde:report_present event). Owns the slide-index persistence + the
          go-live excursion handoff. */}
      <PresentGate onStartGuide={(g) => setTour(g)} />
      {/* Scan-shape/step-size confirm dialog — reads the pending prompt from the
          SpyDE context (so injected test messages reach it too). */}
      <NavShapeGate />
      {/* Load Stack dialog — reorderable list of datasets to combine into one
          5D stack; opened from File → Load Stack…. */}
      <StackGate />
      {/* Help → Check for Updates… / GPU Status… / GPU & CUDA */}
      <UpdateGate />
      <GpuStatusGate />
      <GpuHelpGate />
      {/* First-run welcome walkthrough (docs overhaul Phase 4): auto-opens the
          "First Steps" tour exactly once, tracked by the tutorial_seen settings
          flag. Always re-launchable afterwards from Help → First Steps. */}
      <FirstRunGate onAutoOpen={(g) => setTour((cur) => cur ?? g)} />
    </SpyDEProvider>
  )
}

// A "?" button that opens a small menu of available guided tours.
function HelpButton({ onStartGuide }: { onStartGuide: (g: Guide) => void }) {
  const [open, setOpen] = useState(false)
  const [hover, setHover] = useState(false)
  // Close the menu on any outside click.
  useEffect(() => {
    if (!open) return
    const close = () => setOpen(false)
    window.addEventListener('click', close)
    return () => window.removeEventListener('click', close)
  }, [open])
  return (
    <div style={{ position: 'relative', ...noDrag }}>
      <button
        data-testid="help-button"
        aria-label="Guided tours"
        title="Guided tours"
        onClick={(e) => { e.stopPropagation(); setOpen(v => !v) }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          ...styles.iconBtn,
          color: open ? '#cdd6f4' : '#7f849c',
          background: hover || open ? '#2a2a3c' : 'transparent',
          fontSize: 14, fontWeight: 700,
        }}
      >
        ?
      </button>
      {open && (
        <div data-testid="help-menu" style={styles.helpMenu} onClick={(e) => e.stopPropagation()}>
          <div style={styles.helpMenuTitle}>Guided Tours</div>
          {GUIDES.map((g) => (
            <button
              key={g.id}
              data-testid={`help-guide-${g.id}`}
              style={styles.helpMenuItem}
              onClick={() => { setOpen(false); onStartGuide(g) }}
            >
              <div style={styles.helpMenuItemTitle}>{g.title}</div>
              <div style={styles.helpMenuItemSummary}>{g.summary}</div>
            </button>
          ))}
        </div>
      )}
    </div>
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

// Report-panel toggle glyph: a document with lines (a report page).
function ReportIcon({ open }: { open: boolean }) {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none"
         stroke="currentColor" strokeWidth="1.3">
      <rect x="3" y="1.75" width="10" height="12.5" rx="1.5"
            fill={open ? 'currentColor' : 'none'} opacity={open ? 0.2 : 1} />
      <line x1="5.25" y1="5" x2="10.75" y2="5" />
      <line x1="5.25" y1="8" x2="10.75" y2="8" />
      <line x1="5.25" y1="11" x2="8.75" y2="11" />
    </svg>
  )
}

// Frameless-window top bar. The whole bar is a drag region (so the OS window can
// be moved); interactive controls opt out with -webkit-app-region: no-drag. Left
// padding clears the macOS traffic-light buttons (titleBarStyle: hiddenInset).
function AppBar({ sidebarOpen, onToggleSidebar, reportOpen, onToggleReport, onStartGuide }: {
  sidebarOpen: boolean
  onToggleSidebar: () => void
  reportOpen: boolean
  onToggleReport: () => void
  onStartGuide: (g: Guide) => void
}) {
  const [hover, setHover] = useState(false)
  const isMac = window.electron?.platform === 'darwin'
  return (
    <div
      data-testid="app-bar"
      style={{
        ...styles.appBar,
        // macOS: clear the traffic-light buttons (top-left). Windows/Linux: the
        // native min/max/close are an overlay on the RIGHT (titleBarOverlay), so
        // pad the right instead and keep the logo flush left.
        paddingLeft: isMac ? 84 : 12,
        paddingRight: isMac ? 8 : 146,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, ...noDrag }}>
        <div style={styles.brand}>
          <span style={styles.logoDot} />
          <span style={styles.appTitle}>SpyDE</span>
        </div>
        <MenuBar onStartGuide={onStartGuide} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 2, ...noDrag }}>
        <HelpButton onStartGuide={onStartGuide} />
        <ReportToggle open={reportOpen} onToggle={onToggleReport} />
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
    </div>
  )
}

// Report-panel toggle button (its own hover state, next to the control-panel
// toggle). Opening/closing the report sidebar is pure renderer UI state — no
// backend action fires here, so StrictMode's double-mount can't double-send.
function ReportToggle({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const [hover, setHover] = useState(false)
  return (
    <button
      data-testid="toggle-report"
      aria-label={open ? 'Hide report' : 'Show report'}
      title={open ? 'Hide report' : 'Show report'}
      onClick={onToggle}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        ...styles.iconBtn,
        color: open ? '#cdd6f4' : '#7f849c',
        background: hover ? '#2a2a3c' : 'transparent',
      }}
    >
      <ReportIcon open={open} />
    </button>
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
    // paddingLeft/Right set inline per-platform (traffic lights vs overlay).
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
  helpMenu: {
    position: 'absolute', top: 32, right: 0, zIndex: 9100,
    width: 280, background: '#1e1e2e', border: '1px solid #313244',
    borderRadius: 8, padding: 6, boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
  },
  helpMenuTitle: {
    fontSize: 10.5, color: '#6c7086', letterSpacing: 0.6, textTransform: 'uppercase',
    padding: '4px 8px 6px',
  },
  helpMenuItem: {
    display: 'block', width: '100%', textAlign: 'left',
    background: 'transparent', border: 'none', borderRadius: 6,
    padding: '8px', cursor: 'pointer', color: '#cdd6f4',
  },
  helpMenuItemTitle: { fontSize: 13, fontWeight: 600, color: '#cdd6f4' },
  helpMenuItemSummary: { fontSize: 11, color: '#7f849c', marginTop: 2, lineHeight: 1.35 },
}
