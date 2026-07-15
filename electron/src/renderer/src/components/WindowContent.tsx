import React, { useEffect, useMemo, useRef, useState } from 'react'
import type { SpyDEWindow, SpyDEFigure } from '../kernel/SpyDEContext'
import { useSpyDE } from '../kernel/SpyDEContext'
import {
  NAVIGATOR_DRAG_MIME, WINDOW_DRAG_MIME, FIGURE_DRAG_MIME,
} from '../kernel/dnd'

// Resolve a source window id from a FIGURE_DRAG_MIME or WINDOW_DRAG_MIME drop
// (the window pill stamps both; payload is only readable on drop).
function sourceWindowIdFromDrop(dt: DataTransfer): number | null {
  const fig = dt.getData(FIGURE_DRAG_MIME)
  if (fig) {
    try {
      const { windowId } = JSON.parse(fig) as { windowId?: number }
      if (typeof windowId === 'number') return windowId
    } catch { /* malformed */ }
  }
  const win = dt.getData(WINDOW_DRAG_MIME)
  if (win) {
    const n = parseInt(win, 10)
    if (Number.isFinite(n)) return n
  }
  return null
}

interface Props {
  win: SpyDEWindow
  iframeRefs: React.MutableRefObject<Map<string, HTMLIFrameElement>>
  replayState: (figId: string) => void
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
}

// The reserved view_label of the backend-built side-by-side comparison figure
// (2-D navigators, tiled) and the stacked comparison figure (1-D/movie
// navigators, rows) — see navigator_views.py TILED_LABEL / STACKED_LABEL.
const TILED = '__tiled__'
const STACKED = '__stacked__'
const STRAIN_LABEL: Record<string, string> = { exx: 'εxx', eyy: 'εyy', exy: 'εxy', omega: 'ω' }

// A window's content area: the unified "view" selector + the figure it shows.
//
// A window may hold several NAMED views of one navigation field — strain
// εxx/εyy/εxy, virtual images, the IPF map — each emitted as a figure tagged
// with a `viewLabel` (chip text). On top of that an IPF window carries a second
// `view:"3d"` explorer figure (the 2D⇄3D toggle) and the X/Y/Z direction
// selector.
//
//   • click a chip       → show just that view
//   • ⌘/Ctrl-click chips  → COMPARE: the backend rebuilds ONE anyplotlib figure
//                           with the selected views as side-by-side axes
//                           (shared pan/zoom + a linked crosshair on each)
//   • 2D / 3D             → swap the IPF map for its 3-D sphere explorer
//   • X / Y / Z           → re-colour the IPF by sample direction
//
// All iframes stay MOUNTED; only the active one is shown (instant switch). A
// ResizeObserver keeps the visible figure sized to its box (the single sizing
// authority — handles window resize and the view-bar height).
export function WindowContent({ win, iframeRefs, replayState, sendAction }: Props) {
  const id = String(win.windowId)
  const figs = win.figures
  const { state, dragKind } = useSpyDE()

  // ── MDI overlay drop (Report Builder Phase 2) ──────────────────────────────
  // While a window/figure pill is in flight (dragKind==='window'), the figure
  // iframe (out-of-process, swallows DnD) is covered by a transparent shield with
  // a centered "Overlay images" zone. Dropping another window's pill here asks to
  // layer that source's image over this window's image (overlay_add). Self-drops
  // are ignored; the backend validates shape compatibility and reports errors via
  // status (no pre-validation here). The navigator titlebar drop (acceptSignalDrop
  // in SubWindow) and the MDI-area drops are untouched — this shield sits only over
  // the CONTENT box, not the titlebar.
  const [overlayHover, setOverlayHover] = useState(false)
  const [overlayConfirm, setOverlayConfirm] = useState<number | null>(null)
  // Clear any transient hover once the drag ends.
  useEffect(() => { if (dragKind == null) setOverlayHover(false) }, [dragKind])

  const onOverlayDragOver = (e: React.DragEvent) => {
    const types = e.dataTransfer.types
    if (!types.includes(WINDOW_DRAG_MIME) && !types.includes(FIGURE_DRAG_MIME)) return
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
    setOverlayHover(true)
  }
  const onOverlayDragLeave = (e: React.DragEvent) => {
    if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
      setOverlayHover(false)
    }
  }
  const onOverlayDrop = (e: React.DragEvent) => {
    const src = sourceWindowIdFromDrop(e.dataTransfer)
    setOverlayHover(false)
    if (src == null) return
    e.preventDefault()
    e.stopPropagation()
    if (src === win.windowId) return   // ignore self-drops
    setOverlayConfirm(src)             // ask before layering
  }
  const confirmOverlay = () => {
    if (overlayConfirm != null) {
      sendAction('overlay_add',
        { window_id: win.windowId, source_window_id: overlayConfirm }, win.windowId)
    }
    setOverlayConfirm(null)
  }

  // Navigator chip strip: a navigator window whose tree carries ≥2 NAMED
  // navigators (base sum, vector count map, a dropped-in signal, …) lists them
  // at the top — click switches the live navigator in place; SHIFT-click
  // selects several, which the backend tiles side by side (linked pan/zoom +
  // a duplicated crosshair per panel driving the real selector).
  const navOpts = state.navigatorOptions.get(win.windowId)
  const navNames = navOpts?.names ?? []
  const hasNavChips = navNames.length >= 2
  const [navSel, setNavSel] = useState<string[]>([])
  useEffect(() => {
    // Keep the selection a valid non-empty subset as navigators come and go.
    setNavSel(prev => {
      const valid = prev.filter(n => navNames.includes(n))
      if (valid.length) return valid.length === prev.length ? prev : valid
      const seed = navOpts?.current && navNames.includes(navOpts.current)
        ? navOpts.current : navNames[0]
      return seed ? [seed] : []
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [navNames.join('|')])
  const navMulti = navSel.length >= 2

  const onNavChip = (name: string, e: React.MouseEvent) => {
    const tile = e.shiftKey || e.metaKey || e.ctrlKey
    setNavSel(prev => {
      let next: string[]
      if (!tile) next = [name]
      else if (prev.includes(name)) {
        const rest = prev.filter(n => n !== name)
        next = rest.length ? rest : prev
      } else next = navNames.filter(n => prev.includes(n) || n === name)
      if (next !== prev) sendAction('select_navigator', { names: next }, win.windowId)
      return next
    })
  }

  const fig3d = useMemo(() => figs.find(f => f.view === '3d'), [figs])
  const has3d = !!fig3d
  const figDensity = useMemo(() => figs.find(f => f.view === 'density'), [figs])
  const hasDensity = !!figDensity
  // The IPF colour-key triangle legend — a native anyplotlib figure pinned in
  // the corner of the 2-D map (not a switchable view).
  const figIpfKey = useMemo(() => figs.find(f => f.view === 'ipf_key'), [figs])
  const tiledFig = useMemo(() => figs.find(f => f.viewLabel === TILED), [figs])
  const stackedFig = useMemo(() => figs.find(f => f.viewLabel === STACKED), [figs])
  // Unique chip labels in stable first-seen order (the tiled figure is not a chip).
  const labels = useMemo(() => {
    const seen: string[] = []
    for (const f of figs) if (f.viewLabel && f.viewLabel !== TILED && !seen.includes(f.viewLabel)) seen.push(f.viewLabel)
    return seen
  }, [figs])
  const hasChips = labels.length >= 2

  // Strain window: one figure carrying the component list → an εxx/εyy/εxy/ω
  // toggle that swaps the shown component in place (strain_set_component).
  const strainFig = useMemo(() => figs.find(f => f.strainComponents && f.strainComponents.length), [figs])
  const strainComponents = strainFig?.strainComponents

  const [mode, setMode] = useState<'2d' | '3d' | 'density'>('2d')
  const [dir, setDir] = useState<'x' | 'y' | 'z'>('z')
  const [strainComp, setStrainComp] = useState('exx')

  // Fall back to the 2-D map if the active mode's figure disappears (e.g. a
  // result re-run before the 3-D / density figure re-arrives).
  useEffect(() => {
    if ((mode === '3d' && !has3d) || (mode === 'density' && !hasDensity)) setMode('2d')
  }, [mode, has3d, hasDensity])
  const [selected, setSelected] = useState<string[]>([])

  // Keep `selected` a non-empty subset of the available labels (repairs after a
  // result re-run swaps the figures, and seeds the default to the first view).
  useEffect(() => {
    if (!hasChips) return
    setSelected(prev => {
      const valid = prev.filter(l => labels.includes(l))
      return valid.length ? valid : [labels[0]]
    })
  }, [labels, hasChips])

  const multi = selected.length >= 2

  // When ≥2 views are selected, ask the backend to (re)build the side-by-side
  // comparison figure. Keyed on the selection so it fires once per change (not
  // on every render — sendAction is not referentially stable).
  const selKey = selected.join('|')
  useEffect(() => {
    if (selected.length >= 2) sendAction('tile_views', { labels: selected }, win.windowId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selKey, win.windowId])

  const onChip = (label: string, e: React.MouseEvent) => {
    const tile = e.metaKey || e.ctrlKey
    setSelected(prev => {
      if (!tile) return [label]
      if (prev.includes(label)) {
        const next = prev.filter(l => l !== label)
        return next.length ? next : prev                       // keep ≥1 selected
      }
      return labels.filter(l => prev.includes(l) || l === label) // preserve chip order
    })
  }

  // The single figure to show right now.
  const shownFig = useMemo<SpyDEFigure | null>(() => {
    if (has3d && mode === '3d' && fig3d) return fig3d
    if (hasDensity && mode === 'density' && figDensity) return figDensity
    if (multi && tiledFig) return tiledFig                     // anyplotlib N-axis compare
    if (navMulti && tiledFig) return tiledFig                  // tiled navigators (2-D)
    if (navMulti && stackedFig) return stackedFig              // stacked navigators (1-D/movie)
    if (hasChips) return [...figs].reverse().find(f => f.viewLabel === selected[0]) ?? null
    return figs.find(f => f.view !== '3d' && f.view !== 'density' && f.view !== 'ipf_key'
      && f.viewLabel !== TILED && f.viewLabel !== STACKED) ?? figs[0] ?? null
  }, [has3d, mode, fig3d, hasDensity, figDensity, multi, navMulti, tiledFig, stackedFig, hasChips, selected, figs])

  const shownId = shownFig?.figId

  // Resize the visible figure to its real box (window resize / bar height / the
  // view swap that revealed a previously-hidden iframe).
  const boxRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const fit = () => {
      if (!shownId) return
      const el = iframeRefs.current.get(shownId)
      if (el && el.clientWidth && el.clientHeight)
        window.electron.resizeFigure(shownId, Math.max(80, el.clientWidth), Math.max(80, el.clientHeight))
    }
    let raf = requestAnimationFrame(fit)
    const ro = new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(fit) })
    if (boxRef.current) ro.observe(boxRef.current)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [shownId, iframeRefs])

  const showBar = hasChips || has3d || !!strainComponents || hasNavChips

  return (
    <div style={styles.root}>
      {showBar && (
        <div style={styles.bar} data-testid={`view-bar-${id}`}>
          {hasNavChips && (
            <div style={styles.chips} data-testid={`nav-chips-${id}`}>
              {navNames.map(name => (
                <button
                  key={name}
                  data-testid={`nav-chip-${name}-${id}`}
                  onClick={(e) => onNavChip(name, e)}
                  title="Click to show · Shift-click to tile · Drag out to make its own dataset"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData(NAVIGATOR_DRAG_MIME,
                      JSON.stringify({ windowId: win.windowId, name }))
                    e.dataTransfer.effectAllowed = 'copy'
                  }}
                  style={navSel.includes(name) ? styles.chipActive : styles.chip}
                >{name}</button>
              ))}
            </div>
          )}
          {hasChips && (
            <div style={styles.chips} data-testid={`view-chips-${id}`}>
              {labels.map(label => (
                <button
                  key={label}
                  data-testid={`view-chip-${label}-${id}`}
                  onClick={(e) => onChip(label, e)}
                  title="Click to show · ⌘-click to compare side by side"
                  style={selected.includes(label) ? styles.chipActive : styles.chip}
                >{label}</button>
              ))}
            </div>
          )}
          <div style={{ flex: 1 }} />
          {(has3d || hasDensity) && (
            <div style={styles.group} data-testid={`ipf-view-toggle-${id}`}>
              {([['2d', '2D'], ...(has3d ? [['3d', '3D']] : []),
                 ...(hasDensity ? [['density', 'PDF']] : [])] as const).map(([m, lbl]) => (
                <button key={m} data-testid={`ipf-view-${m}-${id}`}
                  onClick={() => setMode(m as '2d' | '3d' | 'density')}
                  style={m === mode ? styles.btnActive : styles.btn}>{lbl}</button>
              ))}
              <span style={{ width: 6 }} />
              {(['x', 'y', 'z'] as const).map(d => (
                <button key={d} data-testid={`ipf-dir-${d}-${id}`}
                  onClick={() => { setDir(d); sendAction('ipf_set_direction', { direction: d }, win.windowId) }}
                  style={d === dir ? styles.btnActive : styles.btn}>{d.toUpperCase()}</button>
              ))}
            </div>
          )}
          {strainComponents && (
            // The strain MAP window's component toggle (εxx/εyy/εxy/ω). Reference
            // method, spot selection, match radius, and Submit live in the Strain
            // caret (StrainWizard) on the source pattern, not here.
            <div style={styles.group} data-testid={`strain-toggle-${id}`}>
              {strainComponents.map(c => (
                <button key={c} data-testid={`strain-comp-${c}-${id}`}
                  onClick={() => { setStrainComp(c); sendAction('strain_set_component', { component: c }, win.windowId) }}
                  style={c === strainComp ? styles.btnActive : styles.btn}>{STRAIN_LABEL[c] ?? c}</button>
              ))}
            </div>
          )}
        </div>
      )}

      <div ref={boxRef} data-testid={`figure-box-${id}`} style={styles.box}>
        {figs.filter(f => f.view !== 'ipf_key').map(fig => (
          <iframe
            key={fig.figId}
            ref={el => {
              if (el) iframeRefs.current.set(fig.figId, el)
              else iframeRefs.current.delete(fig.figId)
            }}
            src={fig.filePath ?? undefined}
            // Replay any state (image data, selectors) that arrived before this
            // iframe was listening — fixes the black-image race. Size to the
            // iframe's actual box on load (hidden iframes get re-fit when shown).
            onLoad={(e) => {
              replayState(fig.figId)
              const el = e.currentTarget
              window.electron.resizeFigure(fig.figId, Math.max(80, el.clientWidth), Math.max(80, el.clientHeight))
            }}
            style={{ ...styles.frame, display: fig.figId === shownId ? 'block' : 'none' }}
            title={fig.title}
            data-testid={`figure-${fig.figId}`}
          />
        ))}
        {/* IPF colour-key triangle legend — a native anyplotlib figure pinned in
            the corner of the 2-D map (the stereographic fundamental-sector key
            matplotlib/pyxem show), only over the RGB map (mode==='2d'). */}
        {figIpfKey && mode === '2d' && (
          <iframe
            key={figIpfKey.figId}
            ref={el => {
              if (el) iframeRefs.current.set(figIpfKey.figId, el)
              else iframeRefs.current.delete(figIpfKey.figId)
            }}
            src={figIpfKey.filePath ?? undefined}
            onLoad={(e) => {
              replayState(figIpfKey.figId)
              const el = e.currentTarget
              window.electron.resizeFigure(figIpfKey.figId, Math.max(80, el.clientWidth), Math.max(80, el.clientHeight))
            }}
            style={styles.ipfKey}
            title={figIpfKey.title}
            data-testid={`ipf-key-${id}`}
          />
        )}

        {/* MDI overlay-drop shield — mounted ONLY while a window/figure pill is
            being dragged, so it never interferes otherwise. Catches the DnD the
            iframe would swallow; a centered zone reads "Overlay images". */}
        {dragKind === 'window' && overlayConfirm == null && (
          <div
            data-testid={`overlay-drop-shield-${id}`}
            style={styles.overlayShield}
            onDragOver={onOverlayDragOver}
            onDragLeave={onOverlayDragLeave}
            onDrop={onOverlayDrop}
          >
            <div style={{ ...styles.overlayZone, ...(overlayHover ? styles.overlayZoneHot : {}) }}>
              <span style={styles.overlayZoneLabel}>Overlay images</span>
            </div>
          </div>
        )}

        {/* Confirm popover — a small "Overlay onto this image?" before layering. */}
        {overlayConfirm != null && (
          <div style={styles.overlayConfirm} data-testid={`overlay-confirm-${id}`} role="dialog">
            <div style={styles.overlayConfirmText}>Overlay onto this image?</div>
            <div style={styles.overlayConfirmRow}>
              <button
                data-testid={`overlay-confirm-ok-${id}`}
                style={styles.overlayConfirmOk}
                onClick={confirmOverlay}
              >Overlay</button>
              <button
                data-testid={`overlay-confirm-cancel-${id}`}
                style={styles.overlayConfirmCancel}
                onClick={() => setOverlayConfirm(null)}
              >Cancel</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  root: { display: 'flex', flexDirection: 'column', width: '100%', height: '100%' },
  bar: {
    display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0,
    padding: '3px 6px', background: '#181825', borderBottom: '1px solid #313244',
    minHeight: 24,
  },
  chips: { display: 'flex', gap: 3, flexWrap: 'wrap', minWidth: 0 },
  group: {
    display: 'flex', gap: 2, background: 'rgba(24,24,37,0.85)',
    border: '1px solid #313244', borderRadius: 6, padding: 2,
  },
  box: { flex: 1, minHeight: 0, position: 'relative' },
  ipfKey: {
    position: 'absolute', right: 6, bottom: 6, width: 132, height: 120,
    border: 'none', zIndex: 4,
    background: 'rgba(24,24,37,0.72)', borderRadius: 6,
  },
  // MDI overlay-drop shield (over the figure iframe, only during a window drag).
  overlayShield: {
    position: 'absolute', inset: 0, zIndex: 6,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: 16, boxSizing: 'border-box',
  },
  overlayZone: {
    width: '78%', height: '62%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    border: '2px dashed rgba(137,180,250,0.45)', borderRadius: 10,
    background: 'rgba(24,24,37,0.28)',
    transition: 'background 70ms, border-color 70ms',
  },
  overlayZoneHot: {
    borderColor: '#89b4fa', background: 'rgba(137,180,250,0.24)',
  },
  overlayZoneLabel: {
    fontSize: 12, fontWeight: 600, color: '#cdd6f4',
    textShadow: '0 1px 3px rgba(0,0,0,0.85)',
  },
  overlayConfirm: {
    position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)',
    zIndex: 7, minWidth: 190,
    background: 'rgba(24,24,37,0.98)', border: '1px solid #89b4fa',
    borderRadius: 8, padding: '10px 12px',
    boxShadow: '0 6px 22px rgba(0,0,0,0.55)', textAlign: 'center',
  },
  overlayConfirmText: { fontSize: 12, color: '#cdd6f4', marginBottom: 8 },
  overlayConfirmRow: { display: 'flex', gap: 6, justifyContent: 'center' },
  overlayConfirmOk: {
    background: '#89b4fa', color: '#11111b', border: 'none',
    borderRadius: 5, padding: '4px 12px', fontSize: 11, cursor: 'pointer', fontWeight: 600,
  },
  overlayConfirmCancel: {
    background: '#1e1e2e', color: '#a6adc8', border: '1px solid #45475a',
    borderRadius: 5, padding: '4px 12px', fontSize: 11, cursor: 'pointer',
  },
  frame: {
    position: 'absolute', inset: 0, width: '100%', height: '100%',
    border: 'none', minWidth: 0, minHeight: 0,
  },
  chip: {
    background: '#1e1e2e', border: '1px solid #313244', color: '#a6adc8',
    cursor: 'pointer', fontSize: 10, fontWeight: 600, padding: '2px 9px', borderRadius: 10,
  },
  chipActive: {
    background: '#89b4fa', border: '1px solid #89b4fa', color: '#11111b',
    cursor: 'pointer', fontSize: 10, fontWeight: 600, padding: '2px 9px', borderRadius: 10,
  },
  btn: {
    background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
    fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 4,
  },
  btnActive: {
    background: '#89b4fa', border: 'none', color: '#11111b', cursor: 'pointer',
    fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 4,
  },
  btnPrimary: {
    background: '#fab387', border: 'none', color: '#11111b', cursor: 'pointer',
    fontSize: 10, fontWeight: 700, padding: '2px 9px', borderRadius: 4,
  },
  select: {
    background: '#181825', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 4, fontSize: 10, fontWeight: 600, padding: '2px 4px', cursor: 'pointer',
  },
}
