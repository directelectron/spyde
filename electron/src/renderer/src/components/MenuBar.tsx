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

// Curated, ALWAYS-AVAILABLE small in-memory tutorial datasets — unlike the real
// examples (downloaded via em-database), these fire the ungated `tutorial_load`
// backend action and load in a couple of seconds with no network. Keys mirror
// spyde/backend/tutorial_data.py TUTORIAL_LOADERS; the guided walkthroughs
// drive the same action names.
const TUTORIAL_DATA: { key: string; label: string }[] = [
  { key: 'navigation', label: 'Navigation & Virtual Imaging' },
  { key: 'find_vectors', label: 'Find Vectors (Si grains)' },
  { key: 'orientation', label: 'Orientation Mapping' },
  { key: 'multiphase', label: 'Multi-Phase Orientation Mapping' },
  { key: 'strain', label: 'Strain Mapping' },
  { key: 'spectroscopy', label: 'Spectroscopy (1D)' },
  { key: 'movie', label: 'In-situ Movie' },
]

/** One dataset row of the Examples catalogue (spyde/backend/example_catalogue). */
export interface ExampleItem {
  key: string
  label: string
  technique: string
  size: string
  shape: string | null
  downloaded: boolean
  description?: string
  file?: string
  microscope?: string
  voltage?: string
  detector?: string
  detector_manufacturer?: string
}

/** "CeleritasXS (Direct Electron)" — either half alone when that is all there
 *  is (a HAADF dataset has a manufacturer but no named detector). */
function cameraOf(it: ExampleItem): string {
  const [name, make] = [it.detector?.trim(), it.detector_manufacturer?.trim()]
  if (name && make) return `${name} (${make})`
  return name || make || ''
}
interface ExampleGroup { technique: string; items: ExampleItem[] }

/** The themed hover card's contents. Replaces the native `title=` tooltip,
 *  which renders as an unstyled OS bubble in the wrong font, colour and
 *  position — jarring against the rest of the app. */
export interface Tip {
  title: string
  /** Small caps line under the title: technique · microscope · voltage. */
  meta?: string
  /** Facts worth scanning: shape, size, file. */
  facts?: [string, string][]
  body?: string
  /** Download state, shown with the same dot the row uses. */
  downloaded?: boolean
}

type Item =
  | {
      label: string
      onClick: () => void
      disabled?: boolean
      testId?: string
      /** Right-aligned detail (dataset size). */
      detail?: string
      /** Middle column: nav|signal shape, blank when not yet known. */
      shape?: string
      /** Present on dataset rows: already on disk, or a download. */
      downloaded?: boolean
      tip?: Tip
    }
  | { label: string; submenu: Item[]; testId?: string; detail?: string }
  | { separator: true }
  | { header: string }

const hasSubmenu = (it: Item): it is { label: string; submenu: Item[]; testId?: string; detail?: string } =>
  'submenu' in it

/** Filled dot = on disk, hollow = will download. Glyphs rather than colour so
 *  the distinction survives a screenshot, a colour-blind reader and a theme. */
const DOWNLOADED_MARK = '●'      // ●
const NOT_DOWNLOADED_MARK = '○'  // ○

export function MenuBar({ onStartGuide, onShowInfo }: {
  onStartGuide: (g: Guide) => void
  /** Help → <technique> → Info… — opens GuideInfoDialog for that technique. */
  onShowInfo: (g: Guide) => void
}) {
  const { sendAction, openStackDialog, openUpdateDialog, openGpuStatusDialog, openGpuHelpDialog, openReportDialog, state } = useSpyDE()
  const [open, setOpen] = useState<string | null>(null)
  const barRef = useRef<HTMLDivElement>(null)
  const [exampleGroups, setExampleGroups] = useState<ExampleGroup[]>([])
  const [dataDir, setDataDir] = useState('')
  const [examplesReady, setExamplesReady] = useState(false)
  const [tip, setTip] = useState<{ tip: Tip; anchor: DOMRect } | null>(null)

  // The card belongs to whatever menu is open; closing one must not leave it
  // stranded on screen.
  useEffect(() => { if (!open) setTip(null) }, [open])

  // `sendAction` is a fresh closure on EVERY provider render (SpyDEContext builds
  // its value inline), so an effect that lists it as a dependency re-runs on every
  // unrelated context update. Held in a ref instead, the two catalogue effects
  // below depend only on what should genuinely re-trigger them — before this, one
  // menu open fired a dozen `example_catalogue` requests in ~40 ms, each one
  // spawning a shape-warming thread on the backend.
  const sendActionRef = useRef(sendAction)
  sendActionRef.current = sendAction

  // Prefetch as soon as the backend is up, so the menu is already populated the
  // first time it is opened. Two reasons this is not just a nicety: the window
  // exists well BEFORE the Python sidecar does (main/index.ts creates it, then
  // awaits resolvePythonEnv — a whole `uv sync` on a first packaged run), and
  // main/runner.ts's sendAction silently no-ops while `proc.stdin` is null. So a
  // menu opened during startup dropped its request and sat on "Loading examples…"
  // until the user closed and reopened it. `state.ready` is set by the backend's
  // own `ready` message, which it emits immediately before entering its stdin
  // loop — i.e. exactly when an action can first be received.
  const prefetched = useRef(false)
  useEffect(() => {
    if (!state.ready || prefetched.current) return
    prefetched.current = true
    sendActionRef.current('example_catalogue', { warm: true })
  }, [state.ready])

  // The catalogue is cheap to build once em-database is imported (~1 ms; it opens
  // no files), so ask for a fresh one every time the Examples menu opens — that is
  // what keeps the downloaded markers honest after a download finishes or a file is
  // deleted outside the app. `warm` lets the backend fill in shapes for anything
  // downloaded but not yet measured, off this path, and re-send when it has.
  useEffect(() => {
    if (open !== 'Examples') return
    sendActionRef.current('example_catalogue', { warm: true })
  }, [open])

  useEffect(() => {
    const onCatalogue = (e: Event) => {
      const d = (e as CustomEvent).detail as Record<string, unknown>
      setExampleGroups((d.groups as ExampleGroup[]) ?? [])
      setDataDir(String(d.data_dir ?? ''))
      setExamplesReady(true)
    }
    window.addEventListener('spyde:example_catalogue', onCatalogue)
    return () => window.removeEventListener('spyde:example_catalogue', onCatalogue)
  }, [])

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
    Examples: [
      // Real datasets from em-database, one submenu per technique. Each row
      // shows its size, its shape where known, and whether it is already on
      // disk — so you can tell a 42 kB line scan from a 2.4 GB in-situ movie,
      // and an instant load from a download, BEFORE clicking.
      ...exampleGroups.map((g) => ({
        label: g.technique,
        testId: `examples-tech-${g.technique.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`,
        detail: `${g.items.filter((i) => i.downloaded).length}/${g.items.length}`,
        submenu: g.items.map((it) => ({
          label: it.label,
          testId: `example-${it.key}`,
          downloaded: it.downloaded,
          shape: it.shape ?? '',
          detail: it.size,
          tip: {
            title: it.label,
            meta: [it.technique, it.microscope, it.voltage].filter(Boolean).join(' · '),
            facts: ([
              it.shape ? ['Shape', it.shape] : null,
              ['Size', it.size],
              cameraOf(it) ? ['Camera', cameraOf(it)] : null,
              it.file ? ['File', it.file] : null,
            ].filter(Boolean)) as [string, string][],
            body: it.description,
            downloaded: it.downloaded,
          },
          onClick: () => sendAction('load_example', { name: it.key }),
        })),
      })),
      ...(exampleGroups.length === 0
        ? [{
            label: examplesReady ? 'No example datasets found' : 'Loading examples…',
            disabled: true,
            onClick: () => {},
          } as Item]
        : []),
      // Instant, no-download synthetic datasets — their own submenu, so they
      // read as a distinct kind of thing rather than a tail of the real data.
      { separator: true } as Item,
      {
        label: 'Dummy Data',
        testId: 'examples-dummy-data',
        detail: 'no download',
        submenu: TUTORIAL_DATA.map(({ key, label }) => ({
          label,
          testId: `tutorial-${key}`,
          onClick: () => sendAction('tutorial_load', { name: key }),
        })),
      } as Item,
      { separator: true } as Item,
      {
        label: 'Show Example Data Directory',
        testId: 'examples-show-dir',
        detail: dataDir ? undefined : '',
        title: dataDir || undefined,
        onClick: () => sendAction('show_example_dir', {}),
      } as Item,
    ],
    Help: [
      // One row per TECHNIQUE, each opening a two-entry sub-menu: Info (the
      // background + further reading) and Guided tour (the in-app walkthrough).
      // Previously this was a flat list of "Guided Tour: <title>" items, which
      // offered no way to read about a technique without starting a tour.
      { header: 'Techniques' },
      ...GUIDES.map((g) => ({
        label: g.title,
        testId: `help-technique-${g.id}`,
        submenu: [
          {
            label: 'Info…',
            testId: `help-info-${g.id}`,
            onClick: () => onShowInfo(g),
          },
          {
            label: 'Guided tour',
            testId: `help-tour-${g.id}`,
            onClick: () => onStartGuide(g),
          },
        ],
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
      { separator: true },
      { label: 'Report a Problem…', onClick: () => openReportDialog() },
    ],
  }

  return (
    <div ref={barRef} style={styles.bar} data-testid="menu-bar">
      {Object.keys(menus).map((name) => (
        <div key={name} style={{ position: 'relative' }}>
          <button
            data-testid={`menu-${name.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`}
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
            <MenuList items={menus[name]} onClose={() => setOpen(null)}
                      onTip={(t, rect) => setTip(t && rect ? { tip: t, anchor: rect } : null)}
                      testId={`menu-${name.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}-items`} />
          )}
        </div>
      ))}
      {tip && <HoverCard tip={tip.tip} anchor={tip.anchor} />}
    </div>
  )
}

/**
 * The themed hover card — SpyDE's own panel chrome (the dropdown's surface,
 * border, radius and shadow), not the OS's `title=` bubble.
 *
 * Anchored to the RIGHT of the hovered row so it never covers the menu you are
 * reading, and flipped to the left when there is no room. `pointerEvents:none`
 * so it can't itself steal the hover that is keeping it open.
 */
function HoverCard({ tip, anchor }: { tip: Tip; anchor: DOMRect }) {
  const W = 300
  const GAP = 10
  const right = anchor.right + GAP
  const flip = right + W > window.innerWidth
  const left = flip ? Math.max(8, anchor.left - W - GAP) : right
  const top = Math.min(anchor.top - 4, Math.max(8, window.innerHeight - 260))
  return (
    <div data-testid="menu-hover-card"
         style={{ ...styles.card, width: W, left, top }}>
      <div style={styles.cardTitle}>{tip.title}</div>
      {tip.meta && <div style={styles.cardMeta}>{tip.meta}</div>}
      {tip.facts?.length ? (
        <div style={styles.cardFacts}>
          {tip.facts.map(([k, v]) => (
            <div key={k} style={styles.cardFactRow}>
              <span style={styles.cardFactKey}>{k}</span>
              <span style={styles.cardFactVal}>{v}</span>
            </div>
          ))}
        </div>
      ) : null}
      {tip.body && <div style={styles.cardBody}>{tip.body}</div>}
      {tip.downloaded !== undefined && (
        <div style={{ ...styles.cardState,
                      color: tip.downloaded ? '#a6e3a1' : '#89b4fa' }}>
          {tip.downloaded
            ? `${DOWNLOADED_MARK}  On disk — opens immediately`
            : `${NOT_DOWNLOADED_MARK}  Not downloaded — click to fetch`}
        </div>
      )}
    </div>
  )
}

/**
 * One dropdown level. Recurses for fly-out submenus (Examples groups its
 * datasets by technique), which the menu did not have before — the old
 * "Dummy Data" HEADER inside the flat Examples list is now a real submenu.
 *
 * A submenu opens on hover and stays open while the pointer is anywhere in the
 * parent row or the fly-out itself, so the diagonal travel from the row to the
 * panel doesn't dismiss it. It is rendered inside the parent row's relatively
 * positioned wrapper, so it follows the row.
 */
function MenuList({ items, onClose, testId, nested = false, onTip }: {
  items: Item[]
  onClose: () => void
  testId?: string
  nested?: boolean
  onTip?: (tip: Tip | null, anchor?: DOMRect) => void
}) {
  const [openSub, setOpenSub] = useState<string | null>(null)
  return (
    <div
      style={nested ? styles.submenu : styles.dropdown}
      data-testid={testId}
      onClick={(e) => e.stopPropagation()}
    >
      {items.map((it, i) => {
        if ('separator' in it) return <div key={`sep${i}`} style={styles.sep} />
        if ('header' in it) return <div key={`hdr${i}`} style={styles.header}>{it.header}</div>

        const tid = it.testId ?? `menu-item-${it.label.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`
        if (hasSubmenu(it)) {
          return (
            <div key={it.label} style={{ position: 'relative' }}
                 onMouseEnter={() => setOpenSub(it.label)}
                 onMouseLeave={() => setOpenSub((s) => (s === it.label ? null : s))}>
              <button data-testid={tid} style={{ ...styles.item, ...styles.row }}
                      onClick={(e) => { e.stopPropagation(); setOpenSub(it.label) }}>
                <span style={styles.label}>{it.label}</span>
                {it.detail && <span style={styles.detail}>{it.detail}</span>}
                <span style={styles.chevron}>›</span>
              </button>
              {openSub === it.label && (
                <MenuList items={it.submenu} onClose={onClose} nested
                          testId={`${tid}-items`} onTip={onTip} />
              )}
            </div>
          )
        }

        return (
          <button
            key={it.label}
            data-testid={tid}
            disabled={it.disabled}
            style={{ ...styles.item, ...styles.row, opacity: it.disabled ? 0.4 : 1 }}
            onClick={() => { onTip?.(null); onClose(); if (!it.disabled) it.onClick() }}
            onMouseEnter={(e) => {
              if (!it.disabled) e.currentTarget.style.background = '#313244'
              if (it.tip) onTip?.(it.tip, e.currentTarget.getBoundingClientRect())
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent'
              if (it.tip) onTip?.(null)
            }}
          >
            {it.downloaded !== undefined && (
              <span style={it.downloaded ? styles.markOn : styles.markOff}>
                {it.downloaded ? DOWNLOADED_MARK : NOT_DOWNLOADED_MARK}
              </span>
            )}
            <span style={styles.label}>{it.label}</span>
            {it.shape ? <span style={styles.shape}>{it.shape}</span> : null}
            {it.detail && <span style={styles.detail}>{it.detail}</span>}
          </button>
        )
      })}
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
  submenu: {
    position: 'absolute', top: -5, left: '100%', zIndex: 9300,
    minWidth: 260, background: '#1e1e2e', border: '1px solid #313244',
    borderRadius: 8, padding: 5, boxShadow: '0 10px 28px rgba(0,0,0,0.5)',
    maxHeight: '70vh', overflowY: 'auto',
  },
  // Rows are a 3-part grid: marker, label, right-aligned detail — so the sizes
  // and shapes line up in a readable column instead of trailing the names.
  row: { display: 'flex', alignItems: 'center', gap: 10 },
  // On disk = filled accent dot; not yet = hollow grey. Distinguished by BOTH
  // glyph and colour so it survives a greyscale screenshot as well as a
  // colour-blind reader.
  markOn: { fontSize: 11, width: 11, flex: '0 0 auto', color: '#89b4fa' },
  markOff: { fontSize: 11, width: 11, flex: '0 0 auto', color: '#585b70' },
  label: { flex: '1 1 auto', textAlign: 'left' },
  shape: {
    flex: '0 0 auto', fontSize: 11, color: '#6c7086',
    fontVariantNumeric: 'tabular-nums', paddingRight: 4,
  },
  detail: {
    flex: '0 0 auto', fontSize: 11, color: '#9399b2', minWidth: 58,
    textAlign: 'right', fontVariantNumeric: 'tabular-nums',
  },
  chevron: { flex: '0 0 auto', color: '#6c7086', fontSize: 14, lineHeight: 1 },
  // Same surface as the dropdowns — one panel language across the menu.
  card: {
    position: 'fixed', zIndex: 9400, pointerEvents: 'none',
    background: '#181825', border: '1px solid #313244', borderRadius: 8,
    padding: '10px 12px', boxShadow: '0 12px 32px rgba(0,0,0,0.55)',
    color: '#cdd6f4', fontSize: 12,
  },
  cardTitle: { fontSize: 13, fontWeight: 600, color: '#cdd6f4' },
  cardMeta: {
    // NOT uppercased: this line carries units, and text-transform turns
    // "200 kV" into "200 KV".
    marginTop: 2, fontSize: 10.5, fontWeight: 600, letterSpacing: 0.4,
    color: '#89b4fa',
  },
  cardFacts: {
    marginTop: 8, display: 'flex', flexDirection: 'column', gap: 2,
    paddingTop: 8, borderTop: '1px solid #313244',
  },
  cardFactRow: { display: 'flex', justifyContent: 'space-between', gap: 12 },
  cardFactKey: { color: '#6c7086', fontSize: 11 },
  cardFactVal: {
    color: '#bac2de', fontSize: 11, fontVariantNumeric: 'tabular-nums',
    textAlign: 'right', overflowWrap: 'anywhere',
  },
  cardBody: {
    marginTop: 8, paddingTop: 8, borderTop: '1px solid #313244',
    color: '#a6adc8', fontSize: 11.5, lineHeight: 1.45,
    // Long descriptions are useful but must not become a wall of text.
    display: '-webkit-box', WebkitLineClamp: 6, WebkitBoxOrient: 'vertical',
    overflow: 'hidden',
  },
  cardState: { marginTop: 8, fontSize: 11, fontWeight: 500 },
  sep: { height: 1, background: '#313244', margin: '5px 4px' },
  header: {
    padding: '4px 10px 2px', fontSize: 10.5, fontWeight: 700,
    letterSpacing: 0.6, textTransform: 'uppercase', color: '#6c7086',
    whiteSpace: 'nowrap', userSelect: 'none',
  },
}
