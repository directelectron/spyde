/**
 * FloatingToolbar.tsx — Electron port of the Qt floating plot toolbar.
 *
 * A rounded translucent bar floating BELOW the owning PlotWindow that tracks it
 * (parented to the window root, so it shares the window's z-level — an
 * overlapping window that is above the window also covers its toolbar). It
 * falls back to floating INSIDE the window's bottom edge only when there is no
 * room below (maximized / dragged to the area bottom). Mirrors Qt
 * `RoundedToolBar`:
 *   • actions with `parameters`   → a CaretParams popout (params + Run),
 *   • actions with `subfunctions` → a second floating sub-toolbar,
 *   • toggling is exclusive; an "active" (live-output) action highlights and
 *     clicking it again deselects (hides its output + ROI).
 *
 * Popouts are HORIZONTAL — params sit side-by-side so the panel grows WIDER, not
 * taller (Find Vectors' five params used to make a tall popout that ran off the
 * MDI). Carets prefer opening BELOW the bar (clear of the figure); when that
 * would run off the MDI area they float to the window's right (or left), and
 * snap back below as soon as the window is moved somewhere with room.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { ToolbarAction, ParamSpec, SubAction } from '../kernel/SpyDEContext'
import { OrientationWizard } from './OrientationWizard'
import { FindVectorsWizard } from './FindVectorsWizard'
import { VectorOrientationWizard } from './VectorOrientationWizard'
import { CenterZeroBeamWizard } from './CenterZeroBeamWizard'
import { StrainWizard } from './StrainWizard'

const WIZARD_ACTIONS = new Set([
  'Orientation Mapping', 'Find Diffraction Vectors', 'Vector Orientation Mapping',
  'Center Zero Beam', 'Strain Mapping',
])

/**
 * Turn an OS filesystem path into a `spyde-fig://icons/<path>` URL. The Python
 * backend sends native icon paths (absolute package-asset paths). We can't load
 * them as raw `file://` <img> subresources because webSecurity is enabled and
 * the dev renderer runs on an http://localhost origin (Chromium blocks file://
 * subresources from http origins). The main process serves them through the
 * privileged spyde-fig scheme, validating the path is a real .svg/.png under a
 * spyde ".../icons/" directory (see resolveIconPath in main/index.ts).
 */
function fileUrl(p: string): string {
  return `spyde-fig://icons/${encodeURIComponent(p)}`
}

// Actions that draw a live marker overlay on the DP. The overlay is shown only
// while the action's caret is selected, and hidden when it's deselected.
const OVERLAY_ACTIONS = new Set([
  'Find Diffraction Vectors', 'Orientation Mapping', 'Vector Orientation Mapping',
])

const EMPTY = new Set<string>()
const HIDDEN_ACTIONS = new Set(['Reset', 'Zoom In', 'Zoom Out'])

export const BAR_H = 38     // bar box height (30px buttons + 2×3px padding + border)
export const BAR_GAP = 6    // gap between the window's bottom edge and the bar
const CARET_GAP = 10        // gap between the bar/window edge and an open caret
type CaretPlacement = 'below' | 'right' | 'left'

const STYLE_ID = 'spyde-toolbar-style'
if (typeof document !== 'undefined' && !document.getElementById(STYLE_ID)) {
  const s = document.createElement('style')
  s.id = STYLE_ID
  s.textContent =
    '@keyframes spyde-pop{from{opacity:0;transform:translate(-50%,-6px)}to{opacity:1;transform:translate(-50%,0)}}' +
    '@keyframes spyde-pop-up{from{opacity:0;transform:translate(-50%,6px)}to{opacity:1;transform:translate(-50%,0)}}' +
    '.spyde-tb-btn{background:none;border:none;color:#cdd6f4;cursor:pointer;width:30px;height:30px;' +
    'border-radius:6px;display:flex;align-items:center;justify-content:center;transition:background 90ms}' +
    '.spyde-tb-btn:hover{background:rgba(137,180,250,0.18)}'
  document.head.appendChild(s)
}

interface Props {
  actions: ToolbarAction[]
  windowId: number
  onAction: (name: string, windowId: number, params: Record<string, unknown>) => void
  /** Reveal-on-hover: true when the cursor is over the owning window. */
  visible?: boolean
  onHoverShow?: () => void
  onHoverHide?: () => void
  /** Live rect of the owning window in MDI-area coords (drives caret placement). */
  winRect?: { x: number; y: number; w: number; h: number }
  /** Current MDI-area size — carets must stay inside it (the area clips). */
  areaSize?: { w: number; h: number }
  /** True when the bar should sit INSIDE the window's bottom edge (maximized /
   *  no room below the window). */
  inside?: boolean
}

function defaultsOf(parameters: Record<string, ParamSpec>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [key, spec] of Object.entries(parameters || {})) {
    if (spec && 'default' in spec) out[key] = spec.default
  }
  return out
}
const hasParams = (a: ToolbarAction) => Object.keys(a.parameters || {}).length > 0
const hasSubs = (a: ToolbarAction) => (a.subfunctions?.length ?? 0) > 0
const hasPopout = (a: ToolbarAction) => hasParams(a) || hasSubs(a)

export function FloatingToolbar({
  actions, windowId, onAction, visible = true, onHoverShow, onHoverHide,
  winRect, areaSize, inside = false,
}: Props) {
  const { state, sendAction } = useSpyDE()
  const [openName, setOpenName] = React.useState<string | null>(null)
  const [placement, setPlacement] = React.useState<CaretPlacement>('below')
  const rootRef = React.useRef<HTMLDivElement>(null)
  // Wrapper around the open caret (position: static, so it does NOT affect the
  // caret's absolute positioning) used only to measure the caret's real size.
  const caretWrapRef = React.useRef<HTMLDivElement>(null)
  const caretBox = React.useRef<{ w: number; h: number } | null>(null)
  const live = state.activeActions.get(windowId) ?? EMPTY

  // Keep the toolbar shown while a popout/caret is open or an action is live —
  // otherwise reveal only on hover (over the window or the toolbar).
  const forced = openName !== null || live.size > 0
  const shownVisible = visible || forced

  // Caret placement: prefer BELOW the bar (clear of the figure); if the caret
  // would run off the bottom of the MDI area, float it to the window's RIGHT
  // (or LEFT when there is no room right either). Runs after every render so
  // it re-evaluates as the window moves/resizes or the caret's content changes
  // height — a caret pushed to the side snaps back below as soon as there is
  // room again.
  const wr = winRect ?? { x: 0, y: 0, w: 0, h: 0 }
  const area = areaSize ?? { w: 100000, h: 100000 }
  React.useLayoutEffect(() => {
    if (!openName) return
    const el = caretWrapRef.current?.firstElementChild as HTMLElement | null
    if (el) {
      const r = el.getBoundingClientRect()
      if (r.width > 0 && r.height > 0) caretBox.current = { w: r.width, h: r.height }
    }
    const cw = caretBox.current?.w ?? 240
    const ch = caretBox.current?.h ?? 320
    const belowTop = wr.y + wr.h + (inside ? 0 : BAR_H + BAR_GAP) + CARET_GAP
    let next: CaretPlacement = 'below'
    if (belowTop + ch > area.h) {
      next = wr.x + wr.w + CARET_GAP + cw <= area.w ? 'right' : 'left'
    }
    setPlacement(p => (p === next ? p : next))
  })

  React.useEffect(() => {
    if (!openName) return
    // Wizards (Find Diffraction Vectors, Orientation, …) are STATEFUL tools that
    // stay open until explicitly closed via their ✕ or by toggling the toolbar
    // button. They must survive interacting with OTHER windows — e.g. moving the
    // navigator selector to preview a new pattern — so an outside click does not
    // close them. Plain param popouts still dismiss on an outside click.
    if (WIZARD_ACTIONS.has(openName)) return
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node
      if (rootRef.current?.contains(target)) return   // inside the toolbar/caret
      // Keep the caret OPEN while interacting with its own window — e.g. grabbing
      // the title bar to drag, or the resize handle. The caret is parented to the
      // window so it moves along. Only close on a click outside this window.
      const win = rootRef.current?.closest('[data-testid="subwindow"]')
      if (win && win.contains(target)) return
      setOpenName(null)
    }
    window.addEventListener('mousedown', onDown)
    return () => window.removeEventListener('mousedown', onDown)
  }, [openName])

  // The DP marker overlay of an action is visible only while its caret is open;
  // deselecting (closing the caret or opening another) hides it. (Backend is a
  // no-op for windows/actions without an overlay.)
  React.useEffect(() => {
    for (const name of OVERLAY_ACTIONS) {
      sendAction('set_overlay', { name, visible: openName === name }, windowId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openName])

  const shown = actions.filter(a => !HIDDEN_ACTIONS.has(a.name))
  if (!shown.length) return null

  const click = (a: ToolbarAction) => {
    if (live.has(a.name)) {
      sendAction('set_action_active', { name: a.name, active: false }, windowId)
      setOpenName(null)
      return
    }
    // Deselecting an OPEN sub-toolbar action with live items (Virtual
    // Imaging / Vector VI) tears all its ROIs + output windows down (Qt
    // parity: an unchecked action removes its artifacts). Committed trees
    // are independent SignalTrees and survive.
    const items = state.subItems.get(windowId)?.get(a.name) ?? []
    if (openName === a.name && hasSubs(a) && items.length > 0) {
      sendAction('set_action_active', { name: a.name, active: false }, windowId)
      setOpenName(null)
      return
    }
    // Wizard actions (e.g. Vector Orientation Mapping) carry no params/subs but
    // still open a caret, so never dispatch them as a plain toolbar action.
    if (!hasPopout(a) && !WIZARD_ACTIONS.has(a.name)) {
      onAction(a.name, windowId, {}); setOpenName(null); return
    }
    // Opening a different caret: drop the stale size measurement so placement
    // is recomputed from the new caret's real box.
    if (openName !== a.name) caretBox.current = null
    setOpenName(openName === a.name ? null : a.name)
  }

  const openAction = shown.find(a => a.name === openName) || null

  // Where the bar's TOP edge sits in window coords — carets are DOM children of
  // the bar, so the side placements are expressed relative to it.
  const barTopInWin = inside ? wr.h - BAR_H - BAR_GAP : wr.h + BAR_GAP
  const caretPos: React.CSSProperties =
    placement === 'below'
      ? { position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)', marginTop: CARET_GAP }
      : placement === 'right'
        ? { position: 'absolute', top: -barTopInWin, left: '50%', marginLeft: wr.w / 2 + CARET_GAP, transform: 'none' }
        : { position: 'absolute', top: -barTopInWin, right: '50%', marginRight: wr.w / 2 + CARET_GAP, left: 'auto', transform: 'none' }

  return (
    <div
      ref={rootRef}
      data-testid="floating-toolbar"
      onMouseEnter={onHoverShow}
      onMouseLeave={onHoverHide}
      style={{
        ...styles.bar,
        ...(inside ? { bottom: BAR_GAP } : { top: '100%', marginTop: BAR_GAP }),
        opacity: shownVisible ? 1 : 0,
        pointerEvents: shownVisible ? 'auto' : 'none',
        transition: 'opacity 140ms ease',
      }}
    >
      {shown.map(a => (
        <button
          key={a.name}
          title={a.name}
          data-testid={`action-btn-${a.name}`}
          className="spyde-tb-btn"
          style={(openName === a.name || live.has(a.name)
            || (state.subItems.get(windowId)?.get(a.name)?.length ?? 0) > 0)
            ? styles.btnActive : undefined}
          onClick={() => click(a)}
        >
          {a.icon && a.icon.endsWith('.svg')
            ? <img src={fileUrl(a.icon)} width={18} height={18} alt={a.name} />
            : <span style={{ fontSize: 10 }}>{a.name.slice(0, 4)}</span>}
        </button>
      ))}

      {/* Static wrapper (no box of its own) — lets the placement effect measure
          the open caret without affecting its absolute positioning. */}
      <div ref={caretWrapRef}>
        {openAction && openAction.name === 'Orientation Mapping' && (
          <OrientationWizard
            caretPos={caretPos} windowId={windowId} sendAction={sendAction}
            onClose={() => setOpenName(null)}
          />
        )}
        {openAction && openAction.name === 'Find Diffraction Vectors' && (
          <FindVectorsWizard
            caretPos={caretPos} windowId={windowId} sendAction={sendAction}
            onClose={() => setOpenName(null)}
          />
        )}
        {openAction && openAction.name === 'Vector Orientation Mapping' && (
          <VectorOrientationWizard
            caretPos={caretPos} windowId={windowId} sendAction={sendAction}
            onClose={() => setOpenName(null)}
          />
        )}
        {openAction && openAction.name === 'Center Zero Beam' && (
          <CenterZeroBeamWizard
            caretPos={caretPos} windowId={windowId} sendAction={sendAction}
            onClose={() => setOpenName(null)}
          />
        )}
        {openAction && openAction.name === 'Strain Mapping' && (
          <StrainWizard
            caretPos={caretPos} windowId={windowId} sendAction={sendAction}
            onClose={() => setOpenName(null)}
          />
        )}
        {openAction && !WIZARD_ACTIONS.has(openAction.name) && hasParams(openAction) && (
          <ParamPopout
            action={openAction} caretPos={caretPos} below={placement === 'below'}
            onRun={(params) => { onAction(openAction.name, windowId, params); setOpenName(null) }}
            onClose={() => setOpenName(null)}
          />
        )}
      </div>
      {openAction && !hasParams(openAction) && hasSubs(openAction) && (
        <SubToolbar
          action={openAction}
          up={placement !== 'below'}
          items={state.subItems.get(windowId)?.get(openAction.name) ?? []}
          onSub={(sub) => { onAction(sub.name, windowId, defaultsOf(sub.parameters)) }}
          onUpdate={(name, params) => sendAction('update_vi', { name, params }, windowId)}
          onRemove={(itemName) => sendAction('set_action_active', { name: itemName, active: false }, windowId)}
          onCommit={(itemName) => sendAction('vi_commit', { name: itemName }, windowId)}
        />
      )}
    </div>
  )
}

function rowVisible(spec: ParamSpec, values: Record<string, unknown>): boolean {
  const c = spec.display_condition
  if (!c || !c.parameter) return true
  return String(values[c.parameter]) === String(c.value)
}

function ParamPopout({ action, caretPos, below, onRun, onClose }: {
  action: ToolbarAction
  caretPos: React.CSSProperties
  below: boolean
  onRun: (params: Record<string, unknown>) => void
  onClose: () => void
}) {
  const [values, setValues] = React.useState<Record<string, unknown>>(
    () => defaultsOf(action.parameters))
  const allParams = Object.entries(action.parameters || {})
  const set = (k: string, v: unknown) => setValues(s => ({ ...s, [k]: v }))

  // Optional tabs: if any param declares a `tab`, group the caret into tabs.
  const tabs = Array.from(new Set(allParams.map(([, s]) => s.tab).filter(Boolean))) as string[]
  const [tab, setTab] = React.useState<string | null>(tabs[0] ?? null)
  const params = tabs.length
    ? allParams.filter(([, s]) => (s.tab ?? tabs[0]) === tab)
    : allParams

  return (
    <div data-testid="action-flyout"
      style={{
        ...popBase, ...caretPos, flexDirection: 'column', alignItems: 'stretch',
        ...(below ? { animation: 'spyde-pop 130ms ease-out' } : {}),
      }}>
      {below && <div style={styles.caretUp} />}
      <div style={styles.popHead}>
        <span style={styles.popTitle}>{action.name}</span>
        <button data-testid="action-flyout-close" style={styles.closeBtn} onClick={onClose}>✕</button>
      </div>
      {tabs.length > 0 && (
        <div style={styles.tabRow}>
          {tabs.map(t => (
            <button key={t} data-testid={`param-tab-${t}`}
              style={t === tab ? styles.tabActive : styles.tab}
              onClick={() => setTab(t)}>{t}</button>
          ))}
        </div>
      )}
      <div style={styles.paramWrap}>
        {params.map(([key, spec]) => (
          rowVisible(spec, values) && (
            <div key={key} style={styles.cell} data-testid={`param-row-${key}`}>
              <label style={styles.cellLabel}>{spec.name || key}</label>
              <ParamControl paramKey={key} spec={spec} value={values[key]} onChange={(v) => set(key, v)} />
            </div>
          )
        ))}
      </div>
      <button data-testid="action-run" style={{ ...styles.runBtn, alignSelf: 'flex-start' }}
        onClick={() => onRun(values)}>Run</button>
    </div>
  )
}

type ViItem = { name: string; color: string; vtype?: string; calculation?: string }

/** The second floating toolbar (Qt PopoutToolBar), below the main bar (flips
 *  above it when there is no room below the window): a "＋" to add, then one
 *  colour-coded detector-shape icon per virtual image. Clicking a VI icon opens
 *  its own parameter caret (detector type / calc). */
function SubToolbar({ action, up, items, onSub, onUpdate, onRemove, onCommit }: {
  action: ToolbarAction
  up: boolean
  items: ViItem[]
  onSub: (sub: SubAction) => void
  onUpdate: (name: string, params: Record<string, unknown>) => void
  onRemove: (name: string) => void
  onCommit: (name: string) => void
}) {
  const [openVi, setOpenVi] = React.useState<string | null>(null)
  const subs = action.subfunctions || []
  const paramSpec = subs[0]?.parameters ?? {}

  // A freshly-added item opens its caret right away so the user sees the
  // detector options for the ROI they just placed.
  const prevCount = React.useRef(items.length)
  React.useEffect(() => {
    if (items.length > prevCount.current) setOpenVi(items[items.length - 1].name)
    prevCount.current = items.length
  }, [items])

  return (
    <div data-testid="sub-toolbar" style={up ? styles.subBarUp : styles.subBar}>
      <div style={up ? styles.caretDownSub : styles.caretUpSub} />
      {subs.map(sub => (
        <button
          key={sub.name}
          title={sub.label || sub.name}
          data-testid={`subaction-${sub.name}`}
          className="spyde-tb-btn"
          style={{ fontSize: 18, fontWeight: 700 }}
          onClick={() => onSub(sub)}
        >＋</button>
      ))}
      {items.map(it => (
        <div key={it.name} style={{ position: 'relative' }}>
          <button
            data-testid={`vi-icon-${it.name}`}
            title={it.name}
            className="spyde-tb-btn"
            style={openVi === it.name ? styles.btnActive : undefined}
            onClick={() => setOpenVi(openVi === it.name ? null : it.name)}
          >
            <ViShape type={it.vtype || 'disk'} color={it.color} />
          </button>
          {openVi === it.name && (
            <ViCaret
              item={it} spec={paramSpec}
              onChange={(params) => onUpdate(it.name, params)}
              onRemove={() => { onRemove(it.name); setOpenVi(null) }}
              onCommit={() => onCommit(it.name)}
            />
          )}
        </div>
      ))}
    </div>
  )
}

/** Coloured detector-shape icon: disk=filled circle, annular=ring, rectangle=square. */
function ViShape({ type, color }: { type: string; color: string }) {
  if (type === 'annular') {
    return <svg width={18} height={18}><circle cx={9} cy={9} r={6.5} fill="none" stroke={color} strokeWidth={3} /></svg>
  }
  if (type === 'rectangle') {
    return <svg width={18} height={18}><rect x={3} y={3} width={12} height={12} rx={1} fill={color} /></svg>
  }
  return <svg width={18} height={18}><circle cx={9} cy={9} r={7} fill={color} /></svg>
}

/** Per-VI parameter caret (detector type / calculation), opens below its icon. */
function ViCaret({ item, spec, onChange, onRemove, onCommit }: {
  item: ViItem
  spec: Record<string, ParamSpec>
  onChange: (params: Record<string, unknown>) => void
  onRemove: () => void
  onCommit: () => void
}) {
  const values: Record<string, unknown> = { type: item.vtype, calculation: item.calculation }
  const params = Object.entries(spec)
  const label = item.name.replace(/\s*\([^)]*\)\s*$/, '')   // drop the "(red)" suffix
  return (
    <div data-testid={`vi-caret-${item.name}`} style={styles.viCaret}>
      <div style={styles.caretUp} />
      <div style={styles.viCaretHead}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <ViShape type={item.vtype || 'disk'} color={item.color} /> {label}
        </span>
        <button data-testid={`vi-remove-${item.name}`} style={styles.closeBtn}
          title="Remove" onClick={onRemove}>✕</button>
      </div>
      <div style={styles.viCaretRows}>
        {params.map(([key, p]) => (
          <div key={key} style={styles.cell} data-testid={`vi-param-row-${item.name}-${key}`}>
            <label style={styles.cellLabel}>{p.name || key}</label>
            <ParamControl paramKey={`vi-${key}`} spec={p} value={values[key]}
              onChange={(v) => onChange({ [key]: v })} />
          </div>
        ))}
      </div>
      {/* The standard Commit affordance (same as strain): freeze this virtual
          image into its own SignalTree; the live ROI/window stay for tuning. */}
      <button data-testid={`vi-commit-${item.name}`} style={styles.runBtn}
        onClick={onCommit}>Commit to New Tree</button>
    </div>
  )
}

function ParamControl({ paramKey, spec, value, onChange }: {
  paramKey: string; spec: ParamSpec; value: unknown; onChange: (v: unknown) => void
}) {
  const tid = `param-${paramKey}`
  const t = (spec.type || '').toLowerCase()

  if (spec.options && spec.options.length) {
    return (
      <select data-testid={tid} style={styles.control}
        value={String(value ?? spec.default ?? spec.options[0])}
        onChange={(e) => onChange(e.target.value)}>
        {spec.options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    )
  }
  if (t === 'bool' || t === 'boolean') {
    return <input data-testid={tid} type="checkbox" checked={Boolean(value)}
      onChange={(e) => onChange(e.target.checked)} />
  }
  if (t === 'file') {
    const base = value ? String(value).split(/[/\\]/).pop() : 'Choose…'
    return (
      <button data-testid={tid} style={styles.fileBtn} title={value ? String(value) : ''}
        onClick={async () => {
          const path = await window.electron.pickFile({ name: spec.name, extensions: spec.extensions })
          if (path) onChange(path)
        }}>{base}</button>
    )
  }
  const numeric = t === 'int' || t === 'integer' || t === 'float' || t === 'number'
  if (numeric && spec.min != null && spec.max != null) {
    const step = spec.step ?? ((t === 'int' || t === 'integer') ? 1 : 'any')
    const v = value == null ? spec.default ?? spec.min : value
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <input data-testid={tid} type="range" min={spec.min} max={spec.max} step={step}
          value={Number(v)} style={{ width: 72 }}
          onChange={(e) => onChange(Number(e.target.value))} />
        <span style={styles.sliderVal}>{String(v)}</span>
      </div>
    )
  }
  if (numeric) {
    const step = (t === 'int' || t === 'integer') ? 1 : 'any'
    return <input data-testid={tid} type="number" step={step} style={styles.numInput}
      value={value == null ? '' : String(value)}
      onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))} />
  }
  if (t === 'rectangleselector') {
    return <span data-testid={tid} style={styles.hint}>set on plot</span>
  }
  return <input data-testid={tid} type="text" style={styles.numInput}
    value={value == null ? '' : String(value)}
    onChange={(e) => onChange(e.target.value)} />
}

const POP_BG = '#1e1e2e'
const SUB_BG = 'rgba(24,24,37,0.96)'
const caret = (dir: 'up' | 'down', bg: string): React.CSSProperties => ({
  position: 'absolute', left: '50%', marginLeft: -7, width: 0, height: 0,
  borderLeft: '7px solid transparent', borderRight: '7px solid transparent',
  ...(dir === 'up'
    ? { bottom: '100%', borderBottom: `8px solid ${bg}` }
    : { top: '100%', borderTop: `8px solid ${bg}` }),
})

const popBase: React.CSSProperties = {
  position: 'absolute', left: '50%', transform: 'translateX(-50%)',
  background: POP_BG, border: '1px solid #313244', borderRadius: 8,
  padding: '6px 8px', zIndex: 13, color: '#cdd6f4',
  boxShadow: '0 8px 24px rgba(0,0,0,0.55)',
  // Lay out as ONE wide row (max-content), but cap the width so a long action
  // (e.g. Find Vectors' 5 params) WRAPS to a second row instead of getting
  // super long. `width:max-content` stops the absolutely-positioned flex box
  // from shrink-wrapping into a tall narrow column.
  display: 'flex', flexDirection: 'row', flexWrap: 'wrap',
  alignItems: 'flex-end', gap: 8, rowGap: 8,
  width: 'max-content', maxWidth: 470,
}
const subBase: React.CSSProperties = {
  position: 'absolute', left: '50%', transform: 'translateX(-50%)',
  display: 'flex', alignItems: 'center', gap: 2,
  background: SUB_BG, border: '1px solid #313244', borderRadius: 10,
  padding: '3px 5px', zIndex: 13, boxShadow: '0 6px 20px rgba(0,0,0,0.5)',
}

const styles: Record<string, React.CSSProperties> = {
  bar: {
    // Floats BELOW the window (inside it only as a no-room fallback — the
    // vertical position is applied inline), centered, tracking move/resize.
    // Reveal-on-hover is handled by the opacity/pointerEvents applied inline.
    // The bar lives in the window's stacking context, so it shares the
    // window's z-level: a sibling window stacked above also covers the bar,
    // and a hidden bar (pointerEvents:none) never intercepts clicks headed
    // for a window beneath.
    position: 'absolute', left: '50%',
    transform: 'translateX(-50%)',
    display: 'flex', alignItems: 'center', gap: 2,
    background: 'rgba(24,24,37,0.92)', border: '1px solid #313244',
    borderRadius: 10, padding: '3px 5px', zIndex: 12,
    boxShadow: '0 6px 20px rgba(0,0,0,0.5)',
  },
  btnActive: {
    background: '#89b4fa', color: '#11111b',
    border: 'none', cursor: 'pointer', width: 30, height: 30, borderRadius: 6,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  subBar: { ...subBase, top: '100%', marginTop: 10, animation: 'spyde-pop 130ms ease-out' },
  subBarUp: { ...subBase, bottom: '100%', marginBottom: 10, animation: 'spyde-pop-up 130ms ease-out' },
  caretUp: caret('up', POP_BG),
  caretUpSub: caret('up', SUB_BG),
  caretDownSub: caret('down', SUB_BG),
  popHead: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    marginBottom: 4,
  },
  popTitle: { fontSize: 11, fontWeight: 600, color: '#cdd6f4', whiteSpace: 'nowrap' },
  tabRow: {
    display: 'flex', gap: 2, marginBottom: 6,
    borderBottom: '1px solid #313244', paddingBottom: 4,
  },
  tab: {
    background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
    fontSize: 11, padding: '2px 8px', borderRadius: 4,
  },
  tabActive: {
    background: '#313244', border: 'none', color: '#cdd6f4', cursor: 'pointer',
    fontSize: 11, padding: '2px 8px', borderRadius: 4, fontWeight: 600,
  },
  paramWrap: {
    display: 'flex', flexDirection: 'row', flexWrap: 'wrap',
    alignItems: 'flex-end', gap: 8, rowGap: 8, maxWidth: 440, marginBottom: 6,
  },
  cell: { display: 'flex', flexDirection: 'column', gap: 2, alignItems: 'flex-start' },
  cellLabel: { fontSize: 9, color: '#a6adc8', whiteSpace: 'nowrap' },
  control: {
    background: '#11111b', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 4, padding: '3px 5px', fontSize: 11,
  },
  numInput: {
    width: 64, background: '#11111b', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 4, padding: '3px 5px', fontSize: 11,
  },
  sliderVal: { fontSize: 10, color: '#cdd6f4', minWidth: 22, textAlign: 'right' },
  fileBtn: {
    background: '#313244', color: '#cdd6f4', border: '1px solid #45475a',
    borderRadius: 4, padding: '3px 8px', fontSize: 11, cursor: 'pointer',
    maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  hint: { fontSize: 10, color: '#6c7086', fontStyle: 'italic' },
  runBtn: {
    background: '#89b4fa', color: '#11111b', border: 'none', borderRadius: 4,
    padding: '5px 10px', fontSize: 12, fontWeight: 600, cursor: 'pointer', alignSelf: 'center',
  },
  closeBtn: {
    background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
    fontSize: 12, padding: '0 2px', alignSelf: 'flex-start',
  },
  // Per-VI parameter caret — opens below its icon, params on (up to) two rows.
  viCaret: {
    position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
    marginTop: 10, width: 200, maxWidth: '90vw',
    background: POP_BG, border: '1px solid #313244', borderRadius: 8,
    padding: 8, zIndex: 14, color: '#cdd6f4',
    boxShadow: '0 8px 24px rgba(0,0,0,0.55)',
    display: 'flex', flexDirection: 'column', gap: 6,
    animation: 'spyde-pop 130ms ease-out',
  },
  viCaretHead: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    fontSize: 11, fontWeight: 600,
  },
  viCaretRows: {
    display: 'flex', flexDirection: 'row', flexWrap: 'wrap',
    alignItems: 'flex-end', gap: 8,
  },
}
