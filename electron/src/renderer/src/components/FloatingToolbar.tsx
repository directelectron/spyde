/**
 * FloatingToolbar.tsx — Electron port of the Qt floating plot toolbar.
 *
 * A rounded translucent bar floating over the bottom of a PlotWindow that tracks
 * it (parented to the window root). Mirrors Qt `RoundedToolBar`:
 *   • actions with `parameters`   → a CaretParams popout (params + Run),
 *   • actions with `subfunctions` → a second floating sub-toolbar,
 *   • toggling is exclusive; an "active" (live-output) action highlights and
 *     clicking it again deselects (hides its output + ROI).
 *
 * Popouts are HORIZONTAL — params sit side-by-side so the panel grows WIDER, not
 * taller (Find Vectors' five params used to make a tall popout that ran off the
 * MDI). They open below the bar, but flip ABOVE when the window is near the
 * screen bottom so they're never clipped. Hovering an action highlights it.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import type { ToolbarAction, ParamSpec, SubAction } from '../kernel/SpyDEContext'
import { OrientationWizard } from './OrientationWizard'
import { FindVectorsWizard } from './FindVectorsWizard'
import { VectorOrientationWizard } from './VectorOrientationWizard'
import { CenterZeroBeamWizard } from './CenterZeroBeamWizard'

const WIZARD_ACTIONS = new Set([
  'Orientation Mapping', 'Find Diffraction Vectors', 'Vector Orientation Mapping',
  'Center Zero Beam',
])

/**
 * Turn an OS filesystem path into a valid file:// URL. The Python backend sends
 * native icon paths, which on Windows are `C:\…\icon.svg` — a bare
 * `file://C:\…` is malformed (backslashes + drive letter), so the icons never
 * loaded there. Normalise separators and prefix the drive with the extra slash
 * Windows file URLs require.
 */
function fileUrl(p: string): string {
  const fwd = p.replace(/\\/g, '/')
  // Windows absolute path (drive letter) → file:///C:/…  ; POSIX → file:///…
  return /^[a-zA-Z]:\//.test(fwd) ? `file:///${fwd}` : `file://${fwd}`
}

// Actions that draw a live marker overlay on the DP. The overlay is shown only
// while the action's caret is selected, and hidden when it's deselected.
const OVERLAY_ACTIONS = new Set([
  'Find Diffraction Vectors', 'Orientation Mapping', 'Vector Orientation Mapping',
])

const EMPTY = new Set<string>()
const HIDDEN_ACTIONS = new Set(['Reset', 'Zoom In', 'Zoom Out'])

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
}: Props) {
  const { state, sendAction } = useSpyDE()
  const [openName, setOpenName] = React.useState<string | null>(null)
  const [openUp, setOpenUp] = React.useState(false)
  const rootRef = React.useRef<HTMLDivElement>(null)
  const live = state.activeActions.get(windowId) ?? EMPTY

  // Keep the toolbar shown while a popout/caret is open or an action is live —
  // otherwise reveal only on hover (over the window or the toolbar).
  const forced = openName !== null || live.size > 0
  const shownVisible = visible || forced

  React.useEffect(() => {
    if (!openName) return
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

  const click = (a: ToolbarAction, e: React.MouseEvent) => {
    if (live.has(a.name)) {
      sendAction('set_action_active', { name: a.name, active: false }, windowId)
      setOpenName(null)
      return
    }
    // Wizard actions (e.g. Vector Orientation Mapping) carry no params/subs but
    // still open a caret, so never dispatch them as a plain toolbar action.
    if (!hasPopout(a) && !WIZARD_ACTIONS.has(a.name)) {
      onAction(a.name, windowId, {}); setOpenName(null); return
    }
    // Flip the popout above the bar if there isn't room below on screen.
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    setOpenUp(r.bottom + 150 > window.innerHeight)
    setOpenName(openName === a.name ? null : a.name)
  }

  const openAction = shown.find(a => a.name === openName) || null

  return (
    <div
      ref={rootRef}
      data-testid="floating-toolbar"
      onMouseEnter={onHoverShow}
      onMouseLeave={onHoverHide}
      style={{
        ...styles.bar,
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
          style={(openName === a.name || live.has(a.name)) ? styles.btnActive : undefined}
          onClick={(e) => click(a, e)}
        >
          {a.icon && a.icon.endsWith('.svg')
            ? <img src={fileUrl(a.icon)} width={18} height={18} alt={a.name} />
            : <span style={{ fontSize: 10 }}>{a.name.slice(0, 4)}</span>}
        </button>
      ))}

      {openAction && openAction.name === 'Orientation Mapping' && (
        <OrientationWizard
          openUp={openUp} windowId={windowId} sendAction={sendAction}
          onClose={() => setOpenName(null)}
        />
      )}
      {openAction && openAction.name === 'Find Diffraction Vectors' && (
        <FindVectorsWizard
          openUp={openUp} windowId={windowId} sendAction={sendAction}
          onClose={() => setOpenName(null)}
        />
      )}
      {openAction && openAction.name === 'Vector Orientation Mapping' && (
        <VectorOrientationWizard
          openUp={openUp} windowId={windowId} sendAction={sendAction}
          onClose={() => setOpenName(null)}
        />
      )}
      {openAction && openAction.name === 'Center Zero Beam' && (
        <CenterZeroBeamWizard
          openUp={openUp} windowId={windowId} sendAction={sendAction}
          onClose={() => setOpenName(null)}
        />
      )}
      {openAction && !WIZARD_ACTIONS.has(openAction.name) && hasParams(openAction) && (
        <ParamPopout
          action={openAction} openUp={openUp}
          onRun={(params) => { onAction(openAction.name, windowId, params); setOpenName(null) }}
          onClose={() => setOpenName(null)}
        />
      )}
      {openAction && !hasParams(openAction) && hasSubs(openAction) && (
        <SubToolbar
          action={openAction}
          items={state.subItems.get(windowId)?.get(openAction.name) ?? []}
          onSub={(sub) => { onAction(sub.name, windowId, defaultsOf(sub.parameters)) }}
          onUpdate={(name, params) => sendAction('update_vi', { name, params }, windowId)}
          onRemove={(itemName) => sendAction('set_action_active', { name: itemName, active: false }, windowId)}
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

function ParamPopout({ action, openUp, onRun, onClose }: {
  action: ToolbarAction
  openUp: boolean
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
      style={{ ...(openUp ? styles.popoutUp : styles.popout), flexDirection: 'column', alignItems: 'stretch' }}>
      <div style={openUp ? styles.caretDown : styles.caretUp} />
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

/** The second floating toolbar (Qt PopoutToolBar), always BELOW the main bar: a
 *  "＋" to add, then one colour-coded detector-shape icon per virtual image.
 *  Clicking a VI icon opens its own parameter caret (detector type / calc). */
function SubToolbar({ action, items, onSub, onUpdate, onRemove }: {
  action: ToolbarAction
  items: ViItem[]
  onSub: (sub: SubAction) => void
  onUpdate: (name: string, params: Record<string, unknown>) => void
  onRemove: (name: string) => void
}) {
  const [openVi, setOpenVi] = React.useState<string | null>(null)
  const subs = action.subfunctions || []
  const paramSpec = subs[0]?.parameters ?? {}

  return (
    <div data-testid="sub-toolbar" style={styles.subBar}>
      <div style={styles.caretUpSub} />
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
function ViCaret({ item, spec, onChange, onRemove }: {
  item: ViItem
  spec: Record<string, ParamSpec>
  onChange: (params: Record<string, unknown>) => void
  onRemove: () => void
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
    // BELOW the window (the window root is overflow:visible), centered, tracking
    // it on move/resize. Reveal-on-hover is handled by the opacity/pointerEvents
    // applied inline.
    position: 'absolute', top: '100%', marginTop: 8, left: '50%',
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
  popout: { ...popBase, top: '100%', marginTop: 10, animation: 'spyde-pop 130ms ease-out' },
  popoutUp: { ...popBase, bottom: '100%', marginBottom: 10, animation: 'spyde-pop-up 130ms ease-out' },
  subBar: { ...subBase, top: '100%', marginTop: 10, animation: 'spyde-pop 130ms ease-out' },
  subBarUp: { ...subBase, bottom: '100%', marginBottom: 10, animation: 'spyde-pop-up 130ms ease-out' },
  caretUp: caret('up', POP_BG),
  caretDown: caret('down', POP_BG),
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
