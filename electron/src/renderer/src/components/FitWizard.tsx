/**
 * FitWizard.tsx — the Fit caret (#55, #56, #58).
 *
 * Two tabs, matching the Orientation-Mapping caret's shape: **Model** builds
 * the model component by component, **Run** fits it and commits the maps.
 *
 * Layout is HORIZONTAL by preference (the same rule the toolbar popouts
 * follow): a component's parameters sit side by side so the caret grows WIDER
 * rather than taller. A tall caret is not just ugly here — it overhangs the
 * room FloatingToolbar placed it in, and the overhang sits under the MDI area.
 *
 * Three deliberate choices:
 *
 * - **Components are added from a `+` POPUP**, not an always-open palette. The
 *   palette is a one-off action; leaving it permanently expanded spends the
 *   caret's height on something used once per component.
 * - **The popup shows SHAPES, not just names** (#56). The backend samples every
 *   offerable component at defaults over the current signal axis and sends a
 *   normalised polyline, drawn as an inline SVG sparkline. "Gaussian" and
 *   "Lorentzian" are not distinguishable by name to someone who has not met
 *   them, and the point of a picker is to be able to choose.
 * - **The component list is rebuilt from `fit_state`, never edited locally.**
 *   The backend owns the model; a caret that mutated its own copy would drift
 *   out of step the moment an edit was rejected and would then show a model
 *   that is not the one being fitted.
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Select, Check, S } from './WizardShell'
import { useWizardLifecycle, useWizardEvent, CommitButton } from './wizardHooks'

const TABS = ['Model', 'Run'] as const
type Tab = typeof TABS[number]

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

interface ParamState { name: string; value: number; free: boolean; linear: boolean }
interface CompState { name: string; kind: string; active: boolean; parameters: ParamState[] }
interface CatalogueItem { kind: string; description: string; preview: number[] }

/** Inline sparkline of a component's shape — the picker's whole point. */
function Spark({ points }: { points: number[] }) {
  if (!points?.length) return null
  const w = 40, h = 15
  const d = points
    .map((v, i) => `${(i / (points.length - 1)) * w},${h - v * (h - 2) - 1}`)
    .join(' ')
  return (
    <svg width={w} height={h} style={{ flex: '0 0 auto' }} aria-hidden>
      <polyline points={d} fill="none" stroke="#89b4fa" strokeWidth="1.5" />
    </svg>
  )
}

export function FitWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const [tab, setTab] = React.useState<Tab>('Model')
  const [catalogue, setCatalogue] = React.useState<CatalogueItem[]>([])
  const [components, setComponents] = React.useState<CompState[]>([])
  const [fitted, setFitted] = React.useState(false)
  const [status, setStatus] = React.useState('Add a component to begin.')
  const [pickerOpen, setPickerOpen] = React.useState(false)
  const [maxIter, setMaxIter] = React.useState(60)
  const [seeded, setSeeded] = React.useState(true)
  const [weighting, setWeighting] = React.useState<'none' | 'poisson'>('none')
  const [adaptive, setAdaptive] = React.useState(false)
  // Read inside the navigator listener, which is registered once — a state
  // value captured there would be the value at registration forever.
  const adaptiveRef = React.useRef(adaptive)
  adaptiveRef.current = adaptive

  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'fit_open',
    closeAction: 'fit_close',
    deps: [windowId],
  })

  useWizardEvent('spyde:fit_catalogue', windowId, (detail) => {
    const d = detail as { components?: CatalogueItem[] }
    if (d.components) setCatalogue(d.components)
  })

  useWizardEvent('spyde:fit_state', windowId, (detail) => {
    const d = detail as { components?: CompState[]; fitted?: boolean; status?: string }
    if (d.components) setComponents(d.components)
    setFitted(Boolean(d.fitted))
    if (d.status) setStatus(d.status)
  })

  // ── follow the navigator ────────────────────────────────────────────────
  // The navigator's crosshair lives on a DIFFERENT window, so this listens to
  // `spyde:figure_event` unfiltered and ignores anything from this window's own
  // figures — `useWizardEvent` filters by window_id, which would drop exactly
  // the events wanted here.
  //
  // Fires on pointer_up, not pointer_move: a fit per pointer frame while
  // dragging the navigator would queue work faster than it completes.
  React.useEffect(() => {
    const on = (e: Event) => {
      const d = (e as CustomEvent).detail as
        { window_id?: number; event?: { type?: string } }
      if (d.window_id === windowId) return          // our own plot, not the nav
      if (d.event?.type && !/up|click/i.test(String(d.event.type))) return
      sendAction('fit_navigated', { adaptive: adaptiveRef.current }, windowId)
    }
    window.addEventListener('spyde:figure_event', on)
    return () => window.removeEventListener('spyde:figure_event', on)
  }, [windowId, sendAction])

  const add = (kind: string) => {
    sendAction('fit_add_component', { kind }, windowId)
    setPickerOpen(false)
  }

  const setParam = (component: string, parameter: string, value: number) =>
    sendAction('fit_set_param', { component, parameter, value }, windowId)

  const toggleFree = (component: string, parameter: string, free: boolean) =>
    sendAction('fit_set_param', { component, parameter, free }, windowId)

  const run = () => {
    setStatus('Fitting…')
    setTab('Run')
    sendAction('fit_run', { max_iter: maxIter, seeded, weighting }, windowId)
  }

  return (
    <WizardShell testid="fit-wizard" title="Fit" posStyle={caretPos} width={430}
      onClose={onClose} closeTestid="fit-close" status={status}
      statusTestid="fit-status">
      {/* Keep mousedown INSIDE the caret. Letting it bubble to the window
          re-renders the subwindow subtree, which replaces the control being
          clicked between mousedown and mouseup — the mouseup then lands on the
          MDI area, React never sees a click, and the action silently does not
          happen. Measured with capture-phase listeners: before this,
          `mousedown -> the button's span, mouseup -> mdi-area`; after, all
          three land on the span. Only the running app finds this; the handler
          tests pass either way. */}
      <div onMouseDown={(e) => e.stopPropagation()}>
        <TabRow tabs={TABS} active={tab} onSelect={setTab}
          testid={(t) => `fit-tab-${t}`} />

        {tab === 'Model' && (
          <div style={S.page}>
            {components.length === 0 && (
              <span style={S.hint}>No components yet — add one with +.</span>
            )}

            {/* One card per component; its parameters lie side by side so the
                caret grows wider, not taller. */}
            <div data-testid="fit-components"
              style={{ display: 'flex', flexDirection: 'column', gap: 4,
                       maxHeight: 210, overflowY: 'auto' }}>
              {components.map((c) => (
                <div key={c.name} data-testid={`fit-comp-${c.name}`}
                  style={{ background: '#11111b', borderRadius: 4, padding: '4px 6px' }}>
                  <div style={S.fieldRow}>
                    <span style={{ ...S.lbl, color: '#cdd6f4', fontWeight: 600 }}>{c.name}</span>
                    <button data-testid={`fit-remove-${c.name}`} style={S.close}
                      title="Remove component"
                      onClick={() => sendAction('fit_remove_component', { name: c.name }, windowId)}>✕</button>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 10px' }}>
                    {c.parameters.map((p) => (
                      <div key={p.name}
                        style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                        <span style={S.lbl}>{p.name}</span>
                        <NumInput testid={`fit-p-${c.name}-${p.name}`} value={p.value}
                          step="any" width={62}
                          onChange={(v) => setParam(c.name, p.name, v)} />
                        <input type="checkbox" checked={p.free} title="Fit this parameter"
                          data-testid={`fit-free-${c.name}-${p.name}`}
                          onChange={(e) => toggleFree(c.name, p.name, e.target.checked)} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* ── + picker and Fit spectrum, side by side ──
                "Fit spectrum" belongs HERE, not on the Run tab: building a
                model is a loop — add, look, nudge, fit this one, look again —
                and that loop happens entirely on this tab. Run scan is the
                separate, expensive thing you do once the model is right. */}
            <div style={{ position: 'relative', display: 'flex', gap: 8 }}>
              <button data-testid="fit-add-toggle" style={S.primary}
                onClick={() => setPickerOpen((v) => !v)}>+ Component</button>
              <button data-testid="fit-spectrum" style={S.primary}
                disabled={components.length === 0}
                title="Fit only the spectrum on screen — quick, for checking a guess"
                onClick={() => sendAction('fit_current', {}, windowId)}>
                Fit spectrum
              </button>
              {/* Refit automatically as the navigator moves. Each position's
                  answer is remembered, so scrubbing back shows what was found
                  there rather than the last pixel's model. */}
              <Check testid="fit-adaptive" checked={adaptive}
                onChange={setAdaptive} label="Adaptive" />
              {pickerOpen && (
                <div data-testid="fit-palette"
                  style={{
                    position: 'absolute', top: 30, left: 0, zIndex: 20,
                    background: '#181825', border: '1px solid #45475a',
                    borderRadius: 6, padding: 5, boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
                    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3,
                    width: 340,
                  }}>
                  {catalogue.map((c) => (
                    <button key={c.kind} data-testid={`fit-add-${c.kind}`}
                      title={c.description} onClick={() => add(c.kind)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 7,
                        background: '#11111b', border: '1px solid #313244',
                        borderRadius: 4, padding: '3px 6px', color: '#cdd6f4',
                        fontSize: 11, cursor: 'pointer', textAlign: 'left',
                      }}>
                      <Spark points={c.preview} />
                      <span style={{ flex: 1 }}>{c.kind}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {tab === 'Run' && (
          <div style={S.page}>
            {/* Horizontal: the three knobs sit on one row. */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
              <Field label="Max iterations">
                <NumInput testid="fit-max-iter" value={maxIter} step="5" width={58}
                  onChange={(v) => setMaxIter(Math.max(5, Math.round(v)))} />
              </Field>
              <Field label="Weighting">
                <Select testid="fit-weighting" value={weighting}
                  options={[{ value: 'none', label: 'None' },
                            { value: 'poisson', label: 'Poisson' }]}
                  onChange={(v) => setWeighting(v)} />
              </Field>
              <Check testid="fit-seeded" checked={seeded} onChange={setSeeded}
                label="Seed from a coarse grid" />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button data-testid="fit-run" style={S.primary}
                disabled={components.length === 0} onClick={run}>Run scan</button>
              {fitted && (
                <CommitButton wizardKey="fit" windowId={windowId} sendAction={sendAction}
                  label="Commit component maps" />
              )}
            </div>
          </div>
        )}
      </div>
    </WizardShell>
  )
}
