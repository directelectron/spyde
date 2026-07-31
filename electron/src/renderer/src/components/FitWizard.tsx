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
import { useSpyDE } from '../kernel/SpyDEContext'

const TABS = ['Model', 'Run'] as const
type Tab = typeof TABS[number]

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

interface ParamState { name: string; value: number; free: boolean; linear: boolean }
interface CompState {
  name: string; kind: string; active: boolean; parameters: ParamState[]
  /** Peak height relative to the tallest component — see the header label. */
  share?: number
}
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
  const { state } = useSpyDE()
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
  const [coverage, setCoverage] = React.useState({ done: 0, total: 0, here: false })
  // How many positions fit worse than their neighbours — the number the
  // "Refit poor" button acts on, and the honest headline for a scan fit.
  const [poor, setPoor] = React.useState(0)
  // Models already stored on the signal. These are HyperSpy's own — `m.store`
  // puts the components AND every position's fit into the signal's `models`,
  // so they travel with the dataset rather than in a format of ours.
  const [storedModels, setStoredModels] = React.useState<string[]>([])
  const [modelName, setModelName] = React.useState('spyde fit')
  // The elements set in Plot Control's Composition panel — what "From …"
  // builds a model for. Read from the shared state rather than asked for, so
  // the button appears the moment they are set.
  const elements = state.composition.get(windowId)?.elements ?? []
  // Read inside the navigator listener, which is registered once — a state
  // value captured there would be the value at registration forever.
  const adaptiveRef = React.useRef(adaptive)
  adaptiveRef.current = adaptive

  // ── navigator coalescer: one `fit_navigated` in flight, latest wins ─────
  // A drag posts a pointer_move per frame and each one costs a recall, a
  // preview redraw and a full model re-send. Sending one per frame queues work
  // faster than it is served, and the queue drains AFTER the drag: measured 29
  // moves over 486 ms, with the model settling 1159 ms after the last of them.
  // That is the pause, and the value finally arriving is the snap.
  //
  // Skipping intermediate sends loses nothing: `fit_navigated` reads the
  // CURRENT selector position when the backend handles it, so one send after
  // the previous finishes always acts on the newest position. `fit_state`
  // coming back is the completion signal — self-tuning, unlike a fixed
  // throttle, and there is a timeout so a lost reply cannot wedge the gate.
  const inFlight = React.useRef(false)
  const pending = React.useRef(false)
  const wedgeTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  const sendNavigated = React.useCallback(() => {
    if (inFlight.current) { pending.current = true; return }
    inFlight.current = true
    pending.current = false
    if (wedgeTimer.current) clearTimeout(wedgeTimer.current)
    wedgeTimer.current = setTimeout(() => { inFlight.current = false }, 2_000)
    sendAction('fit_navigated', { adaptive: adaptiveRef.current }, windowId)
  }, [sendAction, windowId])

  const navDone = React.useCallback(() => {
    inFlight.current = false
    if (wedgeTimer.current) { clearTimeout(wedgeTimer.current); wedgeTimer.current = null }
    if (pending.current) sendNavigated()
  }, [sendNavigated])

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
    const d = detail as {
      components?: CompState[]; fitted?: boolean; status?: string
      fitted_count?: number; nav_total?: number; position_fitted?: boolean
      poor_count?: number; stored_models?: string[]
    }
    if (d.components) setComponents(d.components)
    setFitted(Boolean(d.fitted))
    setPoor(d.poor_count ?? 0)
    if (d.stored_models) setStoredModels(d.stored_models)
    // Every backend edit ends in a `fit_state`, so this is the completion
    // signal the navigator coalescer waits on — see sendNavigated.
    navDone()
    // TEST SEAM. `fit_navigated` pushes the model overlay (draw_preview) and
    // THEN emits this state, both down the same ordered stdout protocol — so
    // the arrival of a fit_state proves this position's overlay has already
    // landed. That makes this counter the only sound "the curves are now this
    // position's" signal available to e2e; everything else is a guess about
    // quiescence, and a stale overlay is perfectly quiescent.
    const w = window as unknown as { _spyde_fit_state_seq?: number }
    w._spyde_fit_state_seq = (w._spyde_fit_state_seq ?? 0) + 1
    setCoverage({
      done: d.fitted_count ?? 0,
      total: d.nav_total ?? 0,
      here: Boolean(d.position_fitted),
    })
    if (d.status) setStatus(d.status)
  })

  // ── follow the navigator ────────────────────────────────────────────────
  // The navigator's crosshair lives on a DIFFERENT window, so this listens to
  // `spyde:figure_event` unfiltered and ignores anything from OUR OWN figures
  // — `useWizardEvent` filters by window_id, which would drop exactly the
  // events wanted here.
  //
  // The filter is by figId, and it has to be: `spyde:figure_event` carries
  // `{figId, event}` and NO window_id at all (SpyDEContext's re-broadcast), so
  // the old `detail.window_id === windowId` test compared undefined and never
  // fired. Every pointer_up on this window's own plot therefore sent
  // `fit_navigated` — including the ones from the caret's own drag handles.
  // Before a fit that was harmless; after one the position is in the store, so
  // `fit_navigated` RECALLED it and overwrote the drag the instant it landed.
  // That is "once you fit a spectrum you can't move either component".
  const ownFigIds = React.useRef<Set<string>>(new Set())
  ownFigIds.current = new Set(
    (state.windows.get(windowId)?.figures ?? []).map((f) => f.figId))

  React.useEffect(() => {
    const on = (e: Event) => {
      const d = (e as CustomEvent).detail as { figId?: string }
      if (d.figId && ownFigIds.current.has(d.figId)) return   // our own plot
      sendNavigated()
    }
    window.addEventListener('spyde:figure_event', on)
    return () => {
      window.removeEventListener('spyde:figure_event', on)
      if (wedgeTimer.current) clearTimeout(wedgeTimer.current)
    }
  }, [sendNavigated])

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
                    {/* How TALL this component is, relative to the tallest.
                        The value box shows a gaussian's `A`, which is its
                        AREA — so a peak one sixth the height of its neighbour
                        but a tenth as wide reads as "888 next to 57471", i.e.
                        as a component suppressed to zero. It is not; this says
                        so. Amber below 2%, where it really has died. */}
                    {c.share !== undefined && (
                      <span data-testid={`fit-share-${c.name}`}
                        title="peak height, relative to the tallest component"
                        style={{ ...S.lbl, marginLeft: 6,
                                 color: c.share < 0.02 ? '#f5a97f' : '#a6adc8' }}>
                        {c.share >= 0.995 ? '100'
                          : c.share < 0.001 ? '<0.1' : (c.share * 100).toFixed(1)}% tall
                      </span>
                    )}
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
              {/* Build the whole model from the elements in Plot Control's
                  Composition panel: an EELS signal gets a background and an
                  edge per subshell, an EDS one a background and a gaussian per
                  X-ray line. Everything after that is the ordinary caret — the
                  drag handles work on an edge exactly as on a hand-placed
                  gaussian. Shown only when there ARE elements; without them
                  the backend has nothing to build from. */}
              {elements.length > 0 && (
                <button data-testid="fit-from-composition" style={S.primary}
                  title={`Build a model for ${elements.join(', ')} — replaces the current components`}
                  onClick={() => {
                    setStatus(`Building a model for ${elements.join(', ')}…`)
                    sendAction('fit_from_composition', { elements }, windowId)
                  }}>
                  From {elements.join(', ')}
                </button>
              )}
              {/* Refit automatically as the navigator moves. Each position's
                  answer is remembered, so scrubbing back shows what was found
                  there rather than the last pixel's model. */}
              <Check testid="fit-adaptive" checked={adaptive}
                onChange={setAdaptive} label="Adaptive" />
              {/* Coverage, so a skipped position is visible rather than
                  something you discover when a map comes out patchy. */}
              {coverage.total > 0 && (
                <span data-testid="fit-coverage" style={{ ...S.lbl,
                  color: coverage.here ? '#a6da95' : '#a6adc8' }}>
                  {coverage.here ? '●' : '○'} {coverage.done}/{coverage.total} fitted
                </span>
              )}
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
            <div style={{ display: 'flex', alignItems: 'center', gap: 8,
                          flexWrap: 'wrap' }}>
              <button data-testid="fit-run" style={S.primary}
                disabled={components.length === 0}
                title="Fit every spectrum in the dataset; the maps open when it finishes"
                onClick={run}>Fit all Spectra</button>
              {/* Refitting the poor positions runs automatically at the end of
                  a fit. This is the button for doing it again — after changing
                  the model, or to push harder on what is left. */}
              {fitted && (
                <button data-testid="fit-refit-poor" style={S.primary}
                  title="Restart the positions that fit worse than their neighbours from the best neighbour's answer"
                  onClick={() => {
                    setStatus('Refitting the poor positions…')
                    sendAction('fit_refit_poor', {}, windowId)
                  }}>
                  Refit poor{poor > 0 ? ` (${poor})` : ''}
                </button>
              )}
              {fitted && (
                <CommitButton wizardKey="fit" windowId={windowId} sendAction={sendAction}
                  label="Commit components" />
              )}
            </div>

            {/* ── save / load, through HyperSpy's own model store ──
                `m.store(name)` puts the components AND every position's fit
                into the signal's `models`, so saving the .hspy/.zspy saves the
                fit with it. Nothing here is a SpyDE format. */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6,
                          flexWrap: 'wrap', borderTop: '1px solid #313244',
                          paddingTop: 6 }}>
              <span style={S.lbl}>Model</span>
              <input data-testid="fit-model-name" value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                style={{ background: '#11111b', border: '1px solid #313244',
                         borderRadius: 4, color: '#cdd6f4', fontSize: 11,
                         padding: '2px 5px', width: 110 }} />
              <button data-testid="fit-save-model" style={S.primary}
                title="Store this model on the signal — it saves with the dataset"
                onClick={() => sendAction('fit_save_model', { name: modelName }, windowId)}>
                Save
              </button>
              {storedModels.length > 0 && (
                <>
                  <Select testid="fit-stored-models" value={modelName}
                    options={storedModels.map((n) => ({ value: n, label: n }))}
                    onChange={(v) => setModelName(v)} />
                  <button data-testid="fit-load-model" style={S.primary}
                    title="Restore a stored model, per-position fits and all"
                    onClick={() => sendAction('fit_load_model', { name: modelName }, windowId)}>
                    Load
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </WizardShell>
  )
}
