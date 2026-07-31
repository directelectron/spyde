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
/** One offerable exspy core-loss edge — see spyde/spectroscopy/edges.py. */
interface EdgeItem {
  subshell: string; element: string; shell: string; onset: number
  relevance: string; description: string
  /** Belongs to an element the user declared in the Composition panel. */
  suggested: boolean
}
interface EelsOffer {
  eels: boolean; exspy: boolean; edges: EdgeItem[]
  microscope_missing: string[]; install_hint: string
}
const NO_EELS: EelsOffer = {
  eels: false, exspy: false, edges: [], microscope_missing: [], install_hint: '',
}

/** How many non-suggested edges the picker lists before asking for a filter.
 *  The list scrolls inside its own box, so this only bounds how many buttons
 *  are rendered — a 600 eV window contains ~64 major edges and all of them are
 *  worth having, since without a composition the filter is the only way in. */
const EDGE_LIMIT = 80

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
  const [eels, setEels] = React.useState<EelsOffer>(NO_EELS)
  const [edgeFilter, setEdgeFilter] = React.useState('')
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
    const d = detail as Partial<EelsOffer> & { components?: CatalogueItem[] }
    if (d.components) setCatalogue(d.components)
    setEels({
      eels: Boolean(d.eels),
      exspy: Boolean(d.exspy),
      edges: d.edges ?? [],
      microscope_missing: d.microscope_missing ?? [],
      install_hint: d.install_hint ?? '',
    })
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

  /** An exspy core-loss edge takes its subshell — there is no bare EELSCLEdge. */
  const addEdge = (subshell: string) => {
    setStatus(`Adding the ${subshell} edge…`)
    sendAction('fit_add_component',
      { kind: 'EELSCLEdge', element_subshell: subshell }, windowId)
    setPickerOpen(false)
  }

  // Suggested edges (the elements in the Composition panel) are ALWAYS listed:
  // "I know there is oxygen here, give me O-K" is the common case, and hunting
  // for it among the ~130 edges a 600 eV window contains is not. The rest are
  // shown major-first and capped, until a filter narrows them.
  const q = edgeFilter.trim().toLowerCase()
  const suggestedEdges = eels.edges.filter((e) => e.suggested)
  const otherEdges = eels.edges
    .filter((e) => !e.suggested)
    .filter((e) => (q
      ? e.subshell.toLowerCase().includes(q) || e.element.toLowerCase() === q
      : e.relevance === 'Major'))
  const shownEdges = otherEdges.slice(0, EDGE_LIMIT)
  const edgesBlocked = eels.microscope_missing.length > 0

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
                    // The popup hangs BELOW an already-low button row, so its
                    // height is the caret-overhang budget: taller than this and
                    // it runs under the status bar. The edge list scrolls
                    // inside itself rather than growing this box.
                    width: 340, maxHeight: 300, overflowY: 'auto',
                  }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr',
                                gap: 3 }}>
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

                  {/* ── EELS core-loss edges ──
                      Only on an EELS signal, because EELSCLEdge is the one
                      component that is not a bare `Kind()`: it takes a
                      subshell, so it is one button per edge rather than one
                      button for the kind. */}
                  {eels.eels && (
                    <div data-testid="fit-edge-section"
                      style={{ marginTop: 6, borderTop: '1px solid #313244',
                               paddingTop: 5 }}>
                      <div style={{ display: 'flex', alignItems: 'center',
                                    gap: 6, marginBottom: 4 }}>
                        <span style={{ ...S.lbl, fontWeight: 600,
                                       color: '#cdd6f4' }}>EELS edges</span>
                        {eels.exspy && (
                          <input data-testid="fit-edge-filter" value={edgeFilter}
                            placeholder="filter, e.g. Fe"
                            onChange={(e) => setEdgeFilter(e.target.value)}
                            style={{ flex: 1, background: '#11111b',
                                     border: '1px solid #313244',
                                     borderRadius: 4, color: '#cdd6f4',
                                     fontSize: 11, padding: '2px 5px' }} />
                        )}
                      </div>

                      {/* exspy is an optional extra, so say which one and how
                          — an ImportError on a package the user has never
                          heard of is not actionable. */}
                      {!eels.exspy && (
                        <span data-testid="fit-edge-no-exspy"
                          style={{ ...S.hint, color: '#f5a97f' }}>
                          EELS edges need exspy — {eels.install_hint}
                        </span>
                      )}

                      {/* An edge integrates its cross-section at an effective
                          angle derived from these, so exspy raises deep inside
                          model.append without them. Name the fields instead. */}
                      {eels.exspy && edgesBlocked && (
                        <span data-testid="fit-edge-no-microscope"
                          style={{ ...S.hint, color: '#f5a97f' }}>
                          Set the {eels.microscope_missing.join(', ')} in the
                          {' '}Metadata panel before adding an edge.
                        </span>
                      )}

                      {eels.exspy && !edgesBlocked && eels.edges.length === 0 && (
                        <span style={S.hint}>
                          No tabulated edge onsets fall inside this energy range.
                        </span>
                      )}

                      {eels.exspy && !edgesBlocked && (
                        <div data-testid="fit-edge-list"
                          style={{ display: 'grid',
                                   gridTemplateColumns: '1fr 1fr', gap: 3,
                                   maxHeight: 132, overflowY: 'auto' }}>
                          {[...suggestedEdges, ...shownEdges].map((e) => (
                            <button key={e.subshell}
                              data-testid={`fit-add-edge-${e.subshell}`}
                              title={e.description}
                              onClick={() => addEdge(e.subshell)}
                              style={{
                                display: 'flex', alignItems: 'baseline', gap: 6,
                                background: '#11111b',
                                // Composition-seeded edges read as the
                                // recommendation they are.
                                border: `1px solid ${e.suggested ? '#89b4fa' : '#313244'}`,
                                borderRadius: 4, padding: '3px 6px',
                                color: '#cdd6f4', fontSize: 11,
                                cursor: 'pointer', textAlign: 'left',
                              }}>
                              <span style={{ flex: 1 }}>
                                {e.element} {e.shell}
                              </span>
                              <span style={{ color: '#a6adc8', fontSize: 10 }}>
                                {e.onset.toFixed(0)} eV
                              </span>
                            </button>
                          ))}
                        </div>
                      )}

                      {eels.exspy && !edgesBlocked
                        && otherEdges.length > shownEdges.length && (
                        <span style={S.hint}>
                          {otherEdges.length - shownEdges.length} more — type an
                          element symbol to narrow the list.
                        </span>
                      )}
                    </div>
                  )}
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
