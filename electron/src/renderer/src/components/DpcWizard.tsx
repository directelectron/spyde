/**
 * DpcWizard.tsx — the DPC caret (`dpc_` staged actions; backend:
 * spyde/actions/dpc_action.py).
 *
 * Four tabs, walked left to right, mirroring the Orientation caret's shape:
 *
 *   1 Center   — find the undeflected beam. Manual (one crosshair for every
 *                pattern), Vacuum (a second dataset), or Corners (a plane
 *                through four boxes at the edges of the scan). The backend
 *                measures the residual descan on open, so a dataset that is
 *                ALREADY centred says so and the step can be skipped.
 *   2 Rotation — put the detector's x/y onto the scan's x/y. `Solve` fits it
 *                from the data; the slider overrides it; handedness and the
 *                unresolvable 180° are separate toggles.
 *   3 Field    — magnetic (stop at the deflection) or electric (calibrate to
 *                MV/cm, which needs thickness + beam energy + detector scale).
 *   4 Map      — which map to show, and Commit.
 *
 * **Everything except `Re-measure` is instant.** The backend caches the beam
 * positions from one pass at open; centering, rotation and calibration are then
 * arithmetic on a small array. So the rotation slider is a live control, not a
 * submit button — which matters, because judging a rotation by eye against the
 * colour wheel is the whole workflow.
 *
 * The `DEFAULTS` below mirror `dpc_action.DEFAULTS` key for key. A value that
 * drifts from the Python side WINS SILENTLY (see the caret-defaults trap in
 * CLAUDE.md), so `test_dpc_action.py` parses this file and compares.
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Slider, Select, Check, S } from './WizardShell'
import { useWizardLifecycle, useDebouncedAction, useWizardEvent, CommitButton } from './wizardHooks'
import type { SendAction } from './wizardHooks'

const TABS = ['Center', 'Rotation', 'Field', 'Map'] as const
type Tab = typeof TABS[number]

type CenterMode = 'none' | 'manual' | 'vacuum' | 'corners'
type FieldMode = 'magnetic' | 'electric'
type View = 'rgb' | 'fx' | 'fy' | 'magnitude' | 'phase' | 'divergence' | 'curl'

/** `dpc.CENTER_MODES`, in the order the backend docstring ranks them. */
const CENTER_OPTIONS: readonly { value: CenterMode; label: string }[] = [
  { value: 'corners', label: 'Corners (plane through 4 boxes)' },
  { value: 'vacuum', label: 'Vacuum reference (2nd dataset)' },
  { value: 'manual', label: 'Manual (one centre, crosshair)' },
  { value: 'none', label: 'None — already centered' },
]

/** `dpc.BEAM_METHODS`. */
const METHOD_OPTIONS: readonly { value: string; label: string }[] = [
  { value: 'center_of_mass', label: 'Centre of mass' },
  { value: 'blur', label: 'Blur' },
  { value: 'interpolate', label: 'Interpolate' },
  { value: 'cross_correlate', label: 'Cross-correlate' },
]

/** `dpc_display.VIEWS`. */
const VIEW_OPTIONS: readonly { value: View; label: string }[] = [
  { value: 'rgb', label: 'Direction + magnitude' },
  { value: 'fx', label: 'x component' },
  { value: 'fy', label: 'y component' },
  { value: 'magnitude', label: 'Magnitude' },
  { value: 'phase', label: 'Direction (rad)' },
  { value: 'divergence', label: 'Divergence' },
  { value: 'curl', label: 'Curl' },
]

/** `dpc.BEAM_SHAPES` — the beam region's shape, or off. */
type BeamShape = 'off' | 'circle' | 'ring'
const BEAM_SHAPES: readonly { value: BeamShape; label: string }[] = [
  { value: 'off', label: 'Off' },
  { value: 'circle', label: 'Circle' },
  { value: 'ring', label: 'Ring' },
]

/** Mirrors `dpc_action.DEFAULTS` — key for key, value for value. */
interface DpcSaved {
  method: string
  halfSquareWidth: number
  centerMode: CenterMode
  cornerFraction: number
  beamShape: BeamShape
  beamCx: number
  beamCy: number
  beamR: number
  beamRInner: number
  mode: FieldMode
  rotation: number
  flip: boolean
  reverse: boolean
  thicknessNm: number
  beamEnergyKv: number
  mradPerPx: number
  view: View
  autolimSigma: number
}
const DEFAULTS: DpcSaved = {
  method: 'center_of_mass',
  halfSquareWidth: 0,
  centerMode: 'corners',
  cornerFraction: 0.05,
  beamShape: 'off',
  beamCx: 0.0,
  beamCy: 0.0,
  beamR: 0.0,
  beamRInner: 0.0,
  mode: 'magnetic',
  rotation: 0.0,
  flip: false,
  reverse: false,
  thicknessNm: 60.0,
  beamEnergyKv: 200.0,
  mradPerPx: 0.0,
  view: 'rgb',
  autolimSigma: 4.0,
}

// Kept OUTSIDE the component so stepping away and back doesn't reset a solved
// rotation the user spent time on (same reason OrientationWizard caches its
// library state).
const _dpcStore = new Map<number, DpcSaved>()

interface Centering {
  offset: number[]
  ramp: number[]
  worst: number
  centered: boolean
  tol_px: number
}
interface Dataset { index: number; title: string }
interface Region {
  shape: BeamShape
  cx: number; cy: number; r: number; r_inner: number
  /** Region intensity density over the frame average; null when off. */
  brightness: number | null
}
interface State {
  measured: boolean
  navShape: number[] | null
  centering: Centering | null
  mradPerPx: number | null
  beamEnergyKv: number | null
  vacuum: string | null
  datasets: Dataset[]
}
interface Estimate {
  angle: number; flip: boolean; mode: string
  score: number; baseline: number; improvement: number
}
interface Result {
  units: string; mode: string; rotation: number; calibrated: boolean
  max: number; mean: number; divergence: number; curl: number
}

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: SendAction
  onClose: () => void
}

export function DpcWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _dpcStore.get(windowId) ?? DEFAULTS
  const [tab, setTab] = React.useState<Tab>('Center')
  const [method, setMethod] = React.useState(saved.method)
  const [halfSquareWidth, setHalfSquareWidth] = React.useState(saved.halfSquareWidth)
  const [centerMode, setCenterMode] = React.useState<CenterMode>(saved.centerMode)
  const [cornerFraction, setCornerFraction] = React.useState(saved.cornerFraction)
  const [mode, setMode] = React.useState<FieldMode>(saved.mode)
  const [rotation, setRotation] = React.useState(saved.rotation)
  const [flip, setFlip] = React.useState(saved.flip)
  const [reverse, setReverse] = React.useState(saved.reverse)
  const [thicknessNm, setThicknessNm] = React.useState(saved.thicknessNm)
  const [beamEnergyKv, setBeamEnergyKv] = React.useState(saved.beamEnergyKv)
  const [mradPerPx, setMradPerPx] = React.useState(saved.mradPerPx)
  const [view, setView] = React.useState<View>(saved.view)
  const [autolimSigma, setAutolimSigma] = React.useState(saved.autolimSigma)
  const [beamShape, setBeamShape] = React.useState<BeamShape>(saved.beamShape)
  const [beamCx, setBeamCx] = React.useState(saved.beamCx)
  const [beamCy, setBeamCy] = React.useState(saved.beamCy)
  const [beamR, setBeamR] = React.useState(saved.beamR)
  const [beamRInner, setBeamRInner] = React.useState(saved.beamRInner)

  const [state, setState] = React.useState<State | null>(null)
  const [estimate, setEstimate] = React.useState<Estimate | null>(null)
  const [result, setResult] = React.useState<Result | null>(null)
  const [vacuumPick, setVacuumPick] = React.useState<string>('')
  /** The picked beam centre the BACKEND confirmed, echoed on `dpc_state`. */
  const [center, setCenter2] = React.useState<[number, number] | null>(null)
  /** Live region geometry + brightness, echoed while the widget is dragged. */
  const [region, setRegion] = React.useState<Region | null>(null)
  const [status, setStatus] = React.useState('Locating the direct beam…')

  const vals = React.useRef<DpcSaved>(saved)
  vals.current = {
    method, halfSquareWidth, centerMode, cornerFraction,
    beamShape, beamCx, beamCy, beamR, beamRInner,
    mode, rotation, flip,
    reverse, thicknessNm, beamEnergyKv, mradPerPx, view, autolimSigma,
  }
  React.useEffect(() => { _dpcStore.set(windowId, vals.current) })

  /** The backend's parameter names (`dpc_action.DEFAULTS` keys). */
  const params = React.useCallback((): Record<string, unknown> => {
    const v = vals.current
    return {
      method: v.method, half_square_width: v.halfSquareWidth,
      center_mode: v.centerMode, corner_fraction: v.cornerFraction,
      beam_shape: v.beamShape, beam_cx: v.beamCx, beam_cy: v.beamCy,
      beam_r: v.beamR, beam_r_inner: v.beamRInner,
      mode: v.mode, rotation: v.rotation, flip: v.flip, reverse: v.reverse,
      thickness_nm: v.thicknessNm, beam_energy_kv: v.beamEnergyKv,
      mrad_per_px: v.mradPerPx, view: v.view, autolim_sigma: v.autolimSigma,
    }
  }, [])

  // Mount → dpc_open (measures the beam positions once and opens the result
  // window). Unmount → dpc_close. StrictMode-safe.
  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'dpc_open', openPayload: params, closeAction: 'dpc_close',
  })

  const sendTune = useDebouncedAction(sendAction, 'dpc_tune', windowId)
  const tune = () => sendTune(params)
  /** Set a value AND push it — every control on this caret is live. */
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); setTimeout(tune, 0) }

  useWizardEvent('spyde:dpc_state', windowId, (d) => {
    setState({
      measured: Boolean(d.measured),
      navShape: Array.isArray(d.nav_shape) ? (d.nav_shape as number[]).map(Number) : null,
      centering: (d.centering ?? null) as Centering | null,
      mradPerPx: d.mrad_per_px == null ? null : Number(d.mrad_per_px),
      beamEnergyKv: d.beam_energy_kv == null ? null : Number(d.beam_energy_kv),
      vacuum: (d.vacuum ?? null) as string | null,
      datasets: Array.isArray(d.datasets) ? (d.datasets as Dataset[]) : [],
    })
    // Adopt the dataset's own calibration as the starting point, so the
    // Electric tab opens with the right numbers instead of the placeholders.
    if (d.mrad_per_px != null && vals.current.mradPerPx === 0) {
      setMradPerPx(Number(d.mrad_per_px))
    }
    if (d.beam_energy_kv != null) setBeamEnergyKv(Number(d.beam_energy_kv))
    const p = (d.params ?? {}) as Record<string, unknown>
    setCenter2(p.cx == null || p.cy == null ? null : [Number(p.cx), Number(p.cy)])
    const c = d.centering as Centering | null
    if (c) {
      setStatus(c.centered
        ? `Already centered (${c.worst.toFixed(2)} px residual) — skip to Rotation.`
        : `Descan: ${c.worst.toFixed(2)} px. Pick how to remove it.`)
      // Nothing to remove → don't apply a correction for its own sake. The
      // backend has to be told: it opened on `corners`, so a caret that
      // switched only its own state would show `none` while the result it
      // displays and commits was still centred on the corner plane. Not
      // `setCenter`, which would overwrite the status line just set above.
      if (c.centered && vals.current.centerMode === DEFAULTS.centerMode) {
        setCenterMode('none')
        vals.current = { ...vals.current, centerMode: 'none' }
        sendAction('dpc_set_center', params(), windowId)
      }
    }
  })

  // The widget is the authority on its own geometry while it is being dragged,
  // so the backend echoes it back here rather than the caret trying to predict
  // it. Same shape as the Crop / Center-Zero-Beam drag→field round trip.
  useWizardEvent('spyde:dpc_region', windowId, (d) => {
    const r: Region = {
      shape: String(d.shape ?? 'off') as BeamShape,
      cx: Number(d.cx ?? 0), cy: Number(d.cy ?? 0),
      r: Number(d.r ?? 0), r_inner: Number(d.r_inner ?? 0),
      brightness: d.brightness == null ? null : Number(d.brightness),
    }
    setRegion(r)
    setBeamShape(r.shape); setBeamCx(r.cx); setBeamCy(r.cy)
    setBeamR(r.r); setBeamRInner(r.r_inner)
  })

  useWizardEvent('spyde:dpc_estimate', windowId, (d) => {
    const est = {
      angle: Number(d.angle), flip: Boolean(d.flip), mode: String(d.mode),
      score: Number(d.score), baseline: Number(d.baseline),
      improvement: Number(d.improvement),
    }
    setEstimate(est)
    setRotation(est.angle)
    setFlip(est.flip)
    setStatus(`Rotation ${est.angle.toFixed(1)}°${est.flip ? ' (flipped)' : ''}`)
  })

  useWizardEvent('spyde:dpc_result', windowId, (d) => {
    setResult({
      units: String(d.units), mode: String(d.mode), rotation: Number(d.rotation),
      calibrated: Boolean(d.calibrated), max: Number(d.max), mean: Number(d.mean),
      divergence: Number(d.divergence), curl: Number(d.curl),
    })
  })

  const setCenter = (m: CenterMode) => {
    setCenterMode(m)
    vals.current = { ...vals.current, centerMode: m }
    sendAction('dpc_set_center', params(), windowId)
    setStatus(m === 'manual'
      ? 'Drag the teal crosshair onto the undeflected beam, then Use this centre.'
      : m === 'vacuum'
        ? 'Pick the vacuum dataset acquired with the same scan settings.'
        : m === 'corners'
          ? 'The four boxes on the navigator are what the plane is fitted to.'
          : 'No centering applied.')
  }

  const remeasure = () => {
    setStatus('Re-locating the direct beam…')
    sendAction('dpc_run', params(), windowId)
  }
  // The backend's `emit_status` reaches the app's STATUS BAR, not this caret's
  // footer — so a bare dispatch left "Use this centre" looking like it had done
  // nothing. The confirmed position comes back on `dpc_state.params` and is
  // rendered under the button; this is the immediate acknowledgement.
  const useCrosshair = () => {
    setStatus('Using the crosshair position as the beam centre…')
    sendAction('dpc_pick_center', params(), windowId)
  }
  const solveRotation = () => {
    setStatus('Solving the scan↔detector rotation…')
    sendAction('dpc_auto_rotation', params(), windowId)
  }
  const loadVacuumFile = async () => {
    const path = await window.electron.pickFile({
      name: 'Vacuum scan', extensions: ['hspy', 'zspy', 'mrc', 'tif', 'tiff', 'de5'],
    })
    if (path) sendAction('dpc_load_vacuum', { path }, windowId)
  }
  const loadVacuumTree = (v: string) => {
    setVacuumPick(v)
    if (v !== '') sendAction('dpc_load_vacuum', { tree_index: Number(v) }, windowId)
  }
  /** Toggling the shape (or typing a radius) re-measures — the region changes
   *  the centre of mass, so it cannot take effect any other way. A DRAG is
   *  debounced on the backend instead. */
  const setBeam = (patch: Partial<DpcSaved>) => {
    Object.assign(vals.current, patch)
    if (patch.beamShape !== undefined) setBeamShape(patch.beamShape)
    if (patch.beamR !== undefined) setBeamR(patch.beamR)
    if (patch.beamRInner !== undefined) setBeamRInner(patch.beamRInner)
    sendAction('dpc_set_beam', params(), windowId)
  }

  const changeView = (v: View) => {
    setView(v)
    vals.current = { ...vals.current, view: v }
    sendAction('dpc_set_view', { view: v }, windowId)
  }

  const c = state?.centering ?? null
  const datasetOptions = [
    { value: '', label: state?.datasets.length ? 'Choose an open dataset…' : 'No other datasets open' },
    ...(state?.datasets ?? []).map(d => ({ value: String(d.index), label: d.title })),
  ]

  return (
    <WizardShell testid="dpc-wizard" title="DPC" posStyle={caretPos}
      onClose={onClose} closeTestid="dpc-close" status={status}
      statusTestid="dpc-status">
      <TabRow tabs={TABS} active={tab} onSelect={setTab}
        testid={(t) => `dpc-tab-${t}`}
        locked={(t) => t !== 'Center' && !(state?.measured ?? false)} />

      {tab === 'Center' && (
        <div style={S.page}>
          <CenteringReadout centering={c} />
          <Field label="Reference">
            <Select testid="dpc-center-mode" value={centerMode}
              options={CENTER_OPTIONS} onChange={setCenter} />
          </Field>

          {centerMode === 'corners' && (
            <Field label={`Box size${state?.navShape
              ? ` (${cornerBoxSize(state.navShape, cornerFraction)})` : ''}`}>
              <Slider testid="dpc-corner-fraction" value={cornerFraction}
                min={0.01} max={0.45} step={0.01}
                fmt={(n) => `${Math.round(n * 100)}%`}
                onChange={(n) => {
                  setCornerFraction(n)
                  vals.current = { ...vals.current, cornerFraction: n }
                  sendAction('dpc_set_center', params(), windowId)
                }} />
            </Field>
          )}

          {centerMode === 'vacuum' && (
            <>
              <Field label="From open">
                <Select testid="dpc-vacuum-tree" value={vacuumPick}
                  options={datasetOptions} onChange={loadVacuumTree} />
              </Field>
              <button data-testid="dpc-vacuum-file" style={S.fileBtn}
                onClick={loadVacuumFile}>＋ From file…</button>
              {state?.vacuum && (
                <div data-testid="dpc-vacuum-label" style={readoutStyle}>
                  Using {state.vacuum}
                </div>
              )}
            </>
          )}

          {centerMode === 'manual' && (
            <>
              <button data-testid="dpc-use-crosshair" style={S.primary}
                onClick={useCrosshair}>Use this centre</button>
              {/* The backend confirms the picked position on `dpc_state`.
                  Without this the button looked inert: its `emit_status` goes
                  to the app's status BAR, not to this caret. */}
              {center && (
                <div data-testid="dpc-center-xy" style={readoutStyle}>
                  Centre: ({center[0].toFixed(1)}, {center[1].toFixed(1)}) px
                </div>
              )}
            </>
          )}

          {/* The beam region. Deliberately OUTSIDE the centerMode branches: it
              defines which pixels the centre of mass integrates, which matters
              for every reference mode, not just Manual. */}
          <div style={beamWrap}>
            <Field label={<>Beam region <Info testid="dpc-info-beam" text={INFO.beam} /></>}>
              <div style={{ display: 'flex', gap: 2 }}>
                {BEAM_SHAPES.map(s => (
                  <button key={s.value} data-testid={`dpc-beam-${s.value}`}
                    style={s.value === beamShape ? shapeBtnActive : shapeBtn}
                    onClick={() => setBeam({ beamShape: s.value })}>{s.label}</button>
                ))}
              </div>
            </Field>
            {beamShape !== 'off' && (
              <>
                <Field label="Radius (px)">
                  <NumInput testid="dpc-beam-r" value={round1(beamR)} step="1"
                    width={62} onChange={(n) => setBeam({ beamR: n })} />
                </Field>
                {beamShape === 'ring' && (
                  <Field label="Inner (px)">
                    <NumInput testid="dpc-beam-r-inner" value={round1(beamRInner)}
                      step="1" width={62}
                      onChange={(n) => setBeam({ beamRInner: n })} />
                  </Field>
                )}
                <BeamReadout region={region} shape={beamShape} />
              </>
            )}
          </div>

          <Advanced testid="dpc-advanced">
            <Field label="Beam finder">
              <Select testid="dpc-method" value={method} options={METHOD_OPTIONS}
                onChange={setMethod} />
            </Field>
            <Field label={<>Search box (px) <Info testid="dpc-info-search"
              text={INFO.search} /></>}>
              <NumInput testid="dpc-half-width" value={halfSquareWidth} step="1"
                width={56} onChange={setHalfSquareWidth} />
            </Field>
            <button data-testid="dpc-remeasure" style={ghostStyle}
              onClick={remeasure}>Re-measure</button>
          </Advanced>
        </div>
      )}

      {tab === 'Rotation' && (
        <div style={S.page}>
          <div style={rowStyle}>
            <button data-testid="dpc-solve-rotation" style={S.primary}
              onClick={solveRotation}>Solve rotation</button>
            <Info testid="dpc-info-rotation" text={INFO.rotation} />
          </div>
          <EstimateReadout estimate={estimate} />
          <Field label="Rotation">
            <Slider testid="dpc-rotation" value={rotation} min={0} max={360}
              step={0.5} fmt={(n) => `${n.toFixed(1)}°`}
              onChange={live(setRotation)} />
          </Field>
          <Check testid="dpc-flip" checked={flip} onChange={live(setFlip)}
            label="Flip handedness (swap x/y)" />
          <div style={rowStyle}>
            <Check testid="dpc-reverse" checked={reverse}
              onChange={live(setReverse)} label="Reverse (+180°)" />
            <Info testid="dpc-info-reverse" text={INFO.reverse} />
          </div>
        </div>
      )}

      {tab === 'Field' && (
        <div style={S.page}>
          <Field label="Field">
            <Select testid="dpc-mode" value={mode}
              options={[{ value: 'magnetic', label: 'Magnetic' },
                        { value: 'electric', label: 'Electric' }]}
              onChange={live(setMode)} />
          </Field>
          {mode === 'electric' && (
            <>
              <Field label="Thickness (nm)">
                <NumInput testid="dpc-thickness" value={thicknessNm} step="1"
                  width={62} onChange={live(setThicknessNm)} />
              </Field>
              <Field label="Beam energy (kV)">
                <NumInput testid="dpc-beam-energy" value={beamEnergyKv} step="1"
                  width={62} onChange={live(setBeamEnergyKv)} />
              </Field>
            </>
          )}
          <Field label={<>mrad / px <Info testid="dpc-info-scale"
            text={INFO.scale} /></>}>
            <NumInput testid="dpc-mrad-per-px" value={mradPerPx} step="0.0001"
              width={74} onChange={live(setMradPerPx)} />
          </Field>
          <ResultReadout result={result} />
        </div>
      )}

      {tab === 'Map' && (
        <div style={S.page}>
          <Field label="Show">
            <Select testid="dpc-view" value={view} options={VIEW_OPTIONS}
              onChange={changeView} />
          </Field>
          <Field label={<>Colour limit <Info testid="dpc-info-map"
            text={INFO.map} /></>}>
            <Slider testid="dpc-autolim" value={autolimSigma} min={0.5} max={10}
              step={0.5} fmt={(n) => `${n}σ`} onChange={live(setAutolimSigma)} />
          </Field>
          <ResultReadout result={result} />
          <CommitButton wizardKey="dpc" windowId={windowId}
            sendAction={sendAction} label="Commit to New Tree" />
        </div>
      )}
    </WizardShell>
  )
}

// ── readouts ─────────────────────────────────────────────────────────────────

/** How much descan there is, and therefore whether this step is needed at all. */
function CenteringReadout({ centering }: { centering: Centering | null }) {
  if (!centering) {
    return <div data-testid="dpc-centering" style={S.hint}>measuring the beam…</div>
  }
  const { offset, ramp, worst, centered } = centering
  return (
    <div data-testid="dpc-centering" data-centered={String(centered)}
      data-worst={worst.toFixed(3)}
      style={{ ...readoutStyle, color: centered ? '#a6e3a1' : '#f9e2af' }}>
      {centered ? '✓ already centered' : '△ descan present'}
      <span style={S.hint}>
        {' '}offset ({offset[0].toFixed(1)}, {offset[1].toFixed(1)}) px · ramp{' '}
        ({ramp[0].toFixed(1)}, {ramp[1].toFixed(1)}) px
      </span>
    </div>
  )
}

/**
 * The explanations, behind ⓘ.
 *
 * These used to sit inline under each control, and three sentences of prose per
 * setting crowded out the controls in a caret this narrow. The knowledge is
 * still worth having — most of it is the difference between a right DPC map and
 * a plausible wrong one — so it moved behind a disclosure rather than being
 * deleted.
 */
const INFO = {
  beam: 'Restricts the centre of mass to this shape, and its centre is what '
    + 'Manual centering uses. A circle keeps diffracted discs from dragging the '
    + 'centroid; a ring works when the direct beam is saturated or blocked, by '
    + 'using its edge instead. Drag it on the pattern.',
  search: "pyxem's centred square, kept for the finders a region does not apply "
    + 'to. 0 searches the whole frame. A beam region supersedes it.',
  rotation: "The detector's x/y is rotated relative to the scan's, so every "
    + 'direction on the map depends on getting this right. Solve fits it from '
    + 'the data: an electric field must be curl-free, a magnetic deflection '
    + 'divergence-free, so the correct angle is the one that drives the wrong '
    + 'component to zero. The number beside it is how far that residual fell — '
    + 'near 1× means the data did not single out an angle, so judge by eye.',
  reverse: 'The fit cannot tell 0° from 180°: both give the same residual. That '
    + 'is physics, not a bug. Use this when you know which way the field should '
    + 'point — which side of a junction is positive, say.',
  scale: 'Detector calibration. Read from the dataset when it carries one; 0 '
    + 'leaves the map in raw pixels rather than mrad or MV/cm.',
  map: 'The colour wheel on the map is its legend: find a colour on the wheel '
    + 'and it points the way the field does on screen. Scalar views hide it.',
}

/**
 * An ⓘ that opens its text as a popover.
 *
 * Not a hover tooltip: the text is a paragraph, and a paragraph that vanishes
 * when the pointer moves cannot be read. Click to open, click (or Escape) to
 * close.
 *
 * Absolutely positioned rather than expanded in place, for two reasons. It
 * costs zero layout — the point of moving this text out of the caret was the
 * real estate, and an inline block just moves the controls down instead. And
 * these sit inside `Field` labels, which are `white-space: nowrap` so a control
 * label never wraps mid-word; inline text inherited that and ran off the edge
 * of the caret.
 */
function Info({ text, testid }: { text: string; testid: string }) {
  const [open, setOpen] = React.useState(false)
  React.useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])
  return (
    <span style={{ position: 'relative', display: 'inline-flex' }}>
      <button data-testid={testid} aria-expanded={open} title="More information"
        style={infoBtn} onClick={() => setOpen(v => !v)}>ⓘ</button>
      {open && (
        <div data-testid={`${testid}-text`} style={infoText}
          onClick={() => setOpen(false)}>{text}</div>
      )}
    </span>
  )
}

/**
 * Is the region actually on the beam?
 *
 * Shows its intensity density relative to the frame average, NOT the fraction
 * of intensity it captures — that fraction depends on how much other intensity
 * the pattern carries, so no single threshold means anything across datasets.
 * Density is scale-free: many times 1 on the direct beam, below 1 off it.
 *
 * This is the live signal while dragging; the decisive check happens after the
 * next measure, when the backend compares the beam it FOUND against the region
 * that found it and warns in the status line if they disagree. That matters
 * because a region sitting on empty detector still returns a plausible-looking
 * centroid.
 */
function BeamReadout({ region, shape }: { region: Region | null; shape: BeamShape }) {
  const b = region?.brightness
  if (b == null || !Number.isFinite(b)) {
    return (
      <div data-testid="dpc-beam-readout" style={S.hint}>
        Drag the {shape === 'ring' ? 'ring' : 'circle'} onto the direct beam.
      </div>
    )
  }
  const good = b >= 2
  return (
    <div data-testid="dpc-beam-readout" data-brightness={b.toFixed(2)}
      style={{ ...readoutStyle, color: good ? '#a6e3a1' : '#f9e2af' }}>
      {b.toFixed(1)}× frame average
      <span style={S.hint}>
        {good ? ' · on the beam' : ' · looks off the beam — drag it over'}
      </span>
    </div>
  )
}

function round1(n: number): number {
  return Math.round(n * 10) / 10
}

/**
 * The fit's own verdict on itself.
 *
 * `improvement` is how far the wrong-symmetry residual fell — curl for an
 * electric field, divergence for a magnetic one. A large number means the data
 * really did single out an angle; near 1 means it did not, and the slider is
 * the honest tool. Showing it beats reporting an angle with false confidence.
 */
function EstimateReadout({ estimate }: { estimate: Estimate | null }) {
  if (!estimate) {
    return <div data-testid="dpc-estimate" style={S.hint}>not solved yet</div>
  }
  const target = estimate.mode === 'electric' ? 'curl' : 'divergence'
  const good = estimate.improvement >= 2
  return (
    <div data-testid="dpc-estimate" data-angle={estimate.angle.toFixed(2)}
      data-flip={String(estimate.flip)}
      data-improvement={estimate.improvement.toFixed(2)}
      style={{ ...readoutStyle, color: good ? '#a6e3a1' : '#f9e2af' }}>
      {estimate.angle.toFixed(1)}°{estimate.flip ? ' · flipped' : ''}
      <span style={S.hint}>
        {' '}· {target} down {estimate.improvement.toFixed(1)}×
        {good ? '' : ' (weak — check by eye)'}
      </span>
    </div>
  )
}

function ResultReadout({ result }: { result: Result | null }) {
  if (!result) return null
  return (
    <div data-testid="dpc-result" data-units={result.units}
      data-calibrated={String(result.calibrated)} style={readoutStyle}>
      max {fmt(result.max)} {result.units} · mean {fmt(result.mean)}
      <span style={S.hint}>
        {' '}· div {fmt(result.divergence)} · curl {fmt(result.curl)}
      </span>
    </div>
  )
}

function fmt(n: number): string {
  if (!Number.isFinite(n)) return '—'
  const a = Math.abs(n)
  if (a !== 0 && (a < 0.01 || a >= 1e4)) return n.toExponential(1)
  return n.toFixed(a < 1 ? 3 : 2)
}

/** `dpc.corner_slices` sizing, mirrored so the hint matches what is drawn —
 *  including the `MIN_CORNER_PX` floor (a 1x1 box cannot determine a plane). */
const MIN_CORNER_PX = 2
function cornerBoxSize(navShape: number[], fraction: number): string {
  const [ny, nx] = navShape
  const hy = Math.max(MIN_CORNER_PX, Math.min(ny, Math.round(ny * fraction)))
  const hx = Math.max(MIN_CORNER_PX, Math.min(nx, Math.round(nx * fraction)))
  return `${hx}×${hy} px`
}

/** Collapsed disclosure — the same shape DriftWizard uses (plan §0.9a). */
function Advanced({ testid, children }: { testid: string; children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false)
  return (
    <div style={advancedWrap}>
      <button data-testid={`${testid}-toggle`} style={discloseStyle}
        onClick={() => setOpen(v => !v)}>{open ? '▾' : '▸'} Advanced</button>
      {open && <div data-testid={testid} style={S.page}>{children}</div>}
    </div>
  )
}

const readoutStyle: React.CSSProperties = {
  fontSize: 10, fontVariantNumeric: 'tabular-nums',
}
const rowStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap',
}
const infoBtn: React.CSSProperties = {
  background: 'none', border: 'none', color: '#6c7086', cursor: 'pointer',
  fontSize: 11, padding: 0, lineHeight: 1, flex: '0 0 auto',
}
const infoText: React.CSSProperties = {
  position: 'absolute', top: 'calc(100% + 4px)', left: -8,
  width: 216, zIndex: 20,
  fontSize: 10, color: '#cdd6f4', background: '#1e1e2e',
  border: '1px solid #45475a', borderRadius: 5, padding: '6px 8px',
  lineHeight: 1.45, boxShadow: '0 8px 20px rgba(0,0,0,0.55)',
  // `S.lbl` is nowrap so control labels never break mid-word; this is a
  // paragraph and must opt back out of that.
  whiteSpace: 'normal', cursor: 'pointer',
}
const beamWrap: React.CSSProperties = {
  borderTop: '1px solid #313244', paddingTop: 6,
  display: 'flex', flexDirection: 'column', gap: 6,
}
const shapeBtn: React.CSSProperties = {
  background: '#11111b', color: '#a6adc8', border: '1px solid #313244',
  borderRadius: 4, padding: '3px 9px', fontSize: 11, cursor: 'pointer',
}
const shapeBtnActive: React.CSSProperties = {
  ...shapeBtn, background: '#313244', color: '#cdd6f4', fontWeight: 600,
}
const ghostStyle: React.CSSProperties = {
  background: '#313244', color: '#cdd6f4', border: '1px solid #45475a',
  borderRadius: 5, padding: '5px 9px', fontSize: 11, cursor: 'pointer',
  alignSelf: 'flex-start',
}
const advancedWrap: React.CSSProperties = {
  borderTop: '1px solid #313244', paddingTop: 4,
  display: 'flex', flexDirection: 'column', gap: 6,
}
const discloseStyle: React.CSSProperties = {
  background: 'none', border: 'none', color: '#a6adc8', cursor: 'pointer',
  fontSize: 10, padding: 0, textAlign: 'left', alignSelf: 'flex-start',
}
