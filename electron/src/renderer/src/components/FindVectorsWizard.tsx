/**
 * FindVectorsWizard.tsx — the Find-Diffraction-Vectors caret (Qt parity).
 *
 * Opening the caret starts a LIVE preview of the found peaks on the diffraction
 * pattern (`fv_open`); the Method dropdown + parameter sliders re-run it
 * (debounced `fv_tune`) and moving the navigator previews other patterns.
 * "Compute" (`fv_run`) runs the full-dataset batch → a new vectors window;
 * closing the caret tears the preview down (`fv_close`).
 *
 * Compute is held until the backend's automatic estimates for this open have
 * all arrived (`fv_estimates_done`), so the batch runs with the parameters the
 * caret shows rather than whichever ones had landed by the click. A value the
 * user set by hand is never replaced by a late estimate.
 *
 * Three detection methods:
 *   • Neural — the SpotUNet disk detector. Deliberately MINIMAL controls
 *     (user decision 2026-07-16): Spot size (auto-seeded; drives the model's
 *     canonical rescale + NMS + marker radius) and Threshold (model confidence,
 *     ~0.3). Nav blur is never applied for neural; the high-pass (bg σ) is
 *     auto-calibrated invisibly (`fv_calibration`). Model dropdown from the
 *     registry (`fv_models`); its last row checks Hugging Face for new ones.
 *     Every disk's centre is refined on the raw frame; the "Friedel partner"
 *     checkbox adds the step that also reads each disk's Friedel mirror
 *     (spyde.models.centre_refine). The default method.
 *   • NXCORR — window-normalised cross-correlation against a flat disk
 *     (Disk Radius slider; threshold is a [-1,1] correlation score).
 *   • DoG — Difference-of-Gaussians band-pass, best for small (2-3 px) spots
 *     and beam-stopped patterns (σ₁/σ₂ sliders; threshold is a band-pass SNR).
 *
 * Teach (neural only): double-clicks on the pattern are handled by the backend
 * (spyde/actions/find_vectors_adapt.py) — a circle → "not a disk", empty
 * pattern → "a disk is here", a mark → removed. It refits a copy of the model
 * about a second after the last mark and reports `fv_adapt_state`; the caret
 * adopts the taught model as its Model so the preview and Compute both use it.
 * A fit is an unsaved draft until it is named (Save beside Adapt / Revert asks
 * for a name); named models are offered on every dataset, can be renamed or
 * deleted from the dropdown's rows, and can be adapted again.
 */
import React from 'react'
import { WizardShell, Slider, Select, Check, Toggle, Disclosure, S } from './WizardShell'
import { useWizardLifecycle, useDebouncedAction, useWizardEvent } from './wizardHooks'
import { ModelMenu, ModelInfo } from './ModelMenu'

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

type Method = 'neural' | 'nxcorr' | 'dog'
const METHODS: readonly { value: Method; label: string }[] = [
  { value: 'neural', label: 'Neural network' },
  { value: 'nxcorr', label: 'NXCORR (disk)' },
  { value: 'dog', label: 'DoG (small spots)' },
]
// Sensible threshold default per method (neural confidence vs NXCORR score vs DoG SNR).
const THR_DEFAULT: Record<Method, number> = { neural: 0.3, nxcorr: 0.5, dog: 10 }

// Per-window tuned state kept OUTSIDE the component (like the OM/VOM wizards)
// so the caret closing on Compute doesn't lose the tuning — reopening restores
// the last-used parameters.
interface FvSaved {
  method: Method; modelId: string; sigma: number; radius: number
  sigma1: number; sigma2: number; bgSigma: number
  threshold: number; minDist: number; subpixel: boolean; beamstop: boolean
  beamstopDilate: number; showTransform: boolean; persistence: boolean
  friedelPartner: boolean; centerBeam: boolean; extraOpen: boolean
}
const _fvStore = new Map<number, FvSaved>()

// Neural local-norm high-pass scale (px). 12 matches the backend default; the
// backend's one-shot auto-calibration (`fv_calibration`) overwrites it unless
// the user already moved the slider.
const BG_SIGMA_DEFAULT = 12

export function FindVectorsWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const saved = _fvStore.get(windowId)
  const [method, setMethod] = React.useState<Method>(saved?.method ?? 'neural')
  const [modelId, setModelId] = React.useState(saved?.modelId ?? '')
  const [models, setModels] = React.useState<readonly ModelInfo[]>([])
  const [sigma, setSigma] = React.useState(saved?.sigma ?? 0)
  const [radius, setRadius] = React.useState(saved?.radius ?? 5)
  const [sigma1, setSigma1] = React.useState(saved?.sigma1 ?? 0.8)
  const [sigma2, setSigma2] = React.useState(saved?.sigma2 ?? 2.0)
  const [bgSigma, setBgSigma] = React.useState(saved?.bgSigma ?? BG_SIGMA_DEFAULT)
  const [threshold, setThreshold] = React.useState(saved?.threshold ?? 0.3)
  const [minDist, setMinDist] = React.useState(saved?.minDist ?? 5)
  const [subpixel, setSubpixel] = React.useState(saved?.subpixel ?? true)
  const [beamstop, setBeamstop] = React.useState(saved?.beamstop ?? false)
  const [beamstopDilate, setBeamstopDilate] = React.useState(saved?.beamstopDilate ?? 5)
  const [showTransform, setShowTransform] = React.useState(saved?.showTransform ?? false)
  const [persistence, setPersistence] = React.useState(saved?.persistence ?? false)
  const [friedelPartner, setFriedelPartner] = React.useState(saved?.friedelPartner ?? true)
  const [centerBeam, setCenterBeam] = React.useState(saved?.centerBeam ?? true)
  const [extraOpen, setExtraOpen] = React.useState(saved?.extraOpen ?? false)
  const [status, setStatus] = React.useState('Tune the parameters — peaks preview under the crosshair.')
  const [estimating, setEstimating] = React.useState(true)
  // Controls the user has moved by hand since this caret opened.
  const handEdited = React.useRef(new Set<'radius' | 'minDist' | 'threshold'>())
  const [teach, setTeach] = React.useState<TeachState>(TEACH_EMPTY)
  // The draft the Save name box was opened for. Keyed by id rather than a flag,
  // so the box survives the brief moments a state update leaves no draft in
  // use, yet never reopens over a different, later draft.
  const [namingModelId, setNamingModelId] = React.useState<string | null>(null)

  React.useEffect(() => {
    _fvStore.set(windowId, {
      method, modelId, sigma, radius, sigma1, sigma2, bgSigma, threshold, minDist, subpixel,
      beamstop, beamstopDilate, showTransform, persistence, friedelPartner, centerBeam, extraOpen,
    })
  }, [windowId, method, modelId, sigma, radius, sigma1, sigma2, bgSigma, threshold, minDist,
      subpixel, beamstop, beamstopDilate, showTransform, persistence, friedelPartner, centerBeam,
      extraOpen])

  // Live refs so the debounced tune always sends the latest of EVERY control.
  const vals = React.useRef({ method, modelId, sigma, radius, sigma1, sigma2, bgSigma, threshold, minDist, subpixel, beamstop, beamstopDilate, showTransform, persistence, friedelPartner, centerBeam })
  vals.current = { method, modelId, sigma, radius, sigma1, sigma2, bgSigma, threshold, minDist, subpixel, beamstop, beamstopDilate, showTransform, persistence, friedelPartner, centerBeam }
  const params = () => {
    const v = vals.current
    const neural = v.method === 'neural'
    return {
      method: v.method, model_id: v.modelId,
      // Neural never applies nav blur (the backend forces it too); the Spot
      // size slider is the single scale knob — it drives the model's rescale
      // (spot_radius) and the NMS min-distance (~radius/2).
      sigma: neural ? 0 : v.sigma, kernel_radius: v.radius,
      dog_sigma1: v.sigma1, dog_sigma2: v.sigma2,
      bg_sigma: v.bgSigma,
      spot_radius: neural ? v.radius : 0,
      threshold: v.threshold,
      min_distance: neural ? Math.max(1, Math.round(v.radius / 2)) : v.minDist,
      subpixel: v.subpixel, beamstop_auto: v.beamstop,
      beamstop_dilate: v.beamstopDilate,
      show_transform: v.showTransform,
      persistence: v.persistence,
      refine_centres: neural,
      friedel_partner: neural && v.friedelPartner,
      center_beam: v.centerBeam,
    }
  }

  // Start the live preview when the caret opens; tear it down when it closes
  // (StrictMode-safe: exactly one fv_open reaches the backend).
  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'fv_open', openPayload: params, closeAction: 'fv_close',
  })

  // Debounced live tune — dispatch on control settle so matches don't flood;
  // a pending tune is cancelled on unmount so it can't hit a torn-down preview.
  const sendTune = useDebouncedAction(sendAction, 'fv_tune', windowId)
  const tune = () => sendTune(params)
  const live = <T,>(set: (v: T) => void) => (v: T) => { set(v); tune() }
  const byHand = <T,>(key: 'radius' | 'minDist' | 'threshold', set: (v: T) => void) =>
    live((v: T) => { handEdited.current.add(key); set(v) })

  // Populate the Model dropdown from the backend registry. Requested ONCE on
  // mount via a ref (sendAction must never be an effect dep — it's recreated
  // every render and would re-fire the request in a loop).
  const sendRef = React.useRef(sendAction)
  sendRef.current = sendAction
  React.useEffect(() => {
    sendRef.current('fv_models', {}, windowId)
  }, [windowId])
  useWizardEvent('spyde:fv_models', windowId, (d) => {
    const list = Array.isArray(d.models) ? (d.models as ModelInfo[]) : []
    setModels(list.map((m) => ({ ...m, label: m.label ?? m.id })))
    // '' = registry default; surface it as the concrete id once known. A model
    // that is no longer listed (deleted, or an unsaved draft that was discarded)
    // falls back to the default too.
    const current = vals.current.modelId
    if (typeof d.default === 'string' && (!current || !list.some((m) => m.id === current))) {
      setModelId(d.default)
      vals.current = { ...vals.current, modelId: d.default }
      if (current) tune()
    }
    if (d.refreshed) setStatus(`Model list refreshed — ${list.length} available.`)
  })

  // "Check for new models": pull the latest registry from Hugging Face; the
  // backend re-emits fv_models (refreshed: true) with the merged list.
  const refreshModels = () => {
    setStatus('Checking for new models…')
    sendRef.current('fv_refresh_models', {}, windowId)
  }

  // Backend one-shot auto-calibration (neural). The high-pass (bg σ) has NO
  // visible control — it is always adopted (fully automatic); the threshold is
  // adopted only if the user hasn't moved it off the neural default.
  useWizardEvent('spyde:fv_calibration', windowId, (d) => {
    if (vals.current.method !== 'neural') return
    let adopted = false
    if (typeof d.bg_sigma === 'number' && d.bg_sigma !== vals.current.bgSigma) {
      setBgSigma(d.bg_sigma)
      vals.current = { ...vals.current, bgSigma: d.bg_sigma }
      adopted = true
    }
    if (typeof d.thresh === 'number' && !handEdited.current.has('threshold')
        && vals.current.threshold === THR_DEFAULT.neural && d.thresh !== THR_DEFAULT.neural) {
      setThreshold(d.thresh)
      vals.current = { ...vals.current, threshold: d.thresh }
      adopted = true
    }
    if (adopted) {
      const conf = typeof d.confidence === 'number' ? ` (conf ${d.confidence.toFixed(2)})` : ''
      setStatus(`Auto-calibrated background removal${conf}.`)
      tune()
    }
  })

  // Teach: the backend owns the marks and the taught model; the caret follows
  // its Model choice so the preview and Compute both use the taught model, and
  // goes back to the original one on Revert.
  const teachRef = React.useRef<string | null>(null)
  useWizardEvent('spyde:fv_adapt_state', windowId, (d) => {
    const taught = typeof d.model_id === 'string' ? d.model_id : null
    const base = typeof d.base_model_id === 'string' ? d.base_model_id : ''
    setTeach({
      marks: Number(d.marks ?? 0), disk: Number(d.disk ?? 0), notDisk: Number(d.not_disk ?? 0),
      busy: Boolean(d.busy), taught, unsaved: Boolean(d.unsaved),
      learned: typeof d.marks_learned === 'number' ? d.marks_learned : null,
      learnable: typeof d.marks_total === 'number' ? d.marks_total : null,
      defaultName: typeof d.default_name === 'string' ? d.default_name : '',
      originalF1: typeof d.original_f1 === 'number' ? d.original_f1 : null,
      originalF1Base: typeof d.original_f1_base === 'number' ? d.original_f1_base : null,
    })
    if (typeof d.status === 'string') setStatus(d.status)
    const want = taught ?? (teachRef.current ? base : vals.current.modelId)
    teachRef.current = taught
    if (want !== vals.current.modelId) {
      setModelId(want)
      vals.current = { ...vals.current, modelId: want }
      tune()
    }
  })

  // Adopt the backend's auto-estimated disk radius (Qt parity) the first time it
  // arrives, then re-run the preview with it. The user can still override.
  const autoApplied = React.useRef(false)
  useWizardEvent('spyde:fv_auto_params', windowId, (d) => {
    if (autoApplied.current) return
    autoApplied.current = true
    const radiusFree = !handEdited.current.has('radius')
    if (radiusFree && typeof d.kernel_radius === 'number') setRadius(d.kernel_radius)
    if (!handEdited.current.has('minDist') && typeof d.min_distance === 'number') {
      setMinDist(d.min_distance)
    }
    if (radiusFree) setStatus(`Spot size auto-set to ${d.kernel_radius} px from the pattern.`)
    tune()
  })

  useWizardEvent('spyde:fv_estimates_done', windowId, () => setEstimating(false))

  // Switching method resets the threshold to that method's natural scale and
  // re-runs the preview (NXCORR score and DoG SNR are not comparable numbers).
  const onMethod = (m: Method) => {
    setMethod(m)
    setThreshold(THR_DEFAULT[m])
    vals.current = { ...vals.current, method: m, threshold: THR_DEFAULT[m] }
    tune()
  }

  const compute = () => {
    sendAction('fv_run', params(), windowId)
    // Compute collapses the caret back into the toolbar button (its tuned
    // state is kept in _fvStore). The unmount fires fv_close, which only
    // drops the live preview — the batch keeps running.
    onClose()
  }

  const isDog = method === 'dog'
  const isNeural = method === 'neural'
  // Save names the fit just made: the model in use is the taught one, still a draft.
  const currentModel = models.find((m) => m.id === modelId)
  const unsavedModel = currentModel?.unsaved && currentModel.id === teach.taught ? currentModel : undefined

  return (
    <WizardShell testid="find-vectors-wizard" title="Find Diffraction Vectors" posStyle={caretPos}
      onClose={onClose} closeTestid="fv-close" status={status} statusTestid="fv-status"
      width={320}>
      <div style={S.hint}>Move the crosshair on the navigator to preview the found peaks.</div>
      <div style={rowStyle}>
        <label style={rowLabelStyle}>Method</label>
        <div style={growStyle}>
          <Select testid="fv-method" value={method} options={METHODS} onChange={onMethod} />
        </div>
        <Toggle testid="fv-show-transform" checked={showTransform} onChange={live(setShowTransform)}
          label={isDog ? 'DoG' : isNeural ? 'Heatmap' : 'Corr.'}
          title={isDog ? 'Show the band-passed image instead of the pattern'
            : isNeural ? "Show the network's disk heatmap instead of the pattern"
            : 'Show the correlation map instead of the pattern'} />
      </div>
      {isNeural && models.length > 0 && (
        <div style={rowStyle}>
          <label style={rowLabelStyle}>Model</label>
          <div style={growStyle}>
            <ModelMenu testid="fv-model" models={models} value={modelId} onChange={live(setModelId)}
              onRename={(id, name) => sendRef.current('fv_model_name', { model_id: id, name }, windowId)}
              onDelete={(id) => sendRef.current('fv_model_delete', { model_id: id }, windowId)}
              onRefresh={refreshModels} />
          </div>
        </div>
      )}
      <div style={gridStyle}>
        {!isNeural && (
          // Nav blur is a NXCORR/DoG option only (defaults off) — the neural
          // net is trained on single frames and never gets a blurred input.
          <Cell label="Nav Blur σ">
            <Slider testid="fv-sigma" value={sigma} min={0} max={5} step={0.1}
              onChange={live(setSigma)} fmt={(n) => n.toFixed(1)} />
          </Cell>
        )}
        {isNeural && (
          // THE neural scale knob: auto-seeded from the pattern; drives the
          // model's canonical rescale, the NMS spacing and the marker radius.
          <Cell label="Spot size (px)">
            <Slider testid="fv-spot-size" value={radius} min={2} max={30} step={1}
              onChange={byHand('radius', setRadius)} />
          </Cell>
        )}
        {!isDog && !isNeural && (
          <Cell label="Disk Radius">
            <Slider testid="fv-radius" value={radius} min={1} max={30} step={1}
              onChange={byHand('radius', setRadius)} />
          </Cell>
        )}
        {isDog && (
          <>
            <Cell label="DoG σ₁ (spot)">
              <Slider testid="fv-dog-sigma1" value={sigma1} min={0.4} max={3} step={0.1}
                onChange={live(setSigma1)} fmt={(n) => n.toFixed(1)} />
            </Cell>
            <Cell label="DoG σ₂ (bg)">
              <Slider testid="fv-dog-sigma2" value={sigma2} min={1} max={6} step={0.1}
                onChange={live(setSigma2)} fmt={(n) => n.toFixed(1)} />
            </Cell>
          </>
        )}
        <Cell label={isDog ? 'Threshold (SNR)' : isNeural ? 'Threshold (conf.)' : 'Threshold'}>
          <Slider testid="fv-threshold" value={threshold}
            min={0} max={isDog ? 30 : 1} step={isDog ? 0.5 : 0.05}
            onChange={byHand('threshold', setThreshold)} fmt={(n) => (isDog ? n.toFixed(1) : n.toFixed(2))} />
        </Cell>
        {!isNeural && (
          <Cell label="Min Distance">
            <Slider testid="fv-mindist" value={minDist} min={1} max={30} step={1}
              onChange={byHand('minDist', setMinDist)} />
          </Cell>
        )}
      </div>
      <div style={gridStyle}>
        {!isNeural && (
          // The net always emits subpixel offsets — the toggle is a no-op there.
          <Check testid="fv-subpixel" checked={subpixel} onChange={live(setSubpixel)}
            label="Subpixel" />
        )}
        {isNeural && (
          <Check testid="fv-friedel" checked={friedelPartner} onChange={live(setFriedelPartner)}
            label="Friedel partner" />
        )}
        {isNeural && (
          // Batch-only stage-2 refine (scan-neighbour persistence + Friedel);
          // the live preview has no neighbours so it only affects Compute.
          <Check testid="fv-persistence" checked={persistence} onChange={live(setPersistence)}
            label="Neighbor refine" />
        )}
        {/* Compute only: the result is moved so the zero beam is the origin,
            unless a beam stop hides it. The preview is not moved. */}
        <Check testid="fv-center-beam" checked={centerBeam} onChange={setCenterBeam}
          label="Center on zero beam" />
      </div>
      <Disclosure label="Extra" testid="fv-extra" open={extraOpen} onToggle={() => setExtraOpen((v) => !v)}>
        <Check testid="fv-beamstop" checked={beamstop} onChange={live(setBeamstop)}
          label="Mask beam stop" />
        {beamstop && (
          <Cell label="Beam-stop dilate">
            <Slider testid="fv-beamstop-dilate" value={beamstopDilate} min={0} max={20} step={1}
              onChange={live(setBeamstopDilate)} />
          </Cell>
        )}
      </Disclosure>
      {isNeural && (
        <div data-testid="fv-teach" style={teachStyle}
          title={'Double-click a circle that is not a disk, or a disk it missed; double-click a mark to remove it. '
            + 'The model is refitted on your marks about a second after the last one.'}>
          <span data-testid="fv-teach-count">
            {teach.marks === 0
              ? 'Teach: double-click a wrong circle or a missed disk'
              : `Teach: ${teach.notDisk} wrong · ${teach.disk} missed${teach.busy ? ' — adapting…' : ''}`}
          </span>
          <div style={teachRowStyle}>
            <SmallButton testid="fv-adapt" disabled={teach.marks === 0 || teach.busy}
              onClick={() => sendRef.current('fv_adapt', {}, windowId)}>Adapt</SmallButton>
            <SmallButton testid="fv-adapt-revert" disabled={teach.marks === 0 && !teach.taught}
              onClick={() => sendRef.current('fv_adapt_revert', {}, windowId)}>Revert</SmallButton>
            <SmallButton testid="fv-model-save" disabled={!unsavedModel}
              title={unsavedModel ? 'Name the adapted model to keep it' : 'Adapt first: there is no unsaved model'}
              onClick={() => setNamingModelId(unsavedModel?.id ?? null)}>Save</SmallButton>
          </div>
          {unsavedModel && namingModelId === unsavedModel.id && (
            <SaveName initial={teach.defaultName || unsavedModel.label}
              onCancel={() => setNamingModelId(null)}
              onSave={(name) => {
                sendRef.current('fv_model_name', { model_id: unsavedModel.id, name }, windowId)
                setNamingModelId(null)
              }} />
          )}
          {teach.taught && teach.originalF1 !== null && (
            <div data-testid="fv-teach-gauge"
              style={{ color: teach.originalF1 < 0.7 ? '#ff9a3c' : '#a6adc8' }}
              title="F1 on the synthetic patterns the model was trained on: how far teaching pulled it toward this dataset">
              {teach.learned !== null && teach.learnable !== null && teach.learned < teach.learnable
                ? `${teach.learned} of ${teach.learnable} marks learned · `
                : ''}
              {teach.originalF1 < 0.7 ? 'Now specific to this dataset' : 'Using the taught model'}
              {` · general F1 ${teach.originalF1.toFixed(2)}`}
              {teach.originalF1Base !== null ? ` (was ${teach.originalF1Base.toFixed(2)})` : ''}
            </div>
          )}
        </div>
      )}
      <button data-testid="fv-compute" disabled={estimating} onClick={compute}
        style={{ ...S.primary, ...(estimating ? S.primaryBusy : null) }}
        title={estimating ? 'Estimating the spot size and calibrating from the pattern' : undefined}>
        {estimating ? 'Calibrating…' : 'Compute'}
      </button>
    </WizardShell>
  )
}

interface TeachState {
  marks: number; disk: number; notDisk: number; busy: boolean; taught: string | null
  unsaved: boolean; defaultName: string; learned: number | null; learnable: number | null
  originalF1: number | null; originalF1Base: number | null
}
const TEACH_EMPTY: TeachState = {
  marks: 0, disk: 0, notDisk: 0, busy: false, taught: null, unsaved: false, defaultName: '',
  learned: null, learnable: null, originalF1: null, originalF1Base: null,
}

/** The name box Save opens under the teach buttons: Enter or ✓ saves, Escape
 * or ✕ puts it away. Prefilled with the backend's suggested name. */
function SaveName({ initial, onSave, onCancel }: {
  initial: string; onSave: (name: string) => void; onCancel: () => void
}) {
  const [text, setText] = React.useState(initial)
  const save = () => {
    const name = text.trim()
    if (name) onSave(name)
  }
  return (
    <div style={saveNameStyle}>
      <input data-testid="fv-save-name-input" style={{ ...S.num, flex: 1, minWidth: 0 }} value={text}
        autoFocus onFocus={(e) => e.target.select()} placeholder="Name this model"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') save()
          // Escape belongs to this box, not to whatever else listens for it.
          if (e.key === 'Escape') { e.stopPropagation(); onCancel() }
        }} />
      <SmallButton testid="fv-save-name-confirm" disabled={!text.trim()} title="Save"
        onClick={save}>✓</SmallButton>
      <SmallButton testid="fv-save-name-cancel" title="Cancel" onClick={onCancel}>✕</SmallButton>
    </div>
  )
}
const saveNameStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 4, padding: 4,
  background: '#181825', border: '1px solid #313244', borderRadius: 5,
}
const teachStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 3, fontSize: 11,
  borderTop: '1px solid #333', paddingTop: 6,
}
const teachRowStyle: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 4 }
const smallButtonStyle: React.CSSProperties = {
  flex: '0 0 auto', padding: '1px 6px', fontSize: 11, background: 'transparent', color: 'inherit',
  border: '1px solid #555', borderRadius: 4, cursor: 'pointer',
}
const smallButtonDisabledStyle: React.CSSProperties = { opacity: 0.4, cursor: 'default' }

/** The teach row's buttons: dimmed when there is nothing for them to do. */
function SmallButton({ testid, disabled = false, title, onClick, children }: {
  testid: string; disabled?: boolean; title?: string; onClick: () => void; children: React.ReactNode
}) {
  return (
    <button data-testid={testid} disabled={disabled} title={title} onClick={onClick}
      style={{ ...smallButtonStyle, ...(disabled ? smallButtonDisabledStyle : null) }}>{children}</button>
  )
}

const gridStyle: React.CSSProperties = {
  display: 'grid', gridTemplateColumns: '1fr 1fr', columnGap: 10, rowGap: 6,
}
const rowStyle: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }
// One width for both row labels, so the Method and Model controls line up.
const rowLabelStyle: React.CSSProperties = { ...S.lbl, flex: '0 0 auto', width: 36 }
const growStyle: React.CSSProperties = { flex: 1, minWidth: 0 }
const cellStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0,
}

// A labelled control as a compact stacked cell (label above control) for the
// two-column grid. MUST be module-scope, NOT defined inside the component —
// a component defined inline is a new type every render, so React remounts the
// sliders on each keystroke and they lose their drag ("sliders don't work").
function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return <div style={cellStyle}><label style={S.lbl}>{label}</label>{children}</div>
}
