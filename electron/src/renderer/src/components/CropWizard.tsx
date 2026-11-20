/**
 * CropWizard.tsx — the Crop panel: a range on every axis of the dataset.
 *
 * The backend (spyde/actions/crop.py) owns the ranges. Opening the panel
 * (`crop_open`) puts a box on every plot that draws an axis — a rectangle on
 * the image / detector and on a scan navigator, a span on a time navigator —
 * and sends `crop_state`, which this panel renders from: one row per axis with
 * a from/to field pair and a two-thumb slider. A drag on any box, a slider or
 * a field (`crop_set_range`) changes the one set of ranges, and the backend
 * re-sends the state with every box moved to match.
 *
 * Indices are shown inclusive (first and last kept); the backend works in
 * `[start, stop)`.
 */
import React from 'react'
import { useWizardEvent, useWizardLifecycle } from './wizardHooks'
import { WizardShell, NumInput, S } from './WizardShell'

interface CropAxis {
  key: string; group: string; label: string; name: string
  size: number; start: number; stop: number
  scale: number; offset: number; units: string
}

interface CropState {
  axes: CropAxis[]; before: string; after: string; is_crop: boolean; boxes: number
}

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

const SLIDER_CSS = `
.crop-range { position: relative; height: 16px; flex: 1; min-width: 60px; }
.crop-range::before { content: ""; position: absolute; left: 0; right: 0; top: 7px;
  height: 2px; background: #313244; border-radius: 1px; }
.crop-range .crop-fill { position: absolute; top: 7px; height: 2px; background: #f9e2af; }
.crop-range input { position: absolute; left: 0; width: 100%; top: 0; height: 16px;
  margin: 0; background: none; pointer-events: none; -webkit-appearance: none; appearance: none; }
.crop-range input::-webkit-slider-thumb { -webkit-appearance: none; pointer-events: auto;
  width: 10px; height: 10px; border-radius: 50%; background: #f9e2af; border: none; cursor: pointer; }
.crop-range input::-webkit-slider-runnable-track { background: none; }
`

/** "0.05 – 0.15 s": the kept range in the axis's own units. */
function calibrated(axis: CropAxis): string {
  if (!axis.units && axis.scale === 1 && axis.offset === 0) return ''
  const value = (index: number) => axis.offset + index * axis.scale
  const digits = Math.max(0, Math.min(4, -Math.floor(Math.log10(Math.abs(axis.scale) || 1)) + 1))
  const text = (n: number) => n.toFixed(digits)
  return `${text(value(axis.start))} – ${text(value(axis.stop - 1))}${axis.units ? ` ${axis.units}` : ''}`
}

function RangeSlider({ axis, onChange }: {
  axis: CropAxis; onChange: (start: number, stop: number) => void
}) {
  const last = Math.max(axis.size - 1, 1)
  const first = axis.start, end = axis.stop - 1
  return (
    <div className="crop-range">
      <div className="crop-fill"
        style={{ left: `${(first / last) * 100}%`, width: `${((end - first) / last) * 100}%` }} />
      <input data-testid={`crop-${axis.key}-slider-from`} type="range"
        min={0} max={axis.size - 1} step={1} value={first}
        onChange={(e) => onChange(Math.min(Number(e.target.value), end), axis.stop)} />
      <input data-testid={`crop-${axis.key}-slider-to`} type="range"
        min={0} max={axis.size - 1} step={1} value={end}
        onChange={(e) => onChange(axis.start, Math.max(Number(e.target.value), first) + 1)} />
    </div>
  )
}

export function CropWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const [state, setState] = React.useState<CropState | null>(null)

  useWizardLifecycle({
    windowId, sendAction,
    openAction: 'crop_open',
    closeAction: 'crop_close',
    deps: [windowId],
  })

  useWizardEvent('spyde:crop_state', windowId, (detail) => {
    setState(detail as unknown as CropState)
  })

  // Applied here at once so a slider tracks the pointer; the backend's echo
  // carries the same numbers.
  const setRange = (key: string, start: number, stop: number) => {
    setState((current) => current && {
      ...current,
      axes: current.axes.map((axis) => axis.key === key ? { ...axis, start, stop } : axis),
    })
    sendAction('crop_set_range', { key, start, stop }, windowId)
  }

  const groups: [string, CropAxis[]][] = []
  for (const axis of state?.axes ?? []) {
    const group = groups.find(([name]) => name === axis.group)
    if (group) group[1].push(axis)
    else groups.push([axis.group, [axis]])
  }

  return (
    <WizardShell testid="crop-wizard" title="Crop" posStyle={caretPos} width={300}
      onClose={onClose} closeTestid="crop-close" statusTestid="crop-status"
      status={state ? 'Drag the boxes on the plots, the sliders, or type a range.'
                     : 'Reading the dataset…'}>
      <style>{SLIDER_CSS}</style>
      <div style={S.page}>
        {groups.map(([group, axes]) => (
          <div key={group} data-testid={`crop-group-${group.toLowerCase()}`}
            style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ ...S.groupLabel, marginTop: 0 }}>{group}</div>
            {axes.map((axis) => (
              <div key={axis.key} data-testid={`crop-axis-${axis.key}`}
                style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                  <span style={S.lbl}>{axis.label}
                    <span style={{ color: '#6c7086' }}> · {axis.stop - axis.start} of {axis.size}</span>
                  </span>
                  <span data-testid={`crop-${axis.key}-value`} style={S.numLabel}>{calibrated(axis)}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <NumInput testid={`crop-${axis.key}-from`} value={axis.start} step="1" width={52}
                    min={0} max={axis.size - 1} accept={Number.isInteger}
                    onChange={(n) => setRange(axis.key, n, Math.max(axis.stop, n + 1))} />
                  <RangeSlider axis={axis} onChange={(start, stop) => setRange(axis.key, start, stop)} />
                  <NumInput testid={`crop-${axis.key}-to`} value={axis.stop - 1} step="1" width={52}
                    min={0} max={axis.size - 1} accept={Number.isInteger}
                    onChange={(n) => setRange(axis.key, Math.min(axis.start, n), n + 1)} />
                </div>
              </div>
            ))}
          </div>
        ))}
        {state && (
          <div data-testid="crop-summary" style={{ ...S.hint, fontStyle: 'normal', borderTop: '1px solid #313244', paddingTop: 5 }}>
            {state.before}<br />→ {state.after}
          </div>
        )}
        <div style={{ display: 'flex', gap: 6 }}>
          <button data-testid="crop-run" style={{ ...S.primary, ...(state?.is_crop ? {} : S.primaryBusy) }}
            disabled={!state?.is_crop}
            onClick={() => { sendAction('crop_run', {}, windowId); onClose() }}>
            Crop to new dataset
          </button>
          <button data-testid="crop-reset" style={S.fileBtn}
            onClick={() => sendAction('crop_reset', {}, windowId)}>
            Reset
          </button>
        </div>
      </div>
    </WizardShell>
  )
}
