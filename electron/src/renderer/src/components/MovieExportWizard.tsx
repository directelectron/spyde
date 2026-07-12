/**
 * MovieExportWizard.tsx — the "Export Movie" caret (insitu movies only).
 *
 * Staged action key `mvx` (backend: spyde/actions/movie_export/). Opening the
 * caret sends `mvx_open {window_id}` (via useWizardLifecycle) which makes the
 * backend probe ffmpeg, read the movie's time axis, seed cmap/levels from the
 * plot, and re-broadcast an authoritative `mvx_state`. The wizard mirrors that
 * state and drives it back with debounced `mvx_tune`:
 *
 *   1 Format / quality — fps, spatial downsample, temporal stride, + a computed
 *                        output-frames / duration readout.
 *   2 Time range       — start/end shown in the movie's time units (seconds-
 *                        scaled via time.scale_s), defaulting to the full range.
 *   3 Overlays         — timestamp + scale bar checkboxes; a minimal annotation
 *                        list (Text/Rect, each with a [t0,t1] time-range window).
 *   4 Traces           — a dashed drop slot accepting a dragged 1-D plot window
 *                        (WINDOW_DRAG_MIME → mvx_add_trace); trace chips below.
 *   5 Export           — save dialog (mp4/gif) → mvx_run {path}; progress via the
 *                        app's StatusBar; on `mvx_done` a success note. Cancel
 *                        while running → mvx_cancel. Disabled when ffmpeg absent.
 *
 * Live tune is DEBOUNCED (useDebouncedAction) and the whole annotation list is
 * re-sent on any edit — the backend replaces its list wholesale from mvx_tune
 * {annotations}. The caret closes on unmount via `mvx_close`.
 */
import React from 'react'
import { WizardShell, TabRow, Field, NumInput, Select, Check, S } from './WizardShell'
import { useWizardLifecycle, useDebouncedAction, useWizardEvent } from './wizardHooks'
import { WINDOW_DRAG_MIME } from '../kernel/dnd'
import type { MvxStateMessage, MvxParams, MvxTrace, MvxAnnotation } from '../kernel/protocol'

const TABS = ['Format', 'Time', 'Overlays', 'Traces', 'Export'] as const
type Tab = typeof TABS[number]

const DOWNSAMPLE_OPTS = [
  { value: '1', label: '1× (full)' },
  { value: '2', label: '2×' },
  { value: '4', label: '4×' },
  { value: '8', label: '8×' },
] as const

const ANNOTATION_KINDS = [
  { value: 'text' as const, label: 'Text' },
  { value: 'rect' as const, label: 'Rect' },
]

interface Props {
  caretPos: React.CSSProperties
  windowId: number
  sendAction: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
  onClose: () => void
}

const DEFAULT_PARAMS: MvxParams = {
  fps: 12, downsample: 1, stride: 1, t_start: 0, t_end: 0,
  cmap: undefined, clim: null, timestamp: true, scalebar: true, annotations: [],
}

export function MovieExportWizard({ caretPos, windowId, sendAction, onClose }: Props) {
  const [tab, setTab] = React.useState<Tab>('Format')

  // Mirror of the authoritative mvx_state (params/traces/time/n_frames/ffmpeg).
  const [ffmpegOk, setFfmpegOk] = React.useState(true)
  const [running, setRunning] = React.useState(false)
  const [nFrames, setNFrames] = React.useState(0)
  const [time, setTime] = React.useState<{ scale_s: number; units: string }>({ scale_s: 1, units: 'frame' })
  const [params, setParams] = React.useState<MvxParams>(DEFAULT_PARAMS)
  const [traces, setTraces] = React.useState<MvxTrace[]>([])
  const [status, setStatus] = React.useState('Preparing movie export…')
  const [dragOver, setDragOver] = React.useState(false)
  // Full time range in the movie's time units defaults until the first state.
  const gotState = React.useRef(false)

  // Open → mvx_open (backend probes ffmpeg + seeds params); unmount → mvx_close.
  useWizardLifecycle({
    windowId, sendAction, openAction: 'mvx_open', closeAction: 'mvx_close',
  })

  const tune = useDebouncedAction(sendAction, 'mvx_tune', windowId)

  // The authoritative state stream. The first state defines the full time range
  // and seeds cmap/levels/n_frames; subsequent states reflect our own tunes.
  useWizardEvent('spyde:mvx_state', windowId, (raw) => {
    const d = raw as unknown as MvxStateMessage
    setFfmpegOk(Boolean(d.ffmpeg_ok))
    setRunning(Boolean(d.running))
    setNFrames(Number(d.n_frames ?? 0))
    if (d.time) setTime({ scale_s: Number(d.time.scale_s ?? 1), units: String(d.time.units ?? 'frame') })
    if (d.params) setParams(p => ({ ...p, ...d.params }))
    setTraces(Array.isArray(d.traces) ? d.traces : [])
    if (!gotState.current) {
      gotState.current = true
      setStatus(d.ffmpeg_ok
        ? 'Tune the movie, then Export.'
        : 'ffmpeg not available — install/enable it to export.')
    }
  })

  // Export finished — surface the written file's basename.
  useWizardEvent('spyde:mvx_done', windowId, (raw) => {
    const path = String((raw as { path?: unknown }).path ?? '')
    const base = path.split(/[/\\]/).pop() || path
    setRunning(false)
    setStatus(base ? `Exported ${base}` : 'Movie exported.')
  })

  // Progress readout in the caret footer (also surfaces app-wide via StatusBar).
  useWizardEvent('spyde:progress', windowId, (raw) => {
    const d = raw as { done?: number; total?: number; label?: string }
    const total = Number(d.total ?? 0), done = Number(d.done ?? 0)
    if (total > 0 && done < total) {
      const pct = Math.round((done / total) * 100)
      setStatus(`${d.label || 'Exporting'} — ${pct}% (${done}/${total})`)
    }
  })

  // Patch local params + send the WHOLE (patched) params via mvx_tune — the
  // backend replaces its params/annotations wholesale from each tune.
  const patch = (next: Partial<MvxParams>) => {
    setParams(prev => {
      const merged = { ...prev, ...next }
      tune(() => tuneablePayload(merged))
      return merged
    })
  }

  // Seconds-scaled display of a frame-index time value.
  const asSeconds = (v: number) => v * time.scale_s
  const fromSeconds = (s: number) => (time.scale_s ? s / time.scale_s : s)

  // The effective time range: 0..(n-1) until the user narrows it.
  const tStart = params.t_start || 0
  const tEnd = params.t_end || Math.max(0, nFrames - 1)

  // Computed output info: frames ≈ range/stride, duration ≈ frames/fps.
  const stride = Math.max(1, params.stride || 1)
  const outFrames = Math.max(0, Math.floor((tEnd - tStart) / stride) + 1)
  const duration = params.fps > 0 ? outFrames / params.fps : 0

  // ── Annotations ──────────────────────────────────────────────────────────
  const addAnnotation = () => {
    const ann: MvxAnnotation = { kind: 'text', time_range: [tStart, tEnd], text: 'Label', x: 0, y: 0 }
    patch({ annotations: [...params.annotations, ann] })
  }
  const updateAnnotation = (i: number, next: Partial<MvxAnnotation>) => {
    const list = params.annotations.map((a, j) => (j === i ? { ...a, ...next } : a))
    patch({ annotations: list })
  }
  const removeAnnotation = (i: number) => {
    patch({ annotations: params.annotations.filter((_, j) => j !== i) })
  }

  // ── Traces (drop a 1-D plot window) ──────────────────────────────────────
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const raw = e.dataTransfer.getData(WINDOW_DRAG_MIME)
    const src = parseInt(raw, 10)
    if (!Number.isNaN(src)) {
      sendAction('mvx_add_trace', { source_window_id: src }, windowId)
      setStatus('Adding trace…')
    }
  }
  const onDragOver = (e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes(WINDOW_DRAG_MIME)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    if (!dragOver) setDragOver(true)
  }

  // ── Export / cancel ──────────────────────────────────────────────────────
  const doExport = async () => {
    if (!ffmpegOk) return
    const path = await window.electron.reportExportDialog('mp4', 'movie.mp4')
    if (!path) return
    setStatus('Rendering movie…')
    sendAction('mvx_run', { path }, windowId)
  }
  const doCancel = () => {
    sendAction('mvx_cancel', {}, windowId)
    setStatus('Cancelling…')
  }

  return (
    <WizardShell testid="mvx-wizard" title="Export Movie" posStyle={caretPos}
      onClose={onClose} closeTestid="mvx-close" status={status} statusTestid="mvx-status"
      width={280}>
      <TabRow tabs={TABS} active={tab} onSelect={setTab} testid={(t) => `mvx-tab-${t}`} />

      {tab === 'Format' && (
        <div style={S.page}>
          <Field label="Frame rate (fps)">
            <NumInput testid="mvx-fps" value={params.fps} step="1" width={64}
              onChange={(n) => patch({ fps: clamp(n, 1, 60) })} />
          </Field>
          <Field label="Spatial downsample">
            <Select testid="mvx-downsample" value={String(params.downsample) as '1' | '2' | '4' | '8'}
              options={DOWNSAMPLE_OPTS} onChange={(v) => patch({ downsample: Number(v) })} />
          </Field>
          <Field label="Temporal stride">
            <NumInput testid="mvx-stride" value={params.stride} step="1" width={64}
              onChange={(n) => patch({ stride: Math.max(1, Math.round(n)) })} />
          </Field>
          <div style={S.hint} data-testid="mvx-output-info">
            ~{outFrames} frames · ~{duration.toFixed(1)} s at {params.fps} fps
          </div>
        </div>
      )}

      {tab === 'Time' && (
        <div style={S.page}>
          <div style={S.hint}>
            Range in {time.units} (shown in seconds). Full movie: 0 – {asSeconds(Math.max(0, nFrames - 1)).toFixed(2)} s.
          </div>
          <Field label={`Start (s)`}>
            <NumInput testid="mvx-t-start" value={round2(asSeconds(tStart))} step="0.01" width={72}
              onChange={(s) => patch({ t_start: clamp(Math.round(fromSeconds(s)), 0, tEnd) })} />
          </Field>
          <Field label={`End (s)`}>
            <NumInput testid="mvx-t-end" value={round2(asSeconds(tEnd))} step="0.01" width={72}
              onChange={(s) => patch({ t_end: clamp(Math.round(fromSeconds(s)), tStart, Math.max(0, nFrames - 1)) })} />
          </Field>
        </div>
      )}

      {tab === 'Overlays' && (
        <div style={S.page}>
          <Check testid="mvx-timestamp" checked={Boolean(params.timestamp)} label="Timestamp"
            onChange={(b) => patch({ timestamp: b })} />
          <Check testid="mvx-scalebar" checked={Boolean(params.scalebar)} label="Scale bar"
            onChange={(b) => patch({ scalebar: b })} />
          <div style={{ ...S.hint, marginTop: 4 }}>Annotations (shown within their time window)</div>
          <div data-testid="mvx-annotations" style={S.cifList}>
            {params.annotations.map((a, i) => (
              <div key={i} style={annRow} data-testid={`mvx-annotation-${i}`}>
                <select data-testid={`mvx-ann-kind-${i}`} value={a.kind} style={{ ...S.sel, padding: '2px 4px' }}
                  onChange={(e) => updateAnnotation(i, { kind: e.target.value as MvxAnnotation['kind'] })}>
                  {ANNOTATION_KINDS.map(k => <option key={k.value} value={k.value}>{k.label}</option>)}
                </select>
                {a.kind === 'text' && (
                  <input data-testid={`mvx-ann-text-${i}`} type="text" value={a.text ?? ''}
                    style={{ ...S.num, width: 60 }}
                    onChange={(e) => updateAnnotation(i, { text: e.target.value })} />
                )}
                <input data-testid={`mvx-ann-t0-${i}`} type="number" value={a.time_range[0]} title="t0 (frame)"
                  style={{ ...S.num, width: 44 }}
                  onChange={(e) => updateAnnotation(i, { time_range: [Number(e.target.value), a.time_range[1]] })} />
                <input data-testid={`mvx-ann-t1-${i}`} type="number" value={a.time_range[1]} title="t1 (frame)"
                  style={{ ...S.num, width: 44 }}
                  onChange={(e) => updateAnnotation(i, { time_range: [a.time_range[0], Number(e.target.value)] })} />
                <button data-testid={`mvx-ann-remove-${i}`} style={S.close} title="Remove"
                  onClick={() => removeAnnotation(i)}>✕</button>
              </div>
            ))}
          </div>
          <button data-testid="mvx-add-annotation" style={{ ...S.fileBtn, alignSelf: 'flex-start' }}
            onClick={addAnnotation}>＋ Add annotation</button>
        </div>
      )}

      {tab === 'Traces' && (
        <div style={S.page}>
          <div style={S.hint}>Drag a 1-D plot window here to overlay it as a trace synced to the movie time.</div>
          <div data-testid="mvx-trace-drop" onDrop={onDrop} onDragOver={onDragOver}
            onDragLeave={() => setDragOver(false)}
            style={{ ...dropSlot, ...(dragOver ? dropSlotActive : {}) }}>
            Drop a 1-D plot here
          </div>
          <div style={S.cifList}>
            {traces.map(t => (
              <div key={t.id} style={S.cifRow} data-testid={`mvx-trace-${t.id}`}>
                <span style={{ ...S.cifName, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: t.color, flexShrink: 0 }} />
                  {t.label}{t.units ? ` (${t.units})` : ''}
                </span>
                <button data-testid={`mvx-trace-remove-${t.id}`} style={S.close} title="Remove trace"
                  onClick={() => sendAction('mvx_remove_trace', { trace_id: t.id }, windowId)}>✕</button>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'Export' && (
        <div style={S.page}>
          {!ffmpegOk && (
            <div data-testid="mvx-ffmpeg-warning" style={warnBox}>
              ⚠ ffmpeg is not available. Movie export is disabled.
            </div>
          )}
          <div style={S.hint}>
            ~{outFrames} frames · ~{duration.toFixed(1)} s at {params.fps} fps
            {params.downsample > 1 ? ` · ${params.downsample}× downsampled` : ''}
          </div>
          {running ? (
            <button data-testid="mvx-cancel" style={{ ...S.primary, background: '#f38ba8' }}
              onClick={doCancel}>Cancel Export</button>
          ) : (
            <button data-testid="mvx-export" style={{ ...S.primary, ...(ffmpegOk ? {} : disabledBtn) }}
              disabled={!ffmpegOk} onClick={doExport}>Export Movie…</button>
          )}
        </div>
      )}
    </WizardShell>
  )
}

/** The subset of params sent on a tune (cmap/clim seeded backend-side; we only
 *  echo them back when set, so a null clim = "auto" survives). */
function tuneablePayload(p: MvxParams): Record<string, unknown> {
  const out: Record<string, unknown> = {
    fps: p.fps, downsample: p.downsample, stride: p.stride,
    t_start: p.t_start, t_end: p.t_end,
    timestamp: p.timestamp, scalebar: p.scalebar,
    annotations: p.annotations,
  }
  if (p.cmap !== undefined) out.cmap = p.cmap
  if (p.clim !== undefined) out.clim = p.clim
  return out
}

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n))
const round2 = (n: number) => Math.round(n * 100) / 100

const dropSlot: React.CSSProperties = {
  border: '1px dashed #45475a', borderRadius: 6, padding: '14px 8px',
  textAlign: 'center', fontSize: 11, color: '#6c7086', background: '#11111b',
  transition: 'border-color 120ms, color 120ms',
}
const dropSlotActive: React.CSSProperties = { borderColor: '#89b4fa', color: '#89b4fa' }
const annRow: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 4, background: '#11111b',
  borderRadius: 4, padding: '2px 4px',
}
const warnBox: React.CSSProperties = {
  fontSize: 10, color: '#f38ba8', background: 'rgba(243,139,168,0.12)',
  border: '1px solid rgba(243,139,168,0.4)', borderRadius: 4, padding: '5px 7px',
}
const disabledBtn: React.CSSProperties = { background: '#45475a', color: '#6c7086', cursor: 'not-allowed' }
