/**
 * MovieEditor.tsx — the full-screen Movie editor.
 *
 * THE MODEL: a movie IS the source in-situ tree's LIVE 2-D signal figure + its 1-D
 * time navigator. This editor SURFACES the tree's real signal figure (re-parented
 * iframe, keyed by movie_state.signal_fig_id) and drives the REAL navigator:
 *
 *   • Scrubber / Play drive movie_scrub / movie_play → the real navigator (the
 *     signal figure repaints through the real lazy pipeline + GPU tile mode).
 *   • A bottom TIMELINE dock: overlay lanes (Text / Signal / ROI) as draggable
 *     [t0,t1] clips + a SPEED track (slow / fast-forward / hold segments that remap
 *     source→output time on export). Clicking a clip selects it.
 *   • A RIGHT INSPECTOR edits the selected clip (text / colour / size / position)
 *     and the render controls (contrast, colormap, axes + scale-bar toggles) and
 *     the export options (fps, downsample, range, size readout).
 *   • Overlays are anyplotlib annotation widgets on the live signal plot (backend);
 *     the editor drives their time-gating + style. Crop zooms the figure into the
 *     region with a dimmed outside.
 *
 * The header is a window drag-region padded clear of the native title-bar controls
 * (Windows titleBarOverlay is 38 px on the right).
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { SeamlessFigureFrame } from './ReportFigureCell'
import { WINDOW_DRAG_MIME } from '../kernel/dnd'
import type { MovieStateMessage, MovieParams, MovieAnnotation, MovieSpeedSegment } from '../kernel/protocol'

interface Props {
  cellId: string
  sendAction: (action: string, payload?: Record<string, unknown>) => void
  onClose: () => void
}

const CMAPS = ['gray', 'viridis', 'magma', 'inferno', 'plasma', 'cividis', 'hot', 'jet']
const DOWNSAMPLES = [1, 2, 4, 8]
// Speed presets (a segment's multiplier). 0 = hold (freeze); <1 slow-mo; >1 ff.
const SPEEDS = [0, 0.25, 0.5, 1, 2, 4, 8]
const SPEED_LABEL = (s: number) => (s === 0 ? 'hold' : `${s}×`)

// A selected timeline clip (which lane + index), so the inspector edits it.
type Selection =
  | { lane: 'text' | 'roi'; index: number }
  | { lane: 'signal'; index: number }
  | { lane: 'speed'; index: number }
  | null

export function MovieEditor({ cellId, sendAction, onClose }: Props) {
  const { state, iframeRefs, replayState } = useSpyDE()
  const [st, setSt] = React.useState<MovieStateMessage | null>(null)
  const [t, setT] = React.useState(0)                 // scrub frame index
  const [playing, setPlaying] = React.useState(false)
  const [running, setRunning] = React.useState(false)
  const [status, setStatus] = React.useState('Loading movie…')
  const [showNav, setShowNav] = React.useState(false)
  const [cropMode, setCropMode] = React.useState(false)
  const [sel, setSel] = React.useState<Selection>(null)
  const statusKey = React.useRef<string>('')

  const scrubTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const tuneTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  React.useEffect(() => () => {
    if (scrubTimer.current) clearTimeout(scrubTimer.current)
    if (tuneTimer.current) clearTimeout(tuneTimer.current)
  }, [])

  const scrub = React.useCallback((frame: number) => {
    if (scrubTimer.current) clearTimeout(scrubTimer.current)
    scrubTimer.current = setTimeout(
      () => sendAction('movie_scrub', { cell_id: cellId, t: frame }), 40)
  }, [sendAction, cellId])

  const tune = React.useCallback((patch: Record<string, unknown>) => {
    if (tuneTimer.current) clearTimeout(tuneTimer.current)
    tuneTimer.current = setTimeout(
      () => sendAction('movie_tune', { cell_id: cellId, ...patch }), 120)
  }, [sendAction, cellId])

  // ── movie_state / movie_done / movie_frame subscriptions ──────────────────
  React.useEffect(() => {
    const onState = (e: Event) => {
      const d = (e as CustomEvent).detail as MovieStateMessage
      if (d.cell_id !== cellId) return
      setSt(d)
      setRunning(Boolean(d.running))
      const key = d.has_source ? 'src' : 'nosrc'
      if (statusKey.current !== key) {
        statusKey.current = key
        setStatus(d.has_source
          ? (d.ffmpeg_ok ? 'Scrub, annotate, and tune — then Export.'
            : 'ffmpeg missing — mp4 disabled, gif still works.')
          : 'Pick an in-situ movie signal to start.')
      }
    }
    // The backend reports the navigator's current frame as playback / scrub moves.
    const onFrame = (e: Event) => {
      const d = (e as CustomEvent).detail as { cell_id: string; t: number; playing?: boolean }
      if (d.cell_id !== cellId) return
      if (typeof d.t === 'number') setT(d.t)
      if (typeof d.playing === 'boolean') setPlaying(d.playing)
    }
    const onDone = (e: Event) => {
      const d = (e as CustomEvent).detail as { cell_id: string; path: string; frames: number }
      if (d.cell_id !== cellId) return
      setRunning(false)
      const base = d.path.split(/[/\\]/).pop() || d.path
      setStatus(`Exported ${base} (${d.frames} frames).`)
    }
    const onProgress = (e: Event) => {
      const d = (e as CustomEvent).detail as { done?: number; total?: number }
      const total = Number(d.total ?? 0), done = Number(d.done ?? 0)
      if (total > 0 && done < total) setStatus(`Encoding — ${Math.round((done / total) * 100)}%`)
    }
    window.addEventListener('spyde:movie_state', onState)
    window.addEventListener('spyde:movie_frame', onFrame)
    window.addEventListener('spyde:movie_done', onDone)
    window.addEventListener('spyde:progress', onProgress)
    return () => {
      window.removeEventListener('spyde:movie_state', onState)
      window.removeEventListener('spyde:movie_frame', onFrame)
      window.removeEventListener('spyde:movie_done', onDone)
      window.removeEventListener('spyde:progress', onProgress)
    }
  }, [cellId])

  const seeded = React.useRef(false)
  React.useEffect(() => {
    if (st?.has_source && !seeded.current) {
      seeded.current = true
      setT(Number(st.current_index ?? st.params.t_start ?? 0))
    }
  }, [st])

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === ' ' && st?.has_source) { e.preventDefault(); togglePlay() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onClose, st, playing])

  const nFrames = Number(st?.n_frames ?? 0)
  const params: MovieParams = st?.params ?? {}
  const crop = st?.crop ?? null
  const scaleS = Number(st?.time?.scale_s ?? 0)
  const timeUnits = String(st?.time?.units ?? '')
  const duration = scaleS > 0 ? Math.max(0, nFrames - 1) * scaleS : 0
  const secLabel = (frame: number) =>
    scaleS > 0 ? `${(frame * scaleS).toFixed(2)} ${timeUnits || 's'}` : `frame ${frame}`

  const signalFigId = st?.signal_fig_id ?? null
  const signalWindowId = st?.signal_window_id ?? null
  const signalFig = React.useMemo(() => {
    if (signalWindowId == null) return null
    const win = state.windows.get(signalWindowId)
    return win?.figures?.find(fg => fg.figId === signalFigId) ?? win?.figures?.[0] ?? null
  }, [state.windows, signalWindowId, signalFigId])
  const navFigId = st?.nav_fig_id ?? null
  const navFig = React.useMemo(() => {
    if (!navFigId) return null
    for (const win of state.windows.values()) {
      const f = win.figures?.find(fg => fg.figId === navFigId)
      if (f) return f
    }
    return null
  }, [state.windows, navFigId])

  const doScrub = (frame: number) => {
    const f = Math.max(0, Math.min(frame, Math.max(0, nFrames - 1)))
    setT(f)
    scrub(f)
  }
  const togglePlay = () => {
    if (playing) { sendAction('movie_stop', { cell_id: cellId }); setPlaying(false) }
    else { sendAction('movie_play', { cell_id: cellId }); setPlaying(true) }
  }

  const patchParams = (p: Partial<MovieParams>) => {
    setSt(s => (s ? { ...s, params: { ...s.params, ...p } } : s))
    tune({ params: p })
  }

  // ── overlay + speed lists ─────────────────────────────────────────────────
  const anns = st?.annotations ?? []
  const textOverlays = st?.text_overlays ?? []
  const speedSegs = st?.speed_segments ?? []
  const setAnnotations = (list: MovieAnnotation[]) => {
    setSt(s => (s ? { ...s, annotations: list } : s)); tune({ annotations: list })
  }
  const setTextOverlays = (list: typeof textOverlays) => {
    setSt(s => (s ? { ...s, text_overlays: list } : s)); tune({ text_overlays: list })
  }
  const setSpeedSegs = (list: MovieSpeedSegment[]) => {
    setSt(s => (s ? { ...s, speed_segments: list } : s)); tune({ speed_segments: list })
  }
  const fullRange = (): [number, number] => [0, duration]
  const addText = () => {
    setAnnotations([...anns, { kind: 'text', text: 'Label', xy: [24, 24 + 28 * anns.filter(a => a.kind === 'text').length],
      size: 22, color: '#ffffff', time_range: fullRange() }])
    setSel({ lane: 'text', index: anns.length })
  }
  const addRoi = () => {
    setAnnotations([...anns, { kind: 'rect', xy: [20, 20], wh: [80, 80], color: '#f9e2af', width: 4, time_range: fullRange() }])
    setSel({ lane: 'roi', index: anns.length })
  }
  // A speed segment at the current playhead, ~10% of the duration, default slow-mo.
  const addSpeedSeg = () => {
    const t0 = t * scaleS
    const t1 = Math.min(duration, t0 + Math.max(scaleS, duration * 0.1))
    setSpeedSegs([...speedSegs, { time_range: [t0, t1], speed: 0.25 }])
    setSel({ lane: 'speed', index: speedSegs.length })
  }

  // ── drop a window onto the figure ─────────────────────────────────────────
  const [dropOver, setDropOver] = React.useState(false)
  const onSignalDrop = (e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes(WINDOW_DRAG_MIME)) return
    e.preventDefault(); e.stopPropagation(); setDropOver(false)
    const wid = parseInt(e.dataTransfer.getData(WINDOW_DRAG_MIME), 10)
    if (!Number.isNaN(wid)) {
      const fw = st?.frame_size?.[0] ?? 0, fh = st?.frame_size?.[1] ?? 0
      sendAction('movie_drop_window', { cell_id: cellId, source_window_id: wid,
        xy: [Math.round(fw * 0.06), Math.round(fh * 0.9) - 30 * textOverlays.length] })
    }
  }
  const onSignalDragOver = (e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes(WINDOW_DRAG_MIME)) return
    e.preventDefault(); e.stopPropagation(); e.dataTransfer.dropEffect = 'copy'
    if (!dropOver) setDropOver(true)
  }

  const frac = (sec: number) => (duration > 0 ? Math.max(0, Math.min(1, sec / duration)) : 0)
  const isMac = (window as unknown as { electron?: { platform?: string } }).electron?.platform === 'darwin'

  return (
    <div style={styles.overlay} data-testid="movie-editor">
      {/* Header — a window drag region, padded clear of the native title-bar
          controls (Windows: right overlay; mac: left traffic lights). */}
      <div style={{ ...styles.header, WebkitAppRegion: 'drag',
        paddingLeft: isMac ? 84 : 16, paddingRight: isMac ? 12 : 150 } as React.CSSProperties}>
        <span style={styles.title}>🎬 Movie editor{st?.source_title ? ` — ${st.source_title}` : ''}</span>
        <div style={{ flex: 1 }} />
        <label style={{ ...styles.navToggle, WebkitAppRegion: 'no-drag' } as React.CSSProperties}>
          <input type="checkbox" data-testid="movie-show-nav" checked={showNav}
            onChange={(e) => setShowNav(e.target.checked)} /> navigator
        </label>
        <button data-testid="movie-editor-close"
          style={{ ...styles.closeBtn, WebkitAppRegion: 'no-drag' } as React.CSSProperties}
          onClick={onClose}>✕ Close</button>
      </div>

      <div style={styles.body}>
        {/* Left rail — just the ADD buttons (compact). */}
        <div style={styles.rail}>
          <div style={styles.railHead}>Add overlay</div>
          <button style={styles.toolBtn} data-testid="movie-add-text" onClick={addText}>＋ Text</button>
          <button style={styles.toolBtn} data-testid="movie-add-roi" onClick={addRoi}>＋ ROI box</button>
          <button style={styles.toolBtn} data-testid="movie-add-speed" onClick={addSpeedSeg}>＋ Speed segment</button>
          <div style={{ height: 8 }} />
          <div style={styles.railHead}>Crop</div>
          <button data-testid="movie-crop-toggle"
            style={cropMode ? { ...styles.toolBtn, ...styles.toolBtnActive } : styles.toolBtn}
            onClick={() => { const n = !cropMode; setCropMode(n); sendAction('movie_crop_mode', { cell_id: cellId, on: n }) }}>
            {cropMode ? '✓ Cropping — drag' : '⛶ Crop the frame'}</button>
          {crop && (
            <button style={styles.toolBtn} data-testid="movie-crop-clear"
              onClick={() => { setCropMode(false); sendAction('movie_crop_mode', { cell_id: cellId, clear: true }) }}>
              ✕ Clear crop ({crop[2] - crop[0]}×{crop[3] - crop[1]})</button>
          )}
        </div>

        {/* Centre: figure + transport + timeline dock. */}
        <div style={styles.center}>
          <div style={styles.figRow}>
            <div style={styles.figWrap} data-testid="movie-figure-wrap"
              onDrop={onSignalDrop} onDragOver={onSignalDragOver} onDragLeave={() => setDropOver(false)}>
              {signalFig && signalFig.figId ? (
                <SeamlessFigureFrame figId={signalFig.figId} filePath={signalFig.filePath}
                  title="Movie" iframeRefs={iframeRefs} replayState={replayState} />
              ) : (
                <div style={styles.figPlaceholder} data-testid="movie-figure-empty">
                  {st?.has_source ? 'Loading the movie figure…' : 'No signal assigned yet.'}
                </div>
              )}
              {dropOver && (
                <div style={styles.figDropHint} data-testid="movie-signal-drop-hint">
                  Drop a 1-D signal (live value) or a 2-D image (overlaid) onto the movie
                </div>
              )}
            </div>
            {showNav && navFig && navFig.figId && (
              <div style={styles.navWrap} data-testid="movie-nav-wrap">
                <SeamlessFigureFrame figId={navFig.figId} filePath={navFig.filePath}
                  title="Navigator" iframeRefs={iframeRefs} replayState={replayState} />
              </div>
            )}
          </div>

          {/* Transport: Play + scrubber. */}
          <div style={styles.scrubRow}>
            <button data-testid="movie-play" style={styles.playBtn}
              disabled={!st?.has_source} onClick={togglePlay} title="Play / pause (space)">
              {playing ? '⏸' : '▶'}</button>
            <span style={styles.timeLabel} data-testid="movie-time-label">{secLabel(t)}</span>
            <input type="range" data-testid="movie-scrubber" min={0} max={Math.max(0, nFrames - 1)}
              value={t} onChange={(e) => doScrub(Number(e.target.value))}
              style={styles.scrubber} disabled={!st?.has_source} />
            <span style={styles.timeLabel}>{nFrames > 0 ? `${t} / ${nFrames - 1}` : '—'}</span>
          </div>

          {/* Timeline dock: overlay lanes + speed track. */}
          <div style={styles.timeline} data-testid="movie-timeline">
            <TimelineLane label="Text" testid="movie-lane-text">
              <Playhead frac={frac(t * scaleS)} />
              {anns.map((a, i) => a.kind === 'text' ? (
                <Clip key={i} testid={`movie-clip-text-${i}`} color="#89b4fa"
                  t0={frac(a.time_range?.[0] ?? 0)} t1={frac(a.time_range?.[1] ?? duration)}
                  label={a.text || 'text'} selected={sel?.lane === 'text' && sel.index === i}
                  onSelect={() => setSel({ lane: 'text', index: i })}
                  onMove={(n0, n1) => setAnnotations(anns.map((x, j) => j === i ? { ...x, time_range: [n0 * duration, n1 * duration] } : x))}
                  onRemove={() => { setAnnotations(anns.filter((_, j) => j !== i)); setSel(null) }} />
              ) : null)}
            </TimelineLane>
            <TimelineLane label="Signal" testid="movie-lane-signal">
              <Playhead frac={frac(t * scaleS)} />
              {textOverlays.map((o, i) => (
                <Clip key={i} testid={`movie-clip-signal-${i}`} color="#a6e3a1"
                  t0={frac(o.time_range?.[0] ?? 0)} t1={frac(o.time_range?.[1] ?? duration)}
                  label={o.label || 'signal'} selected={sel?.lane === 'signal' && sel.index === i}
                  onSelect={() => setSel({ lane: 'signal', index: i })}
                  onMove={(n0, n1) => setTextOverlays(textOverlays.map((x, j) => j === i ? { ...x, time_range: [n0 * duration, n1 * duration] as [number, number] } : x))}
                  onRemove={() => { setTextOverlays(textOverlays.filter((_, j) => j !== i)); setSel(null) }} />
              ))}
            </TimelineLane>
            <TimelineLane label="ROI" testid="movie-lane-roi">
              <Playhead frac={frac(t * scaleS)} />
              {anns.map((a, i) => a.kind === 'rect' ? (
                <Clip key={i} testid={`movie-clip-roi-${i}`} color="#f9e2af"
                  t0={frac(a.time_range?.[0] ?? 0)} t1={frac(a.time_range?.[1] ?? duration)}
                  label="ROI" selected={sel?.lane === 'roi' && sel.index === i}
                  onSelect={() => setSel({ lane: 'roi', index: i })}
                  onMove={(n0, n1) => setAnnotations(anns.map((x, j) => j === i ? { ...x, time_range: [n0 * duration, n1 * duration] } : x))}
                  onRemove={() => { setAnnotations(anns.filter((_, j) => j !== i)); setSel(null) }} />
              ) : null)}
            </TimelineLane>
            <TimelineLane label="Speed" testid="movie-lane-speed">
              <Playhead frac={frac(t * scaleS)} />
              {speedSegs.map((sg, i) => (
                <Clip key={i} testid={`movie-clip-speed-${i}`}
                  color={sg.speed === 0 ? '#f38ba8' : sg.speed < 1 ? '#94e2d5' : '#fab387'}
                  t0={frac(sg.time_range[0])} t1={frac(sg.time_range[1])}
                  label={SPEED_LABEL(sg.speed)} selected={sel?.lane === 'speed' && sel.index === i}
                  onSelect={() => setSel({ lane: 'speed', index: i })}
                  onMove={(n0, n1) => setSpeedSegs(speedSegs.map((x, j) => j === i ? { ...x, time_range: [n0 * duration, n1 * duration] } : x))}
                  onRemove={() => { setSpeedSegs(speedSegs.filter((_, j) => j !== i)); setSel(null) }} />
              ))}
            </TimelineLane>
          </div>
        </div>

        {/* Right inspector: selected-clip props OR render/export controls. */}
        <div style={styles.inspector} data-testid="movie-inspector">
          <Inspector sel={sel} st={st} anns={anns} textOverlays={textOverlays} speedSegs={speedSegs}
            speeds={SPEEDS} setAnnotations={setAnnotations} setTextOverlays={setTextOverlays}
            setSpeedSegs={setSpeedSegs} />
          <RenderControls params={params} patchParams={patchParams} />
          <ExportPanel st={st} nFrames={nFrames} running={running} params={params}
            patchParams={patchParams}
            onExport={async () => {
              const defaultName = st?.ffmpeg_ok ? 'movie.mp4' : 'movie.gif'
              const path = await window.electron.reportExportDialog('mp4', defaultName)
              if (!path) return
              setStatus('Rendering movie…'); setRunning(true)
              sendAction('movie_export', { cell_id: cellId, path })
            }}
            onCancel={() => { sendAction('movie_cancel', { cell_id: cellId }); setStatus('Cancelling…') }} />
        </div>
      </div>

      <div style={styles.statusBar} data-testid="movie-editor-status">{status}</div>
    </div>
  )
}

// ── inspector ────────────────────────────────────────────────────────────────

function Inspector({ sel, st, anns, textOverlays, speedSegs, speeds,
  setAnnotations, setTextOverlays, setSpeedSegs }: {
  sel: Selection; st: MovieStateMessage | null
  anns: MovieAnnotation[]; textOverlays: NonNullable<MovieStateMessage['text_overlays']>
  speedSegs: MovieSpeedSegment[]; speeds: number[]
  setAnnotations: (l: MovieAnnotation[]) => void
  setTextOverlays: (l: NonNullable<MovieStateMessage['text_overlays']>) => void
  setSpeedSegs: (l: MovieSpeedSegment[]) => void
}) {
  if (!sel) return (
    <div style={styles.inspSection}>
      <div style={styles.inspHead}>Selection</div>
      <div style={styles.hint}>Click a timeline clip to edit it, or add an overlay.</div>
    </div>
  )
  if ((sel.lane === 'text' || sel.lane === 'roi')) {
    const a = anns[sel.index]
    if (!a) return null
    const upd = (patch: Partial<MovieAnnotation>) =>
      setAnnotations(anns.map((x, j) => j === sel.index ? { ...x, ...patch } : x))
    const [fw, fh] = st?.frame_size ?? [0, 0]
    return (
      <div style={styles.inspSection} data-testid="movie-inspector-annotation">
        <div style={styles.inspHead}>{sel.lane === 'text' ? 'Text label' : 'ROI box'}</div>
        {a.kind === 'text' && (
          <Field label="Text">
            <input type="text" data-testid="movie-insp-text" value={a.text ?? ''} style={styles.inp}
              onChange={(e) => upd({ text: e.target.value })} />
          </Field>
        )}
        <Field label="Colour">
          <input type="color" data-testid="movie-insp-color" value={a.color || '#ffffff'} style={styles.color}
            onChange={(e) => upd({ color: e.target.value })} />
        </Field>
        {a.kind === 'text' && (
          <Field label="Size">
            <input type="number" data-testid="movie-insp-size" min={6} max={200} value={a.size ?? 22} style={styles.num}
              onChange={(e) => upd({ size: Math.max(6, Number(e.target.value)) })} />
          </Field>
        )}
        <Field label={`Position (0–${fw}, 0–${fh})`}>
          <div style={{ display: 'flex', gap: 4 }}>
            <input type="number" data-testid="movie-insp-x" value={a.xy?.[0] ?? 0} style={styles.num}
              onChange={(e) => upd({ xy: [Number(e.target.value), a.xy?.[1] ?? 0] })} />
            <input type="number" data-testid="movie-insp-y" value={a.xy?.[1] ?? 0} style={styles.num}
              onChange={(e) => upd({ xy: [a.xy?.[0] ?? 0, Number(e.target.value)] })} />
          </div>
        </Field>
      </div>
    )
  }
  if (sel.lane === 'signal') {
    const o = textOverlays[sel.index]
    if (!o) return null
    const upd = (patch: Partial<typeof o>) =>
      setTextOverlays(textOverlays.map((x, j) => j === sel.index ? { ...x, ...patch } : x))
    return (
      <div style={styles.inspSection} data-testid="movie-inspector-signal">
        <div style={styles.inspHead}>Signal-as-text</div>
        <Field label="Label"><input type="text" data-testid="movie-insp-siglabel" value={o.label ?? ''} style={styles.inp}
          onChange={(e) => upd({ label: e.target.value })} /></Field>
        <Field label="Format"><input type="text" value={o.fmt ?? ''} placeholder="{label} = {value:.1f} {units}" style={styles.inp}
          onChange={(e) => upd({ fmt: e.target.value })} /></Field>
        <Field label="Colour"><input type="color" value={o.color || '#ffffff'} style={styles.color}
          onChange={(e) => upd({ color: e.target.value })} /></Field>
      </div>
    )
  }
  // speed segment
  const sg = speedSegs[sel.index]
  if (!sg) return null
  const upd = (patch: Partial<MovieSpeedSegment>) =>
    setSpeedSegs(speedSegs.map((x, j) => j === sel.index ? { ...x, ...patch } : x))
  return (
    <div style={styles.inspSection} data-testid="movie-inspector-speed">
      <div style={styles.inspHead}>Speed segment</div>
      <Field label="Speed">
        <select data-testid="movie-insp-speed" value={String(sg.speed)} style={styles.sel}
          onChange={(e) => upd({ speed: Number(e.target.value) })}>
          {speeds.map(s => <option key={s} value={s}>{s === 0 ? 'hold (freeze)' : `${s}× ${s < 1 ? '(slow)' : s > 1 ? '(fast)' : ''}`}</option>)}
        </select>
      </Field>
      <div style={styles.hint}>Drag the segment on the Speed track to set its time span.
        Source time inside it plays at this multiplier (0 = hold).</div>
    </div>
  )
}

function RenderControls({ params, patchParams }: {
  params: MovieParams; patchParams: (p: Partial<MovieParams>) => void
}) {
  const clim = params.clim ?? null
  return (
    <div style={styles.inspSection} data-testid="movie-render-controls">
      <div style={styles.inspHead}>Render</div>
      <Field label="Colormap">
        <select data-testid="movie-cmap" value={String(params.cmap ?? 'gray')} style={styles.sel}
          onChange={(e) => patchParams({ cmap: e.target.value })}>
          {CMAPS.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </Field>
      <Field label="Contrast (min / max)">
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <input type="number" data-testid="movie-clim-lo" style={styles.num}
            value={clim ? clim[0] : ''} placeholder="auto"
            onChange={(e) => {
              const lo = e.target.value === '' ? null : Number(e.target.value)
              patchParams({ clim: lo === null ? null : [lo, clim ? clim[1] : lo + 1] })
            }} />
          <input type="number" data-testid="movie-clim-hi" style={styles.num}
            value={clim ? clim[1] : ''} placeholder="auto"
            onChange={(e) => {
              const hi = e.target.value === '' ? null : Number(e.target.value)
              patchParams({ clim: hi === null ? null : [clim ? clim[0] : hi - 1, hi] })
            }} />
          {clim && <button style={styles.miniBtn} data-testid="movie-clim-auto"
            onClick={() => patchParams({ clim: null })}>auto</button>}
        </div>
      </Field>
      <Check testid="movie-axes" checked={params.axes !== false} label="Axes"
        onChange={(b) => patchParams({ axes: b })} />
      <Check testid="movie-scalebar" checked={params.scalebar !== false} label="Scale bar"
        onChange={(b) => patchParams({ scalebar: b })} />
      <Check testid="movie-timestamp" checked={params.timestamp !== false} label="Timestamp"
        onChange={(b) => patchParams({ timestamp: b })} />
    </div>
  )
}

function ExportPanel({ st, nFrames, running, params, patchParams, onExport, onCancel }: {
  st: MovieStateMessage | null; nFrames: number; running: boolean
  params: MovieParams; patchParams: (p: Partial<MovieParams>) => void
  onExport: () => void; onCancel: () => void
}) {
  const fps = Number(params.fps ?? 12)
  const ds = Math.max(1, Number(params.downsample ?? 1))
  const info = st?.output_info
  const outFrames = info ? info.frames : 0
  const outW = info ? info.w : 0
  const outH = info ? info.h : 0
  const duration = fps > 0 ? outFrames / fps : 0
  const maxIdx = Math.max(0, nFrames - 1)
  return (
    <div style={styles.inspSection}>
      <div style={styles.inspHead}>Export</div>
      <Field label="Frame rate (fps)">
        <input type="number" data-testid="movie-fps" min={1} max={60} value={fps} style={styles.num}
          onChange={(e) => patchParams({ fps: Math.max(1, Math.min(60, Number(e.target.value))) })} />
      </Field>
      <Field label="Downsample">
        <select data-testid="movie-downsample" value={String(ds)} style={styles.sel}
          onChange={(e) => patchParams({ downsample: Number(e.target.value) })}>
          {DOWNSAMPLES.map(k => <option key={k} value={k}>{k}×</option>)}
        </select>
      </Field>
      <Field label={`Range (frames 0–${maxIdx})`}>
        <div style={{ display: 'flex', gap: 4 }}>
          <input type="number" data-testid="movie-tstart" min={0} max={maxIdx} value={Number(params.t_start ?? 0)} style={styles.num}
            onChange={(e) => patchParams({ t_start: Math.max(0, Math.min(maxIdx, Number(e.target.value))) })} />
          <input type="number" data-testid="movie-tend" min={0} max={maxIdx} value={Number(params.t_end ?? maxIdx)} style={styles.num}
            onChange={(e) => patchParams({ t_end: Math.max(0, Math.min(maxIdx, Number(e.target.value))) })} />
        </div>
      </Field>
      <div style={styles.readout} data-testid="movie-export-readout">
        <div><b>{outFrames}</b> frames · <b>{duration.toFixed(1)}s</b> @ {fps}fps</div>
        <div>Size: <b>{outW}×{outH}</b>{ds > 1 ? ` (${ds}× down)` : ''}</div>
        <div>{st?.ffmpeg_ok ? 'mp4 / gif' : 'gif only'}</div>
      </div>
      {running ? (
        <button data-testid="movie-export-cancel" style={{ ...styles.primary, background: '#f38ba8' }}
          onClick={onCancel}>Cancel</button>
      ) : (
        <button data-testid="movie-export-btn" style={styles.primary} onClick={onExport}>Export…</button>
      )}
    </div>
  )
}

// ── small pieces ──────────────────────────────────────────────────────────────

function TimelineLane({ label, testid, children }: { label: string; testid: string; children: React.ReactNode }) {
  return (
    <div style={styles.lane} data-testid={testid}>
      <span style={styles.laneLabel}>{label}</span>
      <div style={styles.laneTrack} data-lane-track="1">{children}</div>
    </div>
  )
}
const Playhead = ({ frac }: { frac: number }) => (
  <div style={{ ...styles.playhead, left: `${frac * 100}%` }} />
)

function Clip({ testid, color, t0, t1, label, selected, onSelect, onMove, onRemove }: {
  testid: string; color: string; t0: number; t1: number; label: string
  selected?: boolean; onSelect?: () => void
  onMove: (t0: number, t1: number) => void; onRemove: () => void
}) {
  const gesture = React.useRef<{ mode: 'move' | 'l' | 'r'; startX: number; t0: number; t1: number; w: number; moved: boolean } | null>(null)
  const begin = (mode: 'move' | 'l' | 'r') => (e: React.PointerEvent) => {
    e.preventDefault(); e.stopPropagation()
    const track = (e.currentTarget as HTMLElement).closest('[data-lane-track]') as HTMLElement | null
    const w = track?.clientWidth || 1
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    gesture.current = { mode, startX: e.clientX, t0, t1, w, moved: false }
  }
  const move = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g) return
    const d = (e.clientX - g.startX) / g.w
    if (Math.abs(d) > 0.002) g.moved = true
    let n0 = g.t0, n1 = g.t1
    if (g.mode === 'move') { n0 = g.t0 + d; n1 = g.t1 + d }
    else if (g.mode === 'l') n0 = g.t0 + d
    else n1 = g.t1 + d
    n0 = Math.max(0, Math.min(n0, 1)); n1 = Math.max(0, Math.min(n1, 1))
    if (n1 - n0 < 0.02) return
    onMove(n0, n1)
  }
  const end = (e: React.PointerEvent) => {
    if (!gesture.current) return
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId) } catch { /* */ }
    if (!gesture.current.moved) onSelect?.()      // a click (no drag) selects
    gesture.current = null
  }
  return (
    <div data-testid={testid}
      style={{ ...styles.clip, left: `${t0 * 100}%`, width: `${Math.max(0, t1 - t0) * 100}%`,
        background: color, ...(selected ? styles.clipSelected : {}) }}
      onPointerDown={begin('move')} onPointerMove={move} onPointerUp={end}
      title={`${label} — drag to move, edges to resize, ✕ to remove`}>
      <span style={styles.clipEdge} onPointerDown={begin('l')} onPointerMove={move} onPointerUp={end} />
      <span style={styles.clipLabel}>{label}</span>
      <span style={styles.clipX} data-testid={`${testid}-remove`}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => { e.stopPropagation(); onRemove() }}>✕</span>
      <span style={{ ...styles.clipEdge, right: 0, left: 'auto' }}
        onPointerDown={begin('r')} onPointerMove={move} onPointerUp={end} />
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label style={styles.field}><span style={styles.fieldLabel}>{label}</span>{children}</label>
}
function Check({ testid, checked, label, onChange }: { testid: string; checked: boolean; label: string; onChange: (b: boolean) => void }) {
  return <label style={styles.check}><input type="checkbox" data-testid={testid} checked={checked}
    onChange={(e) => onChange(e.target.checked)} /><span>{label}</span></label>
}

const styles: Record<string, React.CSSProperties> = {
  overlay: { position: 'fixed', inset: 0, zIndex: 9300, background: '#0e0e16', color: '#cdd6f4', display: 'flex', flexDirection: 'column' },
  header: { display: 'flex', alignItems: 'center', gap: 10, padding: '8px 16px', borderBottom: '1px solid #313244', flexShrink: 0, minHeight: 38 },
  title: { fontSize: 15, fontWeight: 600 },
  navToggle: { fontSize: 11, color: '#a6adc8', display: 'flex', alignItems: 'center', gap: 4 },
  closeBtn: { background: '#313244', color: '#cdd6f4', border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 13, cursor: 'pointer' },
  body: { flex: 1, display: 'flex', minHeight: 0 },
  rail: { display: 'flex', flexDirection: 'column', gap: 5, padding: 12, borderRight: '1px solid #313244', flexShrink: 0, width: 132, overflowY: 'auto' },
  railHead: { fontSize: 10, fontWeight: 700, color: '#89b4fa', textTransform: 'uppercase', letterSpacing: 0.4 },
  toolBtn: { background: '#1e1e2e', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 5, padding: '5px 8px', fontSize: 11.5, cursor: 'pointer', textAlign: 'left' },
  toolBtnActive: { background: '#89b4fa', color: '#11111b', borderColor: '#89b4fa', fontWeight: 700 },
  center: { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, padding: 14, gap: 10 },
  figRow: { flex: 1, display: 'flex', gap: 10, minHeight: 0 },
  figWrap: { flex: 1, position: 'relative', background: '#11111b', borderRadius: 8, border: '1px solid #313244', overflow: 'hidden', minWidth: 0 },
  navWrap: { width: 220, position: 'relative', background: '#11111b', borderRadius: 8, border: '1px solid #313244', overflow: 'hidden', flexShrink: 0 },
  figPlaceholder: { position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6c7086', fontSize: 13 },
  figDropHint: { position: 'absolute', inset: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', borderRadius: 8, border: '2px dashed #89b4fa', background: 'rgba(137,180,250,0.14)', color: '#89b4fa', fontSize: 14, fontWeight: 600, pointerEvents: 'none', zIndex: 5 },
  scrubRow: { display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 },
  playBtn: { width: 34, height: 30, borderRadius: 6, border: 'none', background: '#89b4fa', color: '#11111b', fontSize: 14, cursor: 'pointer', flexShrink: 0 },
  scrubber: { flex: 1, accentColor: '#89b4fa' },
  timeLabel: { fontSize: 12, color: '#a6adc8', minWidth: 74, textAlign: 'center' },
  timeline: { display: 'flex', flexDirection: 'column', gap: 3, flexShrink: 0, background: '#11111b', borderRadius: 8, border: '1px solid #313244', padding: 8 },
  lane: { display: 'flex', alignItems: 'center', gap: 8, height: 26 },
  laneLabel: { width: 52, fontSize: 10.5, color: '#a6adc8', flexShrink: 0, textAlign: 'right' },
  laneTrack: { position: 'relative', flex: 1, height: 22, background: '#181825', borderRadius: 4, overflow: 'hidden' },
  playhead: { position: 'absolute', top: 0, bottom: 0, width: 2, background: '#cdd6f4', zIndex: 3, pointerEvents: 'none' },
  clip: { position: 'absolute', top: 2, bottom: 2, borderRadius: 3, minWidth: 8, display: 'flex', alignItems: 'center', cursor: 'grab', overflow: 'hidden', color: '#11111b', fontSize: 10, fontWeight: 700 },
  clipSelected: { outline: '2px solid #cdd6f4', outlineOffset: -1, zIndex: 4 },
  clipEdge: { position: 'absolute', left: 0, top: 0, bottom: 0, width: 6, cursor: 'ew-resize', background: 'rgba(0,0,0,0.15)' },
  clipLabel: { flex: 1, textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', padding: '0 8px', pointerEvents: 'none' },
  clipX: { padding: '0 4px', cursor: 'pointer', color: '#11111b', fontSize: 10 },
  inspector: { width: 260, borderLeft: '1px solid #313244', padding: 12, flexShrink: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14 },
  inspSection: { display: 'flex', flexDirection: 'column', gap: 7 },
  inspHead: { fontSize: 11, fontWeight: 700, color: '#89b4fa', textTransform: 'uppercase', letterSpacing: 0.4 },
  field: { display: 'flex', flexDirection: 'column', gap: 3 },
  fieldLabel: { fontSize: 10.5, color: '#a6adc8' },
  inp: { background: '#11111b', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 5, padding: '4px 6px', fontSize: 12 },
  num: { background: '#11111b', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 5, padding: '4px 6px', fontSize: 12, width: 70 },
  sel: { background: '#11111b', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 5, padding: '4px 6px', fontSize: 12 },
  color: { width: 40, height: 26, padding: 0, border: '1px solid #313244', borderRadius: 4, background: 'none', cursor: 'pointer' },
  miniBtn: { background: '#313244', color: '#cdd6f4', border: 'none', borderRadius: 4, padding: '3px 7px', fontSize: 10.5, cursor: 'pointer' },
  check: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#cdd6f4' },
  hint: { fontSize: 10.5, color: '#6c7086', lineHeight: 1.5 },
  readout: { fontSize: 11, color: '#cdd6f4', background: '#11111b', borderRadius: 6, padding: '6px 8px', lineHeight: 1.5, display: 'flex', flexDirection: 'column', gap: 1 },
  primary: { background: '#89b4fa', color: '#11111b', border: 'none', borderRadius: 6, padding: '8px 12px', fontSize: 12.5, fontWeight: 700, cursor: 'pointer', marginTop: 4 },
  statusBar: { padding: '7px 16px', borderTop: '1px solid #313244', fontSize: 12, color: '#a6adc8', flexShrink: 0 },
}
