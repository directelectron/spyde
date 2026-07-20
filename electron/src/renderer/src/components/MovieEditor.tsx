/**
 * MovieEditor.tsx — the full-screen Movie editor.
 *
 * THE MODEL: a movie IS the source in-situ tree's LIVE 2-D signal figure + its 1-D
 * time navigator (the same data + navigator machinery). This editor does NOT render
 * its own frames — it SURFACES the tree's real signal figure and scrubs the real
 * navigator:
 *
 *   • The preview is the tree's live anyplotlib signal figure, mounted here via
 *     SeamlessFigureFrame keyed by `movie_state.signal_fig_id` (the figId is 1:1
 *     with an iframe, so mounting it here supersedes the MDI iframe while the editor
 *     is open; replayState rehydrates it; live nav frames follow automatically).
 *   • The scrubber drives the REAL navigator via `movie_scrub {t}` (the backend runs
 *     the playback primitive translate_pixels + delayed_update_data(force) → the
 *     signal figure repaints through the real lazy pipeline + GPU tile mode).
 *   • Below: an iMovie-style TIMELINE with lanes (Text / Signal-text / ROI / Freeze).
 *     Each overlay is a clip spanning [t0,t1]; drag to move / resize its edges →
 *     `movie_tune {annotations}` with the new time_range. A playhead marks the
 *     current frame.
 *   • Overlays are anyplotlib annotation widgets on the live signal plot (added
 *     backend-side); the editor drives their time-gating + style.
 *   • Export (mp4/gif) at the bottom with the output-size readout.
 */
import React from 'react'
import { useSpyDE } from '../kernel/SpyDEContext'
import { SeamlessFigureFrame } from './ReportFigureCell'
import type { MovieStateMessage, MovieParams, MovieAnnotation } from '../kernel/protocol'

interface Props {
  cellId: string
  sendAction: (action: string, payload?: Record<string, unknown>) => void
  onClose: () => void
}

const CMAPS = ['gray', 'viridis', 'magma', 'inferno', 'plasma', 'cividis', 'hot', 'jet']

// The lanes shown in the timeline (each overlay kind gets a row of clips).
type LaneKind = 'text' | 'signal' | 'roi' | 'freeze'
const LANES: { key: LaneKind; label: string; color: string }[] = [
  { key: 'text', label: 'Text', color: '#89b4fa' },
  { key: 'signal', label: 'Signal', color: '#a6e3a1' },
  { key: 'roi', label: 'ROI', color: '#f9e2af' },
  { key: 'freeze', label: 'Freeze', color: '#f38ba8' },
]

export function MovieEditor({ cellId, sendAction, onClose }: Props) {
  const { state, iframeRefs, replayState } = useSpyDE()
  const [st, setSt] = React.useState<MovieStateMessage | null>(null)
  const [t, setT] = React.useState(0)                 // scrub frame index
  const [running, setRunning] = React.useState(false)
  const [status, setStatus] = React.useState('Loading movie…')
  const [showNav, setShowNav] = React.useState(false)
  const statusKey = React.useRef<string>('')

  // Debounced scrub + tune (pending timers cleared on unmount).
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

  // ── movie_state / movie_done subscriptions (cell-scoped) ──────────────────
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
    window.addEventListener('spyde:movie_done', onDone)
    window.addEventListener('spyde:progress', onProgress)
    return () => {
      window.removeEventListener('spyde:movie_state', onState)
      window.removeEventListener('spyde:movie_done', onDone)
      window.removeEventListener('spyde:progress', onProgress)
    }
  }, [cellId])

  // Sync the local scrub index to the backend's authoritative current_index the
  // first time state arrives (so the slider starts where the navigator is).
  const seeded = React.useRef(false)
  React.useEffect(() => {
    if (st?.has_source && !seeded.current) {
      seeded.current = true
      setT(Number(st.current_index ?? st.params.t_start ?? 0))
    }
  }, [st])

  // ESC closes.
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const nFrames = Number(st?.n_frames ?? 0)
  const params: MovieParams = st?.params ?? {}
  const scaleS = Number(st?.time?.scale_s ?? 0)
  const timeUnits = String(st?.time?.units ?? '')
  const duration = scaleS > 0 ? Math.max(0, nFrames - 1) * scaleS : 0
  const secLabel = (frame: number) =>
    scaleS > 0 ? `${(frame * scaleS).toFixed(2)} ${timeUnits || 's'}` : `frame ${frame}`

  // Resolve the tree's live signal figure (re-parent its iframe here). The figId
  // comes from movie_state; the filePath from the MDI window's stored figure.
  const signalFigId = st?.signal_fig_id ?? null
  const signalWindowId = st?.signal_window_id ?? null
  const signalFig = React.useMemo(() => {
    if (signalWindowId == null) return null
    const win = state.windows.get(signalWindowId)
    const f = win?.figures?.find(fg => fg.figId === signalFigId) ?? win?.figures?.[0]
    return f ?? null
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

  const patchParams = (p: Partial<MovieParams>) => {
    setSt(s => (s ? { ...s, params: { ...s.params, ...p } } : s))
    tune({ params: p })
  }

  // ── overlay mutations (drive time_range from the timeline) ────────────────
  const anns = st?.annotations ?? []
  const freezes = st?.freezes ?? []
  const textOverlays = st?.text_overlays ?? []
  const setAnnotations = (list: MovieAnnotation[]) => {
    setSt(s => (s ? { ...s, annotations: list } : s))
    tune({ annotations: list })
  }
  const setFreezes = (list: { t: number; hold_s: number }[]) => {
    setSt(s => (s ? { ...s, freezes: list } : s))
    tune({ freezes: list })
  }
  // A new overlay spans the WHOLE movie by default (0 → duration) — the most
  // useful default and always a visible clip; drag its edges on the timeline to
  // limit it to a period. (Anchoring to the current frame could make a zero-width
  // clip when the playhead is at the last frame.)
  const fullRange = (): [number, number] => [0, duration]
  const addText = () => {
    setAnnotations([...anns, {
      kind: 'text', text: 'Label', xy: [20, 30], size: 22, color: '#ffffff',
      time_range: fullRange(),
    }])
  }
  const addRoi = () => {
    setAnnotations([...anns, {
      kind: 'rect', xy: [20, 20], wh: [80, 80], color: '#f9e2af', width: 4,
      time_range: fullRange(),
    }])
  }
  const addFreeze = () => setFreezes([...freezes, { t, hold_s: 1.0 }])

  // Timeline geometry: fraction of the duration for a given seconds value.
  const frac = (sec: number) => (duration > 0 ? Math.max(0, Math.min(1, sec / duration)) : 0)

  return (
    <div style={styles.overlay} data-testid="movie-editor">
      {/* Header. */}
      <div style={styles.header}>
        <span style={styles.title}>🎬 Movie editor{st?.source_title ? ` — ${st.source_title}` : ''}</span>
        <div style={{ flex: 1 }} />
        <label style={styles.navToggle}>
          <input type="checkbox" data-testid="movie-show-nav" checked={showNav}
            onChange={(e) => setShowNav(e.target.checked)} /> navigator
        </label>
        <button data-testid="movie-editor-close" style={styles.closeBtn} onClick={onClose}>✕ Close</button>
      </div>

      <div style={styles.body}>
        {/* Left tools + export. */}
        <div style={styles.rail}>
          <div style={styles.railGroup}>
            <div style={styles.railHead}>Colormap</div>
            <select data-testid="movie-cmap" value={String(params.cmap ?? 'gray')} style={styles.sel}
              onChange={(e) => patchParams({ cmap: e.target.value })}>
              {CMAPS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div style={styles.railGroup}>
            <div style={styles.railHead}>Add overlay</div>
            <button style={styles.toolBtn} data-testid="movie-add-text" onClick={addText}>＋ Text</button>
            <button style={styles.toolBtn} data-testid="movie-add-roi" onClick={addRoi}>＋ ROI box</button>
            <button style={styles.toolBtn} data-testid="movie-add-freeze" onClick={addFreeze}>
              ＋ Freeze @ {t}
            </button>
          </div>
          <div style={{ flex: 1 }} />
          <ExportPanel st={st} nFrames={nFrames} running={running}
            onExport={async () => {
              const defaultName = st?.ffmpeg_ok ? 'movie.mp4' : 'movie.gif'
              const path = await window.electron.reportExportDialog('mp4', defaultName)
              if (!path) return
              setStatus('Rendering movie…'); setRunning(true)
              sendAction('movie_export', { cell_id: cellId, path })
            }}
            onCancel={() => { sendAction('movie_cancel', { cell_id: cellId }); setStatus('Cancelling…') }} />
        </div>

        {/* Centre: the LIVE signal figure (+ optional navigator) + scrubber. */}
        <div style={styles.center}>
          <div style={styles.figRow}>
            <div style={styles.figWrap} data-testid="movie-figure-wrap">
              {signalFig && signalFig.figId ? (
                <SeamlessFigureFrame
                  figId={signalFig.figId}
                  filePath={signalFig.filePath}
                  title="Movie"
                  iframeRefs={iframeRefs}
                  replayState={replayState}
                />
              ) : (
                <div style={styles.figPlaceholder} data-testid="movie-figure-empty">
                  {st?.has_source ? 'Loading the movie figure…' : 'No signal assigned yet.'}
                </div>
              )}
            </div>
            {showNav && navFig && navFig.figId && (
              <div style={styles.navWrap} data-testid="movie-nav-wrap">
                <SeamlessFigureFrame
                  figId={navFig.figId}
                  filePath={navFig.filePath}
                  title="Navigator"
                  iframeRefs={iframeRefs}
                  replayState={replayState}
                />
              </div>
            )}
          </div>

          {/* Scrubber. */}
          <div style={styles.scrubRow}>
            <span style={styles.timeLabel} data-testid="movie-time-label">{secLabel(t)}</span>
            <input type="range" data-testid="movie-scrubber" min={0}
              max={Math.max(0, nFrames - 1)} value={t}
              onChange={(e) => doScrub(Number(e.target.value))}
              style={styles.scrubber} disabled={!st?.has_source} />
            <span style={styles.timeLabel}>{nFrames > 0 ? `${t} / ${nFrames - 1}` : '—'}</span>
          </div>

          {/* iMovie-style timeline: lanes of clips spanning [t0,t1]. */}
          <div style={styles.timeline} data-testid="movie-timeline">
            {LANES.map(lane => (
              <div key={lane.key} style={styles.lane} data-testid={`movie-lane-${lane.key}`}>
                <span style={styles.laneLabel}>{lane.label}</span>
                <div style={styles.laneTrack} data-lane-track="1">
                  {/* Playhead. */}
                  <div style={{ ...styles.playhead, left: `${frac(t * scaleS) * 100}%` }} />
                  {lane.key === 'text' && anns.map((a, i) => a.kind === 'text' ? (
                    <Clip key={i} testid={`movie-clip-text-${i}`} color={lane.color}
                      t0={frac(a.time_range?.[0] ?? 0)} t1={frac(a.time_range?.[1] ?? duration)}
                      label={a.text || 'text'}
                      onMove={(nt0, nt1) => setAnnotations(anns.map((x, j) => j === i
                        ? { ...x, time_range: [nt0 * duration, nt1 * duration] } : x))}
                      onRemove={() => setAnnotations(anns.filter((_, j) => j !== i))} />
                  ) : null)}
                  {lane.key === 'roi' && anns.map((a, i) => a.kind === 'rect' ? (
                    <Clip key={i} testid={`movie-clip-roi-${i}`} color={lane.color}
                      t0={frac(a.time_range?.[0] ?? 0)} t1={frac(a.time_range?.[1] ?? duration)}
                      label="ROI"
                      onMove={(nt0, nt1) => setAnnotations(anns.map((x, j) => j === i
                        ? { ...x, time_range: [nt0 * duration, nt1 * duration] } : x))}
                      onRemove={() => setAnnotations(anns.filter((_, j) => j !== i))} />
                  ) : null)}
                  {lane.key === 'signal' && textOverlays.map((o, i) => (
                    <Clip key={i} testid={`movie-clip-signal-${i}`} color={lane.color}
                      t0={frac(o.time_range?.[0] ?? 0)} t1={frac(o.time_range?.[1] ?? duration)}
                      label={o.label || 'signal'}
                      onMove={(nt0, nt1) => {
                        const list = textOverlays.map((x, j) => j === i
                          ? { ...x, time_range: [nt0 * duration, nt1 * duration] as [number, number] } : x)
                        setSt(s => (s ? { ...s, text_overlays: list } : s)); tune({ text_overlays: list })
                      }}
                      onRemove={() => {
                        const list = textOverlays.filter((_, j) => j !== i)
                        setSt(s => (s ? { ...s, text_overlays: list } : s)); tune({ text_overlays: list })
                      }} />
                  ))}
                  {lane.key === 'freeze' && freezes.map((f, i) => (
                    <div key={i} data-testid={`movie-clip-freeze-${i}`}
                      style={{ ...styles.freezeMark, left: `${frac(f.t * scaleS) * 100}%` }}
                      title={`Freeze ${f.hold_s}s at frame ${f.t}`}
                      onClick={() => setFreezes(freezes.filter((_, j) => j !== i))}>◆</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={styles.statusBar} data-testid="movie-editor-status">{status}</div>
    </div>
  )
}

// A draggable timeline clip spanning [t0,t1] (fractions 0..1 of the duration).
// Drag the body to move; drag either edge to resize. onMove gets new fractions.
function Clip({ testid, color, t0, t1, label, onMove, onRemove }: {
  testid: string; color: string; t0: number; t1: number; label: string
  onMove: (t0: number, t1: number) => void; onRemove: () => void
}) {
  const gesture = React.useRef<{ mode: 'move' | 'l' | 'r'; startX: number; t0: number; t1: number; w: number } | null>(null)
  const begin = (mode: 'move' | 'l' | 'r') => (e: React.PointerEvent) => {
    e.preventDefault(); e.stopPropagation()
    const track = (e.currentTarget as HTMLElement).closest('[data-lane-track]') as HTMLElement | null
    const w = track?.clientWidth || 1
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    gesture.current = { mode, startX: e.clientX, t0, t1, w }
  }
  const move = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g) return
    const d = (e.clientX - g.startX) / g.w
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
    gesture.current = null
  }
  return (
    <div data-testid={testid}
      style={{ ...styles.clip, left: `${t0 * 100}%`, width: `${Math.max(0, t1 - t0) * 100}%`, background: color }}
      onPointerDown={begin('move')} onPointerMove={move} onPointerUp={end}
      title={`${label} — drag to move, edges to resize, ✕ to remove`}>
      <span style={styles.clipEdge} onPointerDown={begin('l')} onPointerMove={move} onPointerUp={end} />
      <span style={styles.clipLabel}>{label}</span>
      <span style={styles.clipX} data-testid={`${testid}-remove`}
        onPointerDown={(e) => { e.stopPropagation() }}
        onClick={(e) => { e.stopPropagation(); onRemove() }}>✕</span>
      <span style={{ ...styles.clipEdge, right: 0, left: 'auto' }}
        onPointerDown={begin('r')} onPointerMove={move} onPointerUp={end} />
    </div>
  )
}

function ExportPanel({ st, nFrames, running, onExport, onCancel }: {
  st: MovieStateMessage | null; nFrames: number; running: boolean
  onExport: () => void; onCancel: () => void
}) {
  const p = st?.params ?? {}
  const fps = Number(p.fps ?? 12)
  const stride = Math.max(1, Number(p.stride ?? 1))
  const t0 = Number(p.t_start ?? 0), t1 = Number(p.t_end ?? Math.max(0, nFrames - 1))
  const outFrames = Math.max(0, Math.floor((t1 - t0) / stride) + 1)
  const duration = fps > 0 ? outFrames / fps : 0
  const crop = st?.crop
  const [fw, fh] = st?.frame_size ?? [0, 0]
  const ds = Math.max(1, Number(p.downsample ?? 1))
  const outW = Math.floor((crop ? crop[2] - crop[0] : fw) / ds)
  const outH = Math.floor((crop ? crop[3] - crop[1] : fh) / ds)
  return (
    <div style={styles.railGroup}>
      <div style={styles.railHead}>Export</div>
      <div style={styles.readout} data-testid="movie-export-readout">
        <div><b>{outFrames}</b> frames · <b>{duration.toFixed(1)}s</b> @ {fps}fps</div>
        <div>Size: <b>{outW}×{outH}</b>{ds > 1 ? ` (${ds}×)` : ''}</div>
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

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 9300, background: '#0e0e16', color: '#cdd6f4',
    display: 'flex', flexDirection: 'column',
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px',
    borderBottom: '1px solid #313244', flexShrink: 0,
  },
  title: { fontSize: 15, fontWeight: 600 },
  navToggle: { fontSize: 11, color: '#a6adc8', display: 'flex', alignItems: 'center', gap: 4 },
  closeBtn: {
    background: '#313244', color: '#cdd6f4', border: 'none', borderRadius: 6,
    padding: '6px 14px', fontSize: 13, cursor: 'pointer',
  },
  body: { flex: 1, display: 'flex', minHeight: 0 },
  rail: {
    display: 'flex', flexDirection: 'column', gap: 12, padding: 12,
    borderRight: '1px solid #313244', flexShrink: 0, width: 150, overflowY: 'auto',
  },
  railGroup: { display: 'flex', flexDirection: 'column', gap: 5 },
  railHead: { fontSize: 10, fontWeight: 700, color: '#89b4fa', textTransform: 'uppercase', letterSpacing: 0.4 },
  toolBtn: {
    background: '#1e1e2e', color: '#cdd6f4', border: '1px solid #313244', borderRadius: 5,
    padding: '5px 8px', fontSize: 11.5, cursor: 'pointer', textAlign: 'left',
  },
  sel: {
    background: '#11111b', color: '#cdd6f4', border: '1px solid #313244',
    borderRadius: 5, padding: '4px 6px', fontSize: 12,
  },
  center: { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, padding: 14, gap: 10 },
  figRow: { flex: 1, display: 'flex', gap: 10, minHeight: 0 },
  figWrap: {
    flex: 1, position: 'relative', background: '#11111b', borderRadius: 8,
    border: '1px solid #313244', overflow: 'hidden', minWidth: 0,
  },
  navWrap: {
    width: 220, position: 'relative', background: '#11111b', borderRadius: 8,
    border: '1px solid #313244', overflow: 'hidden', flexShrink: 0,
  },
  figPlaceholder: {
    position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
    justifyContent: 'center', color: '#6c7086', fontSize: 13,
  },
  scrubRow: { display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 },
  scrubber: { flex: 1, accentColor: '#89b4fa' },
  timeLabel: { fontSize: 12, color: '#a6adc8', minWidth: 74, textAlign: 'center' },
  timeline: {
    display: 'flex', flexDirection: 'column', gap: 3, flexShrink: 0,
    background: '#11111b', borderRadius: 8, border: '1px solid #313244', padding: 8,
  },
  lane: { display: 'flex', alignItems: 'center', gap: 8, height: 26 },
  laneLabel: { width: 52, fontSize: 10.5, color: '#a6adc8', flexShrink: 0, textAlign: 'right' },
  laneTrack: {
    position: 'relative', flex: 1, height: 22, background: '#181825',
    borderRadius: 4, overflow: 'hidden',
  },
  playhead: {
    position: 'absolute', top: 0, bottom: 0, width: 2, background: '#cdd6f4',
    zIndex: 3, pointerEvents: 'none',
  },
  clip: {
    position: 'absolute', top: 2, bottom: 2, borderRadius: 3, minWidth: 8,
    display: 'flex', alignItems: 'center', cursor: 'grab', overflow: 'hidden',
    color: '#11111b', fontSize: 10, fontWeight: 700,
  },
  clipEdge: {
    position: 'absolute', left: 0, top: 0, bottom: 0, width: 6, cursor: 'ew-resize',
    background: 'rgba(0,0,0,0.15)',
  },
  clipLabel: {
    flex: 1, textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis',
    whiteSpace: 'nowrap', padding: '0 8px', pointerEvents: 'none',
  },
  clipX: { padding: '0 4px', cursor: 'pointer', color: '#11111b', fontSize: 10 },
  freezeMark: {
    position: 'absolute', top: '50%', transform: 'translate(-50%, -50%)',
    color: '#f38ba8', fontSize: 13, cursor: 'pointer', zIndex: 2,
  },
  readout: {
    fontSize: 11, color: '#cdd6f4', background: '#11111b', borderRadius: 6,
    padding: '6px 8px', lineHeight: 1.5, display: 'flex', flexDirection: 'column', gap: 1,
  },
  primary: {
    background: '#89b4fa', color: '#11111b', border: 'none', borderRadius: 6,
    padding: '8px 12px', fontSize: 12.5, fontWeight: 700, cursor: 'pointer',
  },
  statusBar: {
    padding: '7px 16px', borderTop: '1px solid #313244', fontSize: 12,
    color: '#a6adc8', flexShrink: 0,
  },
}
