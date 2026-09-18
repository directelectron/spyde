/**
 * ScreenRecorderGate.tsx — Help → Record Screen: capture the SpyDE window to a
 * video file (a demo, a bug report, a scrub someone else needs to see).
 *
 * What is captured is THIS window's own web contents, not a display: main
 * answers the request with the asking frame (setDisplayMediaRequestHandler), so
 * no source picker appears, nothing outside SpyDE can end up in the file, and
 * macOS never asks for Screen Recording permission. The trade-off is that
 * anything the OS draws on top — the native menu bar, a file dialog — is not in
 * the recording.
 *
 * It has to start from a real click: Chromium requires user activation for
 * getDisplayMedia, which a menu item routed through the main process does not
 * carry. Hence the entry point is the in-app Help menu, which dispatches
 * `spyde:toggle_record`.
 */
import React from 'react'

/** mp4 where Chromium can mux it, webm everywhere else. */
const FORMATS = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm']

/** Long enough that the per-chunk IPC is negligible, short enough that a crash
 *  costs seconds of video rather than the whole clip. */
const CHUNK_MS = 3000

const clock = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

export function ScreenRecorderGate() {
  const recorder = React.useRef<MediaRecorder | null>(null)
  const [seconds, setSeconds] = React.useState<number | null>(null)
  const [note, setNote] = React.useState<string | null>(null)

  const stop = () => {
    recorder.current?.stop()
    recorder.current = null
    setSeconds(null)
  }

  const start = async () => {
    setNote(null)
    const mimeType = FORMATS.find((t) => MediaRecorder.isTypeSupported(t)) ?? ''
    const path = await window.electron.startRecording(mimeType.startsWith('video/mp4') ? 'mp4' : 'webm')
    if (!path) return
    const stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 } })
    const rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 })
    // Chained rather than fired-and-forgotten: chunks must reach the file in
    // recording order, and awaiting each write is the backpressure too.
    let written: Promise<void> = Promise.resolve()
    rec.ondataavailable = (e) => {
      if (!e.data.size) return
      written = written.then(async () =>
        window.electron.recordChunk(new Uint8Array(await e.data.arrayBuffer())))
    }
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop())
      await written
      setNote(`Saved ${path.split(/[\\/]/).pop()}`)
    }
    rec.start(CHUNK_MS)
    recorder.current = rec
    setSeconds(0)
  }

  React.useEffect(() => {
    const toggle = () => {
      if (recorder.current) stop()
      else start().catch((e) => setNote(`Recording failed — ${(e as Error)?.message ?? e}`))
    }
    window.addEventListener('spyde:toggle_record', toggle)
    return () => window.removeEventListener('spyde:toggle_record', toggle)
  }, [])

  // One timeout per tick rather than an interval, so the elapsed clock can
  // never outlive the recording it is counting.
  React.useEffect(() => {
    if (seconds === null) return
    const id = setTimeout(() => setSeconds(seconds + 1), 1000)
    return () => clearTimeout(id)
  }, [seconds])

  React.useEffect(() => {
    if (!note) return
    const id = setTimeout(() => setNote(null), 6000)
    return () => clearTimeout(id)
  }, [note])

  if (seconds === null && !note) return null
  return (
    <button onClick={stop} style={styles.pill} data-testid="screen-recorder-pill">
      {seconds === null ? note : `● REC ${clock(seconds)} — stop`}
    </button>
  )
}

const styles: Record<string, React.CSSProperties> = {
  // Floats over the EMPTY middle of the title bar: both bottom corners already
  // belong to cards (downloads, updates) and the console bar, and a capture
  // indicator has to stay out of the way of the thing being demonstrated.
  // no-drag so the title bar's drag region doesn't swallow the click. Sits one
  // BELOW the tour overlay (Tour.tsx, 9000): the tour's callout is the only
  // thing on screen that must never be covered, and since that overlay passes
  // pointer events through, the pill stays clickable under it.
  pill: {
    position: 'fixed', top: 8, left: '50%', transform: 'translateX(-50%)', zIndex: 8999,
    background: '#181825', color: '#f38ba8', border: '1px solid #f38ba8',
    borderRadius: 999, padding: '2px 10px', font: '11px system-ui', cursor: 'pointer',
    WebkitAppRegion: 'no-drag',
  } as React.CSSProperties,
}
