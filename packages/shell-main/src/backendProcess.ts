/**
 * backendProcess.ts — the Python sidecar process manager.
 *
 * Spawns the resolved backend command (see pythonEnv.ts) and maintains
 * the bidirectional PLOTAPP: JSON protocol over stdin/stdout.
 */
import { spawn, spawnSync, ChildProcess } from 'child_process'
import process from 'process'
import { shellConfig } from './config'

export interface BackendHandlers {
  onMessage: (msg: Record<string, unknown>) => void
  onStream:  (text: string, kind: 'stdout' | 'stderr') => void
  // A raw PLOTBIN binary frame: the decoded header (fig_id/key/dims/…) plus the
  // raw pixel bytes (NOT base64). Forwarded to the renderer as a transferable
  // ArrayBuffer so large image frames skip the base64/JSON/atob cost.
  onBinary?: (header: Record<string, unknown>, payload: Buffer) => void
}

const PLOTBIN = Buffer.from('PLOTBIN:')
const NL = 0x0a

let proc: ChildProcess | null = null
let tickTimer: ReturnType<typeof setInterval> | null = null

/**
 * The sidecar's last few hundred output lines, for a problem report.
 *
 * The renderer's log panel has the same text, but it lives in a window that a
 * crash may have taken down, and it is not reachable from the main process
 * where a report is assembled. Bounded so a chatty run cannot grow it without
 * limit; long lines are clipped because a report is meant to be read.
 */
const MAX_OUTPUT_LINES = 300
const MAX_OUTPUT_LINE_CHARS = 1000
const backendOutput: string[] = []

function rememberOutput(text: string): void {
  for (const line of text.split('\n')) {
    if (!line.trim()) continue
    backendOutput.push(line.slice(0, MAX_OUTPUT_LINE_CHARS))
  }
  if (backendOutput.length > MAX_OUTPUT_LINES) {
    backendOutput.splice(0, backendOutput.length - MAX_OUTPUT_LINES)
  }
}

/** The sidecar's recent output, oldest first. */
export function recentBackendOutput(): string[] {
  return [...backendOutput]
}

export function startBackend(
  pythonCmd: string[],
  handlers: BackendHandlers,
  cwd?: string,
): void {
  stopping = false  // fresh process — allow a future stopBackend() to run
  const [cmd, ...args] = pythonCmd
  proc = spawn(cmd, args, {
    cwd,   // run from the project root so `uv run` finds the app's pyproject.toml
    // APL_BINARY_TRANSPORT=1: anyplotlib ships large image pixels as raw PLOTBIN
    // binary frames (no base64/JSON) which this runner demuxes — see the stdout
    // parser below. Verified end-to-end (pixel-correct via GPU readback); cuts the
    // ~200 ms/frame base64+JSON+atob transport on a 4k movie. Set to "0" to force
    // the base64 fallback.
    env: {
      ...process.env, PYTHONUNBUFFERED: '1',
      APL_BINARY_TRANSPORT: process.env.APL_BINARY_TRANSPORT ?? '1',
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  })

  // BACKEND TICK (0.5 Hz): Windows throttles timer delivery to the hidden
  // Python child so aggressively that its timer waits (time.sleep,
  // Event.wait, event-loop timers — incl. dask's task-delivery flushes) can
  // freeze INDEFINITELY, waking only when process I/O arrives. Measured
  // end-to-end (SpyDE: spyde/tests/repro_batch_stall.py + _probe_fv_stall.spec.ts):
  // distributed computes sat idle forever hands-off, and EVERY unstick
  // followed a stdin message within ~4 s — a user click "fixing" it was this
  // pipe write, not the click. Electron's own timers are healthy (foreground
  // app), so this interval is reliable; the backend handles 'tick' as a
  // silent no-op. Two lines of traffic per second, bounded staleness ~6 s.
  if (tickTimer) clearInterval(tickTimer)
  tickTimer = setInterval(() => {
    try { sendAction('tick') } catch { /* backend gone — stop ticking */ }
    if (!proc && tickTimer) { clearInterval(tickTimer); tickTimer = null }
  }, 2000)

  // Custom stdout demuxer: the stream interleaves text lines (PLOTAPP: JSON and
  // plain log output, both '\n'-terminated) with raw PLOTBIN binary frames
  // (PLOTBIN:<hlen>:<plen>\n<header_json><payload>). readline can't carry binary,
  // so we parse the raw Buffer stream ourselves, accumulating partial reads.
  let acc: Buffer = Buffer.alloc(0)
  proc.stdout!.on('data', (chunk: Buffer) => {
    acc = acc.length ? Buffer.concat([acc, chunk]) : chunk
    // Process as many complete units as are buffered; stop when we need more.
    for (;;) {
      if (acc.length === 0) break
      // A binary frame if the buffer starts with the PLOTBIN marker.
      if (acc.length >= PLOTBIN.length &&
          acc.subarray(0, PLOTBIN.length).equals(PLOTBIN)) {
        const nl = acc.indexOf(NL)
        if (nl < 0) break                       // prefix line incomplete
        const prefix = acc.subarray(PLOTBIN.length, nl).toString('ascii')
        const [hlenS, plenS] = prefix.split(':')
        const hlen = parseInt(hlenS, 10), plen = parseInt(plenS, 10)
        if (!(hlen >= 0) || !(plen >= 0)) {     // malformed → drop the line
          acc = acc.subarray(nl + 1); continue
        }
        const bodyStart = nl + 1
        const end = bodyStart + hlen + plen
        if (acc.length < end) break             // body not fully arrived yet
        let header: Record<string, unknown> = {}
        try {
          header = JSON.parse(acc.subarray(bodyStart, bodyStart + hlen).toString('utf8'))
        } catch { /* malformed header — still consume the frame */ }
        // Copy the payload out so it survives `acc` being sliced/reused.
        const payload = Buffer.from(acc.subarray(bodyStart + hlen, end))
        acc = acc.subarray(end)
        try { handlers.onBinary?.(header, payload) } catch { /* ignore */ }
        continue
      }
      // Otherwise a text line up to the next '\n'.
      const nl = acc.indexOf(NL)
      if (nl < 0) break                         // line incomplete
      const line = acc.subarray(0, nl).toString('utf8')
      acc = acc.subarray(nl + 1)
      if (line.startsWith('PLOTAPP:')) {
        try {
          handlers.onMessage(JSON.parse(line.slice(8)) as Record<string, unknown>)
        } catch { /* malformed line — ignore */ }
      } else if (line.trim()) {
        rememberOutput(line)
        handlers.onStream(line + '\n', 'stdout')
      }
    }
  })

  proc.stderr!.on('data', (d: Buffer) => {
    rememberOutput(d.toString())
    handlers.onStream(d.toString(), 'stderr')
  })

  proc.on('close', (code) => {
    proc = null
    rememberOutput(`[exited with code ${code}]`)
    handlers.onStream(`[${shellConfig().appName} exited with code ${code}]\n`, 'stderr')
    // Surface the death to the renderer so the UI doesn't silently freeze —
    // every sendAction() after this no-ops (proc is null), so without this the
    // user gets no indication the analysis backend stopped. Routed through the
    // same onMessage path as a synthetic message (not a PLOTAPP: line).
    handlers.onMessage({ type: 'backend_exited', code })
  })
}

/** Send a JSON action message to the Python backend. */
export function sendAction(
  action: string,
  payload: Record<string, unknown> = {},
  windowId?: number,
): void {
  if (!proc?.stdin) return
  const msg: Record<string, unknown> = { type: 'action', action, payload }
  if (windowId !== undefined) msg.window_id = windowId
  proc.stdin.write(JSON.stringify(msg) + '\n')
}

/** Forward a figure interaction event back to Python. */
export function sendFigureEvent(figId: string, eventJson: string): void {
  if (!proc?.stdin) return
  proc.stdin.write(JSON.stringify({ type: 'figure_event', fig_id: figId, event_json: eventJson }) + '\n')
}

/** Notify Python that a figure's container resized. */
export function sendResize(figId: string, width: number, height: number): void {
  if (!proc?.stdin) return
  proc.stdin.write(JSON.stringify({ type: 'resize', fig_id: figId, width, height }) + '\n')
}

/**
 * Stop the Python backend, leaving NO orphaned worker subprocesses.
 *
 * Strategy:
 *  1. GRACEFUL: write `{type:'quit'}` to stdin. The backend's asyncio loop
 *     (app.py) handles this by breaking and calling `session.shutdown()`, which
 *     tears down the Dask cluster cleanly.
 *  2. BACKSTOP TREE-KILL: the backend may not exit promptly (mid-compute) or
 *     stdin may already be closed, and `proc.kill()` on Windows only kills the
 *     DIRECT child — leaving the Dask worker/nanny GRANDCHILDREN orphaned. So
 *     after a short grace period we kill the whole tree:
 *       - win32: `taskkill /pid <pid> /T /F` (whole tree, force).
 *       - posix: SIGTERM, then SIGKILL after a short timer.
 *
 * This composes with the PYTHON-side process_guard.py: that installs a Windows
 * kill-on-close Job Object so the OS reaps the worker tree whenever the backend
 * process itself dies for ANY reason (clean exit, crash, or our taskkill). The
 * graceful quit here is the preferred path (clean cluster shutdown); the
 * tree-kill is the backstop for when Electron must hard-stop the backend before
 * it can reach its own shutdown(). Both ultimately guarantee no leaked workers.
 *
 * Idempotent and null-safe: callable from window-all-closed, before-quit, and a
 * signal handler without double-killing.
 *
 * `immediate` skips the grace period and tree-kills BEFORE returning. Step 2's
 * timers only fire while this process is still alive, so a caller that is about
 * to end the process (the update handoff — see updater.ts) would otherwise leave
 * the sidecar and its Dask workers running: the graceful `quit` is written, the
 * timer is armed, and Electron exits before it can fire. The Windows installer
 * that starts moments later then finds processes still holding the install
 * directory and refuses to continue.
 */
let stopping = false
export function stopBackend(options: { immediate?: boolean } = {}): void {
  const p = proc
  if (!p || stopping) {
    proc = null
    return
  }
  stopping = true
  proc = null  // every sendAction() after this no-ops; prevents re-entrant kills
  // KNOWN BUG (open): quitting while a find-vectors batch is still streaming can
  // WEDGE shutdown on Windows. Clearing the tick timer here stops the stdin tick
  // that keeps the hidden backend scheduled (the very starvation the tick was
  // added for), so a mid-batch backend may never get scheduled long enough to
  // process the graceful `quit` — only the 1.5 s taskkill backstop ends it. The
  // e2e specs work around it by waiting for the '[fv-batch] finalized' log line
  // before closing the app. The app-side fix (keep ticking until the backend
  // exits, or force-reap the batch) is still open.
  if (tickTimer) { clearInterval(tickTimer); tickTimer = null }

  // 1. Ask the backend to quit gracefully (clean Dask shutdown).
  try {
    if (p.stdin && p.stdin.writable) {
      p.stdin.write(JSON.stringify({ type: 'quit' }) + '\n')
    }
  } catch { /* stdin may already be torn down — fall through to tree-kill */ }

  const pid = p.pid

  // 2. Backstop: if it hasn't exited shortly, kill the whole process tree so no
  //    Dask worker/nanny grandchildren are left behind.
  if (options.immediate) {
    killTreeNow(p, pid)
    return
  }
  if (process.platform === 'win32') {
    if (pid !== undefined) {
      setTimeout(() => {
        if (p.exitCode !== null || p.signalCode !== null) return  // already gone
        try {
          spawn('taskkill', ['/pid', String(pid), '/T', '/F'], { stdio: 'ignore' })
        } catch { try { p.kill() } catch { /* nothing else to do */ } }
      }, 1500)
    } else {
      try { p.kill() } catch { /* */ }
    }
  } else {
    setTimeout(() => {
      if (p.exitCode !== null || p.signalCode !== null) return
      try { p.kill('SIGTERM') } catch { /* */ }
      setTimeout(() => {
        if (p.exitCode !== null || p.signalCode !== null) return
        try { p.kill('SIGKILL') } catch { /* */ }
      }, 1500)
    }, 1500)
  }
}

/**
 * Kill the sidecar's whole process tree and return once the request has been
 * made — no timers, so it still runs when the caller is about to end this
 * process. `spawnSync` is the point: a detached `spawn` would only be queued.
 */
function killTreeNow(p: ChildProcess, pid: number | undefined): void {
  if (p.exitCode !== null || p.signalCode !== null) return
  if (process.platform === 'win32') {
    if (pid !== undefined) {
      try {
        spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { stdio: 'ignore' })
        return
      } catch { /* fall through to the direct kill */ }
    }
    try { p.kill() } catch { /* nothing else to do */ }
    return
  }
  try { p.kill('SIGKILL') } catch { /* nothing else to do */ }
  // The workers are grandchildren, so also target the child's process group
  // when it leads one. A pid that leads no group simply has no group to match.
  if (pid !== undefined) {
    try { process.kill(-pid, 'SIGKILL') } catch { /* no such group */ }
  }
}
