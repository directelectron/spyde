/**
 * runner.ts — SpyDE Python process manager.
 *
 * Spawns `uv run python -m spyde` (or a bundled python binary) and maintains
 * the bidirectional PLOTAPP: JSON protocol over stdin/stdout.
 */
import { spawn, ChildProcess } from 'child_process'
import { createInterface } from 'readline'
import process from 'process'

export interface SpyDEHandlers {
  onMessage: (msg: Record<string, unknown>) => void
  onStream:  (text: string, kind: 'stdout' | 'stderr') => void
}

let proc: ChildProcess | null = null

export function startSpyDE(
  pythonCmd: string[],
  handlers: SpyDEHandlers,
  cwd?: string,
): void {
  stopping = false  // fresh process — allow a future stopSpyDE() to run
  const [cmd, ...args] = pythonCmd
  proc = spawn(cmd, args, {
    cwd,   // run from the repo root so `uv run` finds spyde's pyproject.toml
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['pipe', 'pipe', 'pipe'],
  })

  const rl = createInterface({ input: proc.stdout! })
  rl.on('line', (line) => {
    if (line.startsWith('PLOTAPP:')) {
      try {
        const msg = JSON.parse(line.slice(8)) as Record<string, unknown>
        handlers.onMessage(msg)
      } catch { /* malformed line — ignore */ }
    } else if (line.trim()) {
      handlers.onStream(line + '\n', 'stdout')
    }
  })

  proc.stderr!.on('data', (d: Buffer) =>
    handlers.onStream(d.toString(), 'stderr')
  )

  proc.on('close', (code) => {
    proc = null
    handlers.onStream(`[SpyDE exited with code ${code}]\n`, 'stderr')
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

/** Notify Python of an MDI subwindow resize. */
export function sendResize(figId: string, width: number, height: number): void {
  if (!proc?.stdin) return
  proc.stdin.write(JSON.stringify({ type: 'resize', fig_id: figId, width, height }) + '\n')
}

/**
 * Stop the Python backend, leaving NO orphaned Dask worker subprocesses.
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
 */
let stopping = false
export function stopSpyDE(): void {
  const p = proc
  if (!p || stopping) {
    proc = null
    return
  }
  stopping = true
  proc = null  // every sendAction() after this no-ops; prevents re-entrant kills

  // 1. Ask the backend to quit gracefully (clean Dask shutdown).
  try {
    if (p.stdin && p.stdin.writable) {
      p.stdin.write(JSON.stringify({ type: 'quit' }) + '\n')
    }
  } catch { /* stdin may already be torn down — fall through to tree-kill */ }

  const pid = p.pid

  // 2. Backstop: if it hasn't exited shortly, kill the whole process tree so no
  //    Dask worker/nanny grandchildren are left behind.
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
