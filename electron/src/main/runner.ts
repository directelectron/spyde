/**
 * runner.ts — SpyDE Python process manager.
 *
 * Spawns `uv run python -m spyde` (or a bundled python binary) and maintains
 * the bidirectional PLOTAPP: JSON protocol over stdin/stdout.
 */
import { spawn, ChildProcess } from 'child_process'
import { createInterface } from 'readline'

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

export function stopSpyDE(): void {
  proc?.kill()
  proc = null
}
