/**
 * preload/index.ts — Autopilot's contextBridge surface.
 *
 * Entirely the shell's core (see @de/shell-preload): backend messages, raw
 * stdio, actions, figure events, resize. The app adds nothing of its own yet —
 * every control it has goes through `action()`.
 */
import { exposeShellBridge } from '@de/shell-preload'

exposeShellBridge({ appId: 'autopilot' })
