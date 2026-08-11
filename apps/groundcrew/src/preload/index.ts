/**
 * preload/index.ts — the contextBridge surface for Ground Crew.
 *
 * Deliberately small. Every `on*` returns an UNSUBSCRIBE function: the renderer
 * registers them in a useEffect, and without cleanup React StrictMode's
 * double-invoke (and every HMR remount) stacks duplicate ipcRenderer listeners,
 * so each message gets dispatched twice, then three times, and the app degrades
 * as you work. Returning a disposer lets the effect remove the exact listener.
 *
 * This is the second copy of that shape — SpyDE's preload has the same one.
 * When @de/shell-preload lands, both collapse into it.
 */
import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('groundcrew', {
  platform: process.platform,

  /** Any message from the Python backend. Returns an unsubscribe fn. */
  onMessage: (cb: (msg: Record<string, unknown>) => void) => {
    const h = (_: unknown, msg: Record<string, unknown>) => cb(msg)
    ipcRenderer.on('groundcrew:message', h)
    return () => ipcRenderer.removeListener('groundcrew:message', h)
  },

  /** Send an action to the backend. */
  action: (action: string, payload: Record<string, unknown> = {}) =>
    ipcRenderer.send('groundcrew:action', action, payload),

  /** Forward an interaction event from an anyplotlib iframe to Python. */
  figureEvent: (figId: string, eventJson: string) =>
    ipcRenderer.send('groundcrew:figure-event', figId, eventJson),

  /** Tell Python the figure's container resized, so its layout keeps up. */
  resizeFigure: (figId: string, width: number, height: number) =>
    ipcRenderer.send('groundcrew:resize', figId, width, height),
})
