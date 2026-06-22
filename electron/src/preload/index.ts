/**
 * preload/index.ts — contextBridge API exposed to the renderer.
 */
import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electron', {
  // ── Python → Renderer ─────────────────────────────────────────────────────

  /** Listen for any PLOTAPP: message from Python (persistent listener). */
  onMessage: (cb: (msg: Record<string, unknown>) => void) =>
    ipcRenderer.on('spyde:message', (_, msg) => cb(msg)),

  /** Listen for raw stdout/stderr lines from Python. */
  onStream: (cb: (text: string, kind: 'stdout' | 'stderr') => void) =>
    ipcRenderer.on('spyde:stream', (_, text, kind) => cb(text, kind)),

  /** Window tile command from menu. */
  onTile: (cb: () => void) => ipcRenderer.on('spyde:tile', () => cb()),

  /** Open Dask dashboard command from menu. */
  onOpenDashboard: (cb: () => void) =>
    ipcRenderer.on('spyde:open_dashboard', () => cb()),

  /** Launch a guided tour by id (from the Help menu). */
  onStartGuide: (cb: (id: string) => void) =>
    ipcRenderer.on('spyde:start_guide', (_, id: string) => cb(id)),

  // ── Renderer → Python ─────────────────────────────────────────────────────

  /** Send a toolbar/menu action to Python. */
  action: (
    action: string,
    payload: Record<string, unknown> = {},
    windowId?: number,
  ) => ipcRenderer.send('spyde:action', action, payload, windowId),

  /** Open a native file picker (result sent directly to Python). */
  openFile: (): Promise<void> => ipcRenderer.invoke('spyde:open-file'),

  /** Open a native save dialog. */
  saveDialog: (): Promise<void> => ipcRenderer.invoke('spyde:save-dialog'),

  /** Pick a file and return its path (for action params, e.g. a .cif). */
  pickFile: (opts: { name?: string; extensions?: string[] }): Promise<string | null> =>
    ipcRenderer.invoke('spyde:pick-file', opts),

  /** Forward an interaction event from an anyplotlib iframe to Python. */
  figureEvent: (figId: string, eventJson: string) =>
    ipcRenderer.send('spyde:figure-event', figId, eventJson),

  /** Notify Python of a subwindow resize so figure layout stays in sync. */
  resizeFigure: (figId: string, width: number, height: number) =>
    ipcRenderer.send('spyde:resize', figId, width, height),

  openExternal: (url: string) => ipcRenderer.send('open-external', url),
})
