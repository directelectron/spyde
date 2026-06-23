/**
 * preload/index.ts — contextBridge API exposed to the renderer.
 */
import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electron', {
  // OS platform ('darwin' | 'win32' | 'linux') — the renderer uses this to lay
  // out the custom title bar (macOS traffic lights on the left vs Windows
  // titleBarOverlay buttons on the right).
  platform: process.platform,

  // ── Python → Renderer ─────────────────────────────────────────────────────

  // Each on* returns an UNSUBSCRIBE function. The renderer registers these in a
  // useEffect; without cleanup, React StrictMode's double-invoke (and any HMR /
  // re-mount) would stack duplicate ipcRenderer listeners, so every message gets
  // dispatched 2×, 3×, … and the app's logs/updates appear doubled and degrade
  // over time. Returning a disposer lets the effect remove the exact listener.

  /** Listen for any PLOTAPP: message from Python. Returns an unsubscribe fn. */
  onMessage: (cb: (msg: Record<string, unknown>) => void) => {
    const h = (_: unknown, msg: Record<string, unknown>) => cb(msg)
    ipcRenderer.on('spyde:message', h)
    return () => ipcRenderer.removeListener('spyde:message', h)
  },

  /** Listen for raw stdout/stderr lines from Python. Returns an unsubscribe fn. */
  onStream: (cb: (text: string, kind: 'stdout' | 'stderr') => void) => {
    const h = (_: unknown, text: string, kind: 'stdout' | 'stderr') => cb(text, kind)
    ipcRenderer.on('spyde:stream', h)
    return () => ipcRenderer.removeListener('spyde:stream', h)
  },

  /** Window tile command from menu. Returns an unsubscribe fn. */
  onTile: (cb: () => void) => {
    const h = () => cb()
    ipcRenderer.on('spyde:tile', h)
    return () => ipcRenderer.removeListener('spyde:tile', h)
  },

  /** Open Dask dashboard command from menu. Returns an unsubscribe fn. */
  onOpenDashboard: (cb: () => void) => {
    const h = () => cb()
    ipcRenderer.on('spyde:open_dashboard', h)
    return () => ipcRenderer.removeListener('spyde:open_dashboard', h)
  },

  /** Launch a guided tour by id (from the Help menu). Returns an unsubscribe fn. */
  onStartGuide: (cb: (id: string) => void) => {
    const h = (_: unknown, id: string) => cb(id)
    ipcRenderer.on('spyde:start_guide', h)
    return () => ipcRenderer.removeListener('spyde:start_guide', h)
  },

  /** Open the in-app Load Stack dialog (from the File menu). Returns an unsubscribe fn. */
  onOpenStackDialog: (cb: () => void) => {
    const h = () => cb()
    ipcRenderer.on('spyde:open_stack_dialog', h)
    return () => ipcRenderer.removeListener('spyde:open_stack_dialog', h)
  },

  // ── Renderer → Python ─────────────────────────────────────────────────────

  /** Send a toolbar/menu action to Python. */
  action: (
    action: string,
    payload: Record<string, unknown> = {},
    windowId?: number,
  ) => ipcRenderer.send('spyde:action', action, payload, windowId),

  /** Open a native file picker (result sent directly to Python). */
  openFile: (): Promise<void> => ipcRenderer.invoke('spyde:open-file'),

  /** Open a .zspy/.zarr DIRECTORY store (folder picker → load). */
  openZarrFolder: (): Promise<void> => ipcRenderer.invoke('spyde:open-zarr-folder'),

  /** Quit the app (custom title-bar menu replaces native File→Quit). */
  quit: (): Promise<void> => ipcRenderer.invoke('spyde:quit'),

  /** Open a native save dialog. */
  saveDialog: (): Promise<void> => ipcRenderer.invoke('spyde:save-dialog'),

  /** Pick a file and return its path (for action params, e.g. a .cif). */
  pickFile: (opts: { name?: string; extensions?: string[] }): Promise<string | null> =>
    ipcRenderer.invoke('spyde:pick-file', opts),

  /** Multi-select picker that RETURNS the chosen paths (for the Load Stack dialog). */
  pickFiles: (opts?: { name?: string; extensions?: string[] }): Promise<string[]> =>
    ipcRenderer.invoke('spyde:pick-files', opts),

  /** Multi-select DIRECTORY picker (RETURNS paths) — for .zspy/.zarr folders. */
  pickFolders: (): Promise<string[]> => ipcRenderer.invoke('spyde:pick-folders'),

  /** Forward an interaction event from an anyplotlib iframe to Python. */
  figureEvent: (figId: string, eventJson: string) =>
    ipcRenderer.send('spyde:figure-event', figId, eventJson),

  /** Notify Python of a subwindow resize so figure layout stays in sync. */
  resizeFigure: (figId: string, width: number, height: number) =>
    ipcRenderer.send('spyde:resize', figId, width, height),

  openExternal: (url: string) => ipcRenderer.send('open-external', url),
})
