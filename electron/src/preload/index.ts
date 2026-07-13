/**
 * preload/index.ts — contextBridge API exposed to the renderer.
 */
import { contextBridge, ipcRenderer, webUtils } from 'electron'

contextBridge.exposeInMainWorld('electron', {
  // OS platform ('darwin' | 'win32' | 'linux') — the renderer uses this to lay
  // out the custom title bar (macOS traffic lights on the left vs Windows
  // titleBarOverlay buttons on the right).
  platform: process.platform,

  // True only in an electron-builder–packaged production app. The main process
  // sets SPYDE_PACKAGED=1 from `app.isPackaged` before the window loads. Dev
  // (`npm run dev`) and the Playwright e2e (which launches the BUILT bundle by
  // path, not a packaged app) leave it unset → false. The renderer uses this to
  // gate test-only window hooks OFF in production while keeping them live in dev
  // and e2e.
  isPackaged: process.env.SPYDE_PACKAGED === '1',

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

  /** Open the "Check for Updates" dialog (from the Help menu). Returns an unsubscribe fn. */
  onOpenUpdateDialog: (cb: () => void) => {
    const h = () => cb()
    ipcRenderer.on('spyde:open_update_dialog', h)
    return () => ipcRenderer.removeListener('spyde:open_update_dialog', h)
  },

  /** Open the "GPU Status" dialog (from the Help menu). Returns an unsubscribe fn. */
  onOpenGpuStatusDialog: (cb: () => void) => {
    const h = () => cb()
    ipcRenderer.on('spyde:open_gpu_status_dialog', h)
    return () => ipcRenderer.removeListener('spyde:open_gpu_status_dialog', h)
  },

  /** Open the "GPU & CUDA" help dialog (from the Help menu). Returns an unsubscribe fn. */
  onOpenGpuHelpDialog: (cb: () => void) => {
    const h = () => cb()
    ipcRenderer.on('spyde:open_gpu_help_dialog', h)
    return () => ipcRenderer.removeListener('spyde:open_gpu_help_dialog', h)
  },

  /** electron-updater's check/download/install progress. Returns an unsubscribe fn. */
  onUpdateStatus: (cb: (status: Record<string, unknown>) => void) => {
    const h = (_: unknown, status: Record<string, unknown>) => cb(status)
    ipcRenderer.on('spyde:update-status', h)
    return () => ipcRenderer.removeListener('spyde:update-status', h)
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

  /** Report save dialog (RETURNS the chosen path or null) — for the Report
   *  sidebar's Save/Save As. */
  reportSaveDialog: (defaultName?: string): Promise<string | null> =>
    ipcRenderer.invoke('report:save-dialog', defaultName),

  /** Report open dialog (RETURNS the chosen path or null) — for the Report
   *  sidebar's Open. */
  reportOpenDialog: (): Promise<string | null> => ipcRenderer.invoke('report:open-dialog'),

  /** Report export dialog (RETURNS the chosen path or null) — 'html'/'pdf' are
   *  save dialogs, 'folder' is a directory picker. For the Report sidebar's
   *  Export menu. */
  reportExportDialog: (kind: 'html' | 'pdf' | 'folder' | 'mp4', defaultName?: string): Promise<string | null> =>
    ipcRenderer.invoke('report:export-dialog', kind, defaultName),

  /** Render an exported report HTML file to PDF via a hidden BrowserWindow. */
  reportExportPdf: (htmlPath: string, pdfPath: string): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('report:export-pdf', htmlPath, pdfPath),

  /** Write a PNG data URL to the OS clipboard as an image. */
  clipboardWritePng: (dataUrl: string): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('clipboard:write-png', dataUrl),

  /** OS path of a dropped File (sandboxed renderers have no File.path) —
   *  powers drag-and-drop of datasets (incl. .zspy folders) onto the MDI. */
  pathForFile: (file: File): string | null => {
    try {
      return webUtils.getPathForFile(file) || null
    } catch {
      return null
    }
  },

  /** Forward an interaction event from an anyplotlib iframe to Python. */
  figureEvent: (figId: string, eventJson: string) =>
    ipcRenderer.send('spyde:figure-event', figId, eventJson),

  /** Notify Python of a subwindow resize so figure layout stays in sync. */
  resizeFigure: (figId: string, width: number, height: number) =>
    ipcRenderer.send('spyde:resize', figId, width, height),

  openExternal: (url: string) => ipcRenderer.send('open-external', url),

  // ── Updates / GPU status ──────────────────────────────────────────────────

  /** Current channel, whether this build supports auto-update, last known
   *  status, and the running app's version (for the "About" section). */
  getUpdateInfo: (): Promise<{
    channel: 'stable' | 'beta'
    supported: boolean
    status: Record<string, unknown>
    appVersion: string
  }> => ipcRenderer.invoke('spyde:get-update-info'),

  /** Manual "Check Now". Result arrives via onUpdateStatus. */
  checkForUpdates: () => ipcRenderer.send('spyde:check-for-updates'),

  /** Start downloading a detected update. Progress arrives via onUpdateStatus. */
  downloadUpdate: () => ipcRenderer.send('spyde:download-update'),

  /** Quit and install a downloaded update. */
  quitAndInstallUpdate: () => ipcRenderer.send('spyde:quit-and-install'),

  /** Flip the update channel (stable/beta). */
  setUpdateChannel: (channel: 'stable' | 'beta') => ipcRenderer.send('spyde:set-update-channel', channel),

  /** GPU triage probe (Help → GPU & CUDA): nvidia-smi result + managed-env
   *  facts. torch-side facts come from the backend's get_gpu_status. */
  gpuTriage: (): Promise<{
    nvidia: { name: string; driver: string } | null
    managedEnv: boolean
    envExists: boolean
    lockedTorch: string | null
    busy: boolean
  }> => ipcRenderer.invoke('gpu:triage'),

  /** Re-install torch into the managed env with --torch-backend=auto (the
   *  triage "Fix PyTorch install"). Progress arrives on the raw-output stream;
   *  restart is manual after ok:true. */
  gpuFixTorch: (): Promise<{ ok: boolean; error?: string }> =>
    ipcRenderer.invoke('gpu:fix-torch'),
})
