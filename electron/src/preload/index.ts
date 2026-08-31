/**
 * preload/index.ts — SpyDE's contextBridge API.
 *
 * The core surface (platform / isPackaged / onMessage / onStream / action /
 * figureEvent / resizeFigure / openExternal / pathForFile) comes from
 * @de/shell-preload, shared with de-groundcrew. What is spread on below is
 * SpyDE's own: the file and report dialogs, PDF export, clipboard, the update
 * controls and GPU triage, plus the menu-driven `on*` events.
 *
 * Every `on*` returns an UNSUBSCRIBE function — see the note in
 * @de/shell-preload for why that is load-bearing rather than tidy.
 */
import { contextBridge, ipcRenderer } from 'electron'
import { createShellBridge } from '@de/shell-preload'

const shell = createShellBridge({ appId: 'spyde', packagedEnvVar: 'SPYDE_PACKAGED' })

/** Register a menu-driven event, returning its exact disposer. */
const onEvent = shell.onChannel

contextBridge.exposeInMainWorld('electron', {
  ...shell,
  /** Window tile command from menu. Returns an unsubscribe fn. */
  onTile: (cb: () => void) => onEvent('spyde:tile', cb),

  /** Open Dask dashboard command from menu. Returns an unsubscribe fn. */
  onOpenDashboard: (cb: () => void) => onEvent('spyde:open_dashboard', cb),

  /** Launch a guided tour by id (from the Help menu). Returns an unsubscribe fn. */
  onStartGuide: (cb: (v: string) => void) => onEvent<[string]>('spyde:start_guide', cb),

  /** Open the in-app Load Stack dialog (from the File menu). Returns an unsubscribe fn. */
  onOpenStackDialog: (cb: () => void) => onEvent('spyde:open_stack_dialog', cb),

  /** Open the "Check for Updates" dialog (from the Help menu). Returns an unsubscribe fn. */
  onOpenUpdateDialog: (cb: () => void) => onEvent('spyde:open_update_dialog', cb),

  /** Open the "GPU Status" dialog (from the Help menu). Returns an unsubscribe fn. */
  onOpenGpuStatusDialog: (cb: () => void) => onEvent('spyde:open_gpu_status_dialog', cb),

  /** Open the "GPU & CUDA" help dialog (from the Help menu). Returns an unsubscribe fn. */
  onOpenGpuHelpDialog: (cb: () => void) => onEvent('spyde:open_gpu_help_dialog', cb),

  /** Open the "Report a Problem" dialog (from the Help menu). Returns an unsubscribe fn. */
  onOpenReportDialog: (cb: () => void) => onEvent('spyde:open_report_dialog', cb),

  /** electron-updater's check/download/install progress. Returns an unsubscribe fn. */
  onUpdateStatus: (cb: (v: Record<string, unknown>) => void) => onEvent<[Record<string, unknown>]>('spyde:update-status', cb),

  // ── Renderer → Python (SpyDE's own; the core action/figure/stream
  //    channels come from the shell spread above) ──────────────────────────

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

  /** Reveal a local DIRECTORY in the OS file manager (Examples → Show Example
   *  Data Directory). Separate from openExternal, which allowlists web/mail
   *  protocols so it can never open a local path; main verifies the target is
   *  a real directory. */
  openPath: (path: string) => ipcRenderer.send('open-path', path),

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

  // ── Problem reporting ─────────────────────────────────────────────────────

  /** What the Report a Problem dialog shows before anything is written: whether
   *  a report can be sent at all, and the machine facts it would include.
   *  Reading this sends nothing. */
  reportDiagnostics: (): Promise<{
    canSend: boolean
    diagnostics: Record<string, unknown>
  }> => ipcRenderer.invoke('spyde:report-diagnostics'),

  /** Send the report. Always saves a copy locally; `sent` says whether it also
   *  reached the maintainers. */
  submitReport: (input: { message: string; contact?: string }): Promise<{
    sent: boolean
    eventId?: string
    bundlePath?: string
    error?: string
  }> => ipcRenderer.invoke('spyde:submit-report', input),

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
