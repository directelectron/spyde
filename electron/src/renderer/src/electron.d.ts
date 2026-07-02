// Type declarations for window.electron (set by preload/index.ts)
declare global {
  interface Window {
    electron: {
      platform: string
      isPackaged: boolean
      onMessage: (cb: (msg: Record<string, unknown>) => void) => () => void
      onStream: (cb: (text: string, kind: 'stdout' | 'stderr') => void) => () => void
      onTile: (cb: () => void) => () => void
      onOpenDashboard: (cb: () => void) => () => void
      onStartGuide: (cb: (id: string) => void) => () => void
      onOpenStackDialog: (cb: () => void) => () => void
      onOpenUpdateDialog: (cb: () => void) => () => void
      onOpenGpuStatusDialog: (cb: () => void) => () => void
      onUpdateStatus: (cb: (status: Record<string, unknown>) => void) => () => void
      action: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
      openFile: () => Promise<void>
      openZarrFolder: () => Promise<void>
      quit: () => Promise<void>
      saveDialog: () => Promise<void>
      pickFile: (opts: { name?: string; extensions?: string[] }) => Promise<string | null>
      pickFiles: (opts?: { name?: string; extensions?: string[] }) => Promise<string[]>
      pickFolders: () => Promise<string[]>
      pathForFile?: (file: File) => string | null
      figureEvent: (figId: string, eventJson: string) => void
      resizeFigure: (figId: string, width: number, height: number) => void
      openExternal: (url: string) => void
      getUpdateInfo: () => Promise<{
        channel: 'stable' | 'beta'
        supported: boolean
        status: Record<string, unknown>
        appVersion: string
      }>
      checkForUpdates: () => void
      downloadUpdate: () => void
      quitAndInstallUpdate: () => void
      setUpdateChannel: (channel: 'stable' | 'beta') => void
    }

    // Test-only hooks attached by the renderer for Playwright e2e (DEV /
    // non-packaged builds only — never in `npm run dist`). See SpyDEContext.tsx.
    _spyde_test_inject?: (msg: Record<string, unknown>) => void
    _spyde_test_widgets?: (
      figId: string,
    ) => Array<{ panel_id: string; id: string; type: string; data: Record<string, unknown> }>
    _spyde_test_image_sig?: (figId: string) => string
  }
}

export {}
