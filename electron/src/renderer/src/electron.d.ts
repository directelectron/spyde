// Type declarations for window.electron (set by preload/index.ts)
declare global {
  interface Window {
    electron: {
      onMessage: (cb: (msg: Record<string, unknown>) => void) => void
      onStream: (cb: (text: string, kind: 'stdout' | 'stderr') => void) => void
      onTile: (cb: () => void) => void
      onOpenDashboard: (cb: () => void) => void
      action: (action: string, payload?: Record<string, unknown>, windowId?: number) => void
      openFile: () => Promise<void>
      saveDialog: () => Promise<void>
      pickFile: (opts: { name?: string; extensions?: string[] }) => Promise<string | null>
      figureEvent: (figId: string, eventJson: string) => void
      resizeFigure: (figId: string, width: number, height: number) => void
      openExternal: (url: string) => void
    }
  }
}

export {}
