export {}

declare global {
  interface Window {
    groundcrew: {
      platform: string
      onMessage: (cb: (msg: Record<string, unknown>) => void) => () => void
      action: (action: string, payload?: Record<string, unknown>) => void
      figureEvent: (figId: string, eventJson: string) => void
      resizeFigure: (figId: string, width: number, height: number) => void
      /** Native open dialog. Resolves to a path, or null if cancelled —
       *  cancelling is a normal outcome, not an error. */
      openFile?: (filters?: Array<{ name: string; extensions: string[] }>)
        => Promise<string | null>
      /** Native save dialog. Resolves to a path, or null if cancelled. */
      saveFile?: (filters?: Array<{ name: string; extensions: string[] }>,
                  defaultPath?: string) => Promise<string | null>
    }
  }
}
