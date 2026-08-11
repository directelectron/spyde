export {}

declare global {
  interface Window {
    groundcrew: {
      platform: string
      onMessage: (cb: (msg: Record<string, unknown>) => void) => () => void
      action: (action: string, payload?: Record<string, unknown>) => void
      figureEvent: (figId: string, eventJson: string) => void
      resizeFigure: (figId: string, width: number, height: number) => void
    }
  }
}
