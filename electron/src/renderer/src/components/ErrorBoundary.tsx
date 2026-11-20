/**
 * ErrorBoundary.tsx — keep a render error where it happened, and say so.
 *
 * React unmounts the WHOLE tree when a render throws and nothing catches it,
 * which left the app as an empty window in the page background colour: no
 * error, no windows, while the Python backend carried on. A boundary turns
 * that into a fallback in place of the part that failed — one window, its
 * toolbar, the dock — and reports the error to the main process, which keeps
 * it for a problem report and in lifecycle.log, so the next one names the
 * component that threw.
 *
 * "Try again" re-renders the failed part. Whatever it displays lives in
 * SpyDEContext, above every boundary but the root one, so nothing is lost.
 */
import React from 'react'

interface Props {
  /** What failed, for the report: 'app', 'window 3', 'toolbar 3', 'dock'. */
  scope: string
  /** What to show instead; `retry` re-renders the children. */
  fallback: (error: Error, retry: () => void) => React.ReactNode
  children: React.ReactNode
}

interface State { error: Error | null }

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // console.error reaches the main process's stderr as RENDERER-ERROR.
    console.error(`[spyde] ${this.props.scope} failed to render:`, error, info.componentStack)
    try {
      window.electron?.reportRendererError?.({
        scope: this.props.scope,
        message: String(error?.message ?? error),
        stack: String(error?.stack ?? ''),
        componentStack: String(info.componentStack ?? ''),
      })
    } catch { /* reporting must never throw out of the boundary */ }
  }

  retry = (): void => this.setState({ error: null })

  render(): React.ReactNode {
    return this.state.error ? this.props.fallback(this.state.error, this.retry) : this.props.children
  }
}

const S: Record<string, React.CSSProperties> = {
  inline: {
    height: '100%', boxSizing: 'border-box', padding: 12, display: 'flex',
    flexDirection: 'column', gap: 8, color: '#cdd6f4', fontSize: 12, overflow: 'auto',
  },
  title: { color: '#f38ba8', fontWeight: 600 },
  message: { fontFamily: 'ui-monospace, monospace', fontSize: 11, color: '#a6adc8', whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
  button: {
    alignSelf: 'flex-start', background: '#313244', color: '#cdd6f4', border: '1px solid #45475a',
    borderRadius: 4, padding: '4px 10px', fontSize: 11, cursor: 'pointer',
  },
  screen: {
    position: 'fixed', inset: 0, background: '#11111b', color: '#cdd6f4', display: 'flex',
    alignItems: 'center', justifyContent: 'center', padding: 24, fontSize: 13,
  },
  card: {
    maxWidth: 640, display: 'flex', flexDirection: 'column', gap: 10, background: '#1e1e2e',
    border: '1px solid #313244', borderRadius: 10, padding: 20,
  },
}

/** The error as a person can read it. A production build replaces React's
 *  own messages with a numbered pointer to its website ("Minified React error
 *  #31; visit https://… for the full message …"); the number is kept, the
 *  rest is noise in a dialog. The full text still goes to the report. */
function readable(error: Error): string {
  const text = String(error?.message ?? error)
  return text.replace(/; visit https:\/\/\S+ for the full message[\s\S]*$/, '')
}

/** Fallback for one part of the app — a window's figure, a toolbar, the dock. */
export function PartFailed({ what, error, retry }: {
  what: string; error: Error; retry: () => void
}): React.JSX.Element {
  return (
    <div data-testid="render-error" style={S.inline}>
      <span style={S.title}>{what} could not be drawn</span>
      <span style={S.message}>{readable(error)}</span>
      <span>The rest of SpyDE is still running. The error is saved for Help → Report a Problem.</span>
      <button data-testid="render-error-retry" style={S.button} onClick={retry}>Try again</button>
    </div>
  )
}

/** Fallback for the whole interface: the last resort, above every other boundary. */
export function AppFailed({ error, retry }: { error: Error; retry: () => void }): React.JSX.Element {
  return (
    <div data-testid="app-render-error" style={S.screen}>
      <div style={S.card}>
        <span style={S.title}>SpyDE's interface stopped on an error</span>
        <span style={S.message}>{readable(error)}</span>
        <span>
          Your data is still loaded in the analysis backend. Reloading the interface
          closes its windows; reopen the files to see them again. The error is saved
          for Help → Report a Problem.
        </span>
        <div style={{ display: 'flex', gap: 8 }}>
          <button data-testid="app-render-error-retry" style={S.button} onClick={retry}>Try again</button>
          <button data-testid="app-render-error-reload" style={S.button}
            onClick={() => window.location.reload()}>Reload interface</button>
        </div>
      </div>
    </div>
  )
}
