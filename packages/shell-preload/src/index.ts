/**
 * @de/shell-preload — the contextBridge surface every shell app exposes.
 *
 * Both apps had written the same handful of channels by hand: receive backend
 * messages and raw stdio, send an action, forward a figure event, report a
 * figure resize. This builds that core from the app's id and lets the app spread
 * its own extras alongside.
 *
 * ## Disposers are the whole point of the `on*` shape
 *
 * Every listener registration returns an UNSUBSCRIBE function. The renderer
 * registers these in a `useEffect`, and without cleanup React StrictMode's
 * double-invoke — and every HMR remount — stacks duplicate `ipcRenderer`
 * listeners, so each message is dispatched twice, then three times, and the app
 * degrades as you work. Returning a disposer lets the effect remove the exact
 * listener it added. Do not "simplify" these to bare `ipcRenderer.on`.
 */
import { contextBridge, ipcRenderer, webUtils } from 'electron'

/** An Electron dialog file filter, restated so the renderer need not depend on
 *  electron's types. */
export interface FileFilter { name: string; extensions: string[] }

export interface ShellBridgeOptions {
  /** Matches @de/shell-main's `appId` — channels are `<appId>:<name>`. */
  appId: string
  /** Env var the main process sets from `app.isPackaged`, if the app gates
   *  test-only hooks on it. Defaults to `<APPID>_PACKAGED`. */
  packagedEnvVar?: string
}

/**
 * The channels every shell app shares. Returned rather than exposed, so an app
 * can spread its own on top:
 *
 *     contextBridge.exposeInMainWorld('myapp', {
 *       ...createShellBridge({ appId: 'myapp' }),
 *       myOwnThing: () => ipcRenderer.invoke('myapp:thing'),
 *     })
 */
export function createShellBridge(opts: ShellBridgeOptions) {
  const { appId } = opts
  const packagedVar = opts.packagedEnvVar
    ?? `${appId.toUpperCase().replace(/[^A-Z0-9]/g, '_')}_PACKAGED`
  const channel = (name: string) => `${appId}:${name}`

  /** Register `handler` on `ch`, returning its exact disposer. */
  const on = <A extends unknown[]>(
    ch: string, cb: (...args: A) => void,
  ): (() => void) => {
    const h = (_: unknown, ...args: A) => cb(...args)
    ipcRenderer.on(ch, h as never)
    return () => { ipcRenderer.removeListener(ch, h as never) }
  }

  return {
    /** 'darwin' | 'win32' | 'linux' — hosts lay out the title bar from it. */
    platform: process.platform,

    /** True only in a packaged production app. Dev and the Playwright e2e
     *  (which launches the BUILT bundle by path, not a packaged app) leave the
     *  env var unset, so test-only hooks stay live in both. */
    isPackaged: process.env[packagedVar] === '1',

    /** Any backend message. Returns an unsubscribe fn. */
    onMessage: (cb: (msg: Record<string, unknown>) => void) =>
      on<[Record<string, unknown>]>(channel('message'), cb),

    /** Raw stdout/stderr lines from the backend. Returns an unsubscribe fn. */
    onStream: (cb: (text: string, kind: 'stdout' | 'stderr') => void) =>
      on<[string, 'stdout' | 'stderr']>(channel('stream'), cb),

    /** Send an action to the backend. */
    action: (action: string, payload: Record<string, unknown> = {}, windowId?: number) =>
      ipcRenderer.send(channel('action'), action, payload, windowId),

    /** Forward an interaction event from an anyplotlib iframe to the backend. */
    figureEvent: (figId: string, eventJson: string) =>
      ipcRenderer.send(channel('figure-event'), figId, eventJson),

    /** Tell the backend a figure's container resized, so its layout keeps up. */
    resizeFigure: (figId: string, width: number, height: number) =>
      ipcRenderer.send(channel('resize'), figId, width, height),

    /** Open a URL in the user's browser. Main allowlists the protocol. */
    openExternal: (url: string) => ipcRenderer.send('open-external', url),

    /** OS path of a dropped File. A sandboxed renderer has no `File.path`, so
     *  drag-and-drop of datasets needs this. */
    pathForFile: (file: File): string | null => {
      try {
        return webUtils.getPathForFile(file) || null
      } catch {
        return null
      }
    },

    /** Native open dialog. Resolves to a path, or null if cancelled.
     *
     *  `invoke`, not `send`: the caller needs the answer, and threading a
     *  reply back through the message channel would mean correlating requests
     *  by hand. Main owns the dialog because a sandboxed renderer cannot make
     *  one, and because main is where the parent window is. */
    openFile: (filters?: FileFilter[]): Promise<string | null> =>
      ipcRenderer.invoke(channel('open-file'), filters),

    /** Native save dialog. Resolves to a path, or null if cancelled. */
    saveFile: (filters?: FileFilter[], defaultPath?: string): Promise<string | null> =>
      ipcRenderer.invoke(channel('save-file'), filters, defaultPath),

    /** Escape hatch for an app's own channels, so it does not have to
     *  re-implement the disposer discipline above. */
    onChannel: on,
  }
}

/** Build the core surface and expose it as `window[appId]`, merged with
 *  `extra`. The common case; use `createShellBridge` directly if the app needs
 *  to name the global something else. */
export function exposeShellBridge(
  opts: ShellBridgeOptions, extra: Record<string, unknown> = {},
): void {
  contextBridge.exposeInMainWorld(opts.appId, {
    ...createShellBridge(opts), ...extra,
  })
}
