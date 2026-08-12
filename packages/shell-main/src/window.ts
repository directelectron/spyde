/**
 * window.ts — the app window, and getting backend messages into it.
 *
 * Both live apps had written this identically: create a BrowserWindow with the
 * preload attached, buffer backend messages until the renderer is listening,
 * flush them in order, and tee renderer console output to the terminal.
 *
 * The buffering is the part that is not obvious. Backend messages can arrive
 * before the renderer has finished loading and registered its `ipcRenderer`
 * listener, and `webContents.send()` DROPS anything sent before then — silently.
 * That swallowed the first message after any quiet period: in SpyDE, the
 * nav-shape prompt on opening a file (the dialog only appeared once a later load
 * pushed more messages); in a live app, the very first figure. So messages queue
 * until `did-finish-load` and are then flushed in order.
 */
import { BrowserWindow } from 'electron'
import { join } from 'path'
import { channel, shellConfig } from './config'

export interface ShellWindowOptions {
  /** Directory of the running main bundle — normally `__dirname`. Preload and
   *  renderer are resolved relative to it (`../preload`, `../renderer`). */
  mainDir: string
  width?: number
  height?: number
  backgroundColor?: string
  /** Extra BrowserWindow options, merged last. */
  browserWindow?: Electron.BrowserWindowConstructorOptions
  /** Tee renderer + figure-iframe console output to this process's stdout.
   *  Warnings and errors always; `logFilter` opts extra lines in. */
  teeConsole?: boolean
  /** Return true to tee a console message that is below warning level. */
  logFilter?: (message: string) => boolean
}

export interface ShellWindow {
  /** The window. Null once it has been closed. */
  get(): BrowserWindow | null
  /** Send a backend message to the renderer, buffering until it is listening. */
  sendToRenderer(msg: Record<string, unknown>): void
  /** True when there is a live window whose webContents is not destroyed. */
  alive(): boolean
}

/**
 * Create the app's window and its message pipe.
 *
 * Loads `ELECTRON_RENDERER_URL` when electron-vite's dev server set it, and the
 * built `../renderer/index.html` otherwise.
 */
export function createShellWindow(opts: ShellWindowOptions): ShellWindow {
  const cfg = shellConfig()
  const messageChannel = channel('message')

  let win: BrowserWindow | null = null
  let rendererReady = false
  const pending: Array<Record<string, unknown>> = []

  const alive = () =>
    !!win && !win.isDestroyed() && !win.webContents.isDestroyed()

  const flush = () => {
    if (!alive()) return
    while (pending.length) win!.webContents.send(messageChannel, pending.shift())
  }

  const sendToRenderer = (msg: Record<string, unknown>) => {
    if (!rendererReady || !alive()) { pending.push(msg); return }
    win!.webContents.send(messageChannel, msg)
  }

  win = new BrowserWindow({
    width: opts.width ?? 1280,
    height: opts.height ?? 860,
    backgroundColor: opts.backgroundColor ?? '#14161c',
    // Shown on ready-to-show rather than immediately, so the user never sees an
    // empty white frame while the renderer boots.
    show: false,
    webPreferences: {
      preload: join(opts.mainDir, '..', 'preload', 'index.js'),
      sandbox: false,
    },
    ...opts.browserWindow,
  })

  win.once('ready-to-show', () => win?.show())
  win.webContents.on('did-finish-load', () => { rendererReady = true; flush() })
  win.on('closed', () => { win = null; rendererReady = false })

  if (opts.teeConsole !== false) {
    // Renderer AND figure-iframe console output, so a JS error inside a figure
    // frame is visible without opening devtools and switching frame context.
    // level: 0=log 1=warning 2=error 3=info.
    win.webContents.on('console-message', (_e, level, message) => {
      if (level >= 1 || opts.logFilter?.(message)) {
        console.log(`[${cfg.appId} renderer] ${message}`)
      }
    })
  }

  if (process.env.ELECTRON_RENDERER_URL) {
    win.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    win.loadFile(join(opts.mainDir, '..', 'renderer', 'index.html'))
  }

  return { get: () => win, sendToRenderer, alive }
}
