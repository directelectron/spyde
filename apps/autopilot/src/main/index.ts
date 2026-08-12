/**
 * index.ts — Electron main process for Autopilot.
 *
 * Short on purpose. Spawning and supervising the Python sidecar, bootstrapping
 * its uv environment, relaying the PLOTAPP protocol and auto-update all come
 * from @de/shell-main; what is left is this app's window, its menu, and the
 * figure scheme.
 *
 * If something here starts looking like it should be shared with SpyDE, it
 * probably should — push it down into the shell rather than copying it.
 */
import { app, BrowserWindow, ipcMain, Menu, nativeTheme } from 'electron'
import { join } from 'path'
import {
  configureShell,
  startBackend, stopBackend, sendAction, sendFigureEvent, sendResize,
  resolvePythonEnv,
} from '@de/shell-main'

configureShell({
  appId: 'autopilot',
  appName: 'Autopilot',
  pythonModule: 'de_autopilot',
  pythonDist: 'de-autopilot',
})

let win: BrowserWindow | null = null
let binaryFrames = 0
let stateUpdates = 0

// NB no custom figure scheme here, unlike SpyDE. Autopilot's viewer builds its
// figure with the ESM fully INLINED, so the renderer can mount it in a `srcdoc`
// iframe and nothing has to be served off disk. SpyDE needs the scheme because
// it swaps the inlined bundle for a shared file:// URL (so Chromium reuses the
// V8 code cache across many figure iframes) — an optimisation that only pays off
// when there are many figures, and one that a `srcdoc` iframe cannot load.
// Should this app ever open figures by the dozen, lift SpyDE's scheme into the
// shell rather than copying it.

// ── Renderer messaging ────────────────────────────────────────────────────────
//
// Backend messages can arrive before the renderer has registered its listener;
// webContents.send() silently drops those. Buffer until the frame says it is
// ready, then flush in order — otherwise the very first figure goes missing.
let rendererReady = false
const pending: Array<Record<string, unknown>> = []

function rendererAlive(): boolean {
  return !!win && !win.isDestroyed() && !win.webContents.isDestroyed()
}

function sendToRenderer(msg: Record<string, unknown>): void {
  if (!rendererReady || !rendererAlive()) { pending.push(msg); return }
  win!.webContents.send('autopilot:message', msg)
}

function flushPending(): void {
  if (!rendererAlive()) return
  while (pending.length) win!.webContents.send('autopilot:message', pending.shift())
}

function createWindow(): BrowserWindow {
  const w = new BrowserWindow({
    width: 1280,
    height: 860,
    backgroundColor: '#14161c',
    show: false,
    webPreferences: {
      preload: join(__dirname, '..', 'preload', 'index.js'),
      sandbox: false,
    },
  })
  w.once('ready-to-show', () => w.show())
  w.webContents.on('did-finish-load', () => { rendererReady = true; flushPending() })
  // Tee renderer AND figure-iframe console output to this terminal, so a JS
  // error inside the figure frame is visible without switching devtools frame
  // context. level: 0=log 1=warning 2=error 3=info.
  w.webContents.on('console-message', (_e, level, message) => {
    if (level >= 1 || message.startsWith('[gc]')) {
      console.log(`[autopilot renderer] ${message}`)
    }
  })
  w.on('closed', () => { win = null; rendererReady = false })

  if (process.env.ELECTRON_RENDERER_URL) {
    w.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    w.loadFile(join(__dirname, '..', 'renderer', 'index.html'))
  }
  return w
}

app.whenReady().then(async () => {
  // The figures theme off prefers-color-scheme; the chrome is dark, so pin it.
  nativeTheme.themeSource = 'dark'

  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(process.platform === 'darwin' ? [{ role: 'appMenu' as const }] : []),
    {
      label: 'Recipe',
      submenu: [
        { label: 'Run', click: () => sendAction('run_recipe') },
        { label: 'Pause', click: () => sendAction('pause_recipe') },
        { label: 'Stop', click: () => sendAction('stop_recipe') },
      ],
    },
    { role: 'viewMenu' },
  ]))

  win = createWindow()

  // __dirname is out/main → two levels up is the app root, which holds the
  // pyproject.toml `uv run` needs.
  const projectRoot = join(__dirname, '..', '..')
  const resolved = await resolvePythonEnv({
    isPackaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
    projectRoot,
    userData: app.getPath('userData'),
  })

  startBackend(resolved.cmd, {
    onMessage: (msg) => {
      // Echo lifecycle messages to this terminal. They otherwise exist only on
      // the PLOTAPP channel, which THIS process consumes — so without the echo
      // backend health is invisible from outside, and an e2e harness has no
      // ready signal to wait on (it watches the app's stdout, not Python's).
      if (msg.type === 'ready' || msg.type === 'error') {
        console.log(`[autopilot backend] ${msg.type}: ${msg.text ?? ''}`)
      }
      if (msg.type === 'state_update' && ++stateUpdates === 1) {
        console.log(`[autopilot backend] first state_update: key=${msg.key}`)
      }
      sendToRenderer(msg)
    },
    onBinary: (header, payload) => {
      // A raw PLOTBIN image frame — this is how every pixel update arrives,
      // because the shell's backendProcess turns on APL_BINARY_TRANSPORT. Omit
      // this handler and the runner parses the frames and drops them on the
      // floor: the figure mounts, the stats strip ticks, and the image stays on
      // its opening placeholder forever, with nothing logged.
      //
      // Shaped like a state_update so the renderer routes it into the figure by
      // fig_id/key, with `buffer` (bytes) in place of `value` (base64).
      // Log the first frame only: enough to tell "pixels never arrived" from
      // "pixels arrived and the renderer dropped them", which are otherwise
      // indistinguishable from a black image pane.
      if (++binaryFrames === 1) {
        console.log(`[autopilot backend] first binary frame: fig=${header.fig_id} ` +
                    `key=${header.key} bytes=${payload.byteLength}`)
      }
      const buf = payload.buffer.slice(
        payload.byteOffset, payload.byteOffset + payload.byteLength)
      sendToRenderer({
        type: 'state_update_binary',
        fig_id: header.fig_id,
        key: header.key,
        header,
        buffer: new Uint8Array(buf),
      })
    },
    onStream: (text, kind) => {
      // Tee to this terminal: the backend's own errors travel the PLOTAPP
      // channel, so without this a dying sidecar dies silently.
      process[kind === 'stderr' ? 'stderr' : 'stdout'].write(text)
      sendToRenderer({ type: 'stream', text, kind })
    },
  }, resolved.cwd)
})

ipcMain.on('autopilot:action', (_e, action: string, payload: Record<string, unknown>) =>
  sendAction(action, payload))
ipcMain.on('autopilot:figure-event', (_e, figId: string, eventJson: string) =>
  sendFigureEvent(figId, eventJson))
ipcMain.on('autopilot:resize', (_e, figId: string, width: number, height: number) =>
  sendResize(figId, width, height))

app.on('window-all-closed', () => {
  stopBackend()
  if (process.platform !== 'darwin') app.quit()
})
app.on('before-quit', () => stopBackend())
