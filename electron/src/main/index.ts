/**
 * index.ts — Electron main process for SpyDE.
 */
import { app, BrowserWindow, dialog, ipcMain, Menu, shell, nativeTheme, net, protocol } from 'electron'
import { join, basename } from 'path'
import { pathToFileURL } from 'url'
import { tmpdir } from 'os'
import { existsSync, realpathSync, writeFileSync, rmSync } from 'fs'
import {
  startSpyDE, sendAction, sendFigureEvent, sendResize,
  stopSpyDE,
} from './runner'
import { resolvePythonEnv } from './pythonEnv'
import {
  initUpdater, checkForUpdates, downloadUpdate, quitAndInstall,
  readUpdateChannel, setUpdateChannel, getLastUpdateStatus, updatesSupported,
} from './updater'

let win: BrowserWindow | null = null

// ── Figure protocol ───────────────────────────────────────────────────────────
//
// Figures are anyplotlib-generated HTML written to the OS temp dir and shown in
// iframes. They used to load via raw `file://` URLs, which required app-wide
// `webSecurity: false` (it disables same-origin policy EVERYWHERE, not just for
// figures). Instead we serve them through a dedicated, locked-down custom scheme
// (`spyde-fig://`) so `webSecurity` can stay at its secure default (true).
//
// The scheme is registered as a STANDARD + SECURE + fetch-capable origin so the
// figure page (origin `spyde-fig://figures`) can dynamic-`import()` its sibling
// JS bundle under the SAME origin (cross-scheme module import from a secure page
// to `file://` is blocked by web security — which is the whole point).
const FIG_SCHEME = 'spyde-fig'
const FIG_HOST = 'figures'
const ICON_HOST = 'icons'
// Only ever serve the two kinds of files SpyDE itself writes to tmp; anything
// else (path traversal, arbitrary reads) is refused.
const FIG_NAME_RE = /^spyde_(?:fig_[\w.-]+\.html|figure_esm_[0-9a-f]+\.js)$/
// Toolbar icons are package assets (absolute paths from the Python backend's
// resolve_icon_path). With webSecurity enabled the renderer's http://localhost
// (dev) origin can't load file:// <img> subresources, so icons are served via
// this scheme instead. Guard: the resolved real path must be an .svg/.png that
// lives under a ".../spyde/.../icons/" directory — no arbitrary reads.
const ICON_EXT_RE = /\.(svg|png)$/i
const ICON_CONTAINMENT_RE = /[/\\]spyde[/\\](?:.*[/\\])?icons[/\\][^/\\]+$/i

protocol.registerSchemesAsPrivileged([
  { scheme: FIG_SCHEME, privileges: { standard: true, secure: true, supportFetchAPI: true } },
])

/** Map a `spyde-fig://figures/<name>` request to its temp-dir file, with a strict
 *  basename allowlist (no traversal, no arbitrary reads). */
function resolveFigPath(reqUrl: string): string | null {
  let u: URL
  try { u = new URL(reqUrl) } catch { return null }
  if (u.host !== FIG_HOST) return null
  const name = basename(decodeURIComponent(u.pathname))
  if (!FIG_NAME_RE.test(name)) return null
  const full = join(tmpdir(), name)
  // basename() already strips any `..`; double-check the resolved file actually
  // lives directly in tmpdir and exists before serving.
  if (join(tmpdir(), basename(full)) !== full || !existsSync(full)) return null
  return full
}

/** A `spyde-fig://figures/<name>` URL for a temp file basename. */
function figUrl(name: string): string {
  return `${FIG_SCHEME}://${FIG_HOST}/${encodeURIComponent(name)}`
}

/** Map a `spyde-fig://icons/<abs-path>` request to a package icon file. Serves
 *  only an .svg/.png whose REAL path lives under a spyde ".../icons/" directory;
 *  realpath collapses any `..`/symlink before the containment + extension check. */
function resolveIconPath(reqUrl: string): string | null {
  let u: URL
  try { u = new URL(reqUrl) } catch { return null }
  if (u.host !== ICON_HOST) return null
  // pathname is "/<percent-encoded-absolute-path>"; strip the leading slash.
  const raw = decodeURIComponent(u.pathname.replace(/^\//, ''))
  if (!raw || !ICON_EXT_RE.test(raw) || !existsSync(raw)) return null
  let real: string
  try { real = realpathSync(raw) } catch { return null }
  if (!ICON_CONTAINMENT_RE.test(real) || !ICON_EXT_RE.test(real)) return null
  return real
}

// Messages from the Python backend can arrive before the renderer has finished
// loading and registered its ipcRenderer listener. webContents.send() drops
// anything sent before the frame is ready, which silently swallowed the FIRST
// message after a quiet period — e.g. the nav_shape_prompt when opening a file
// (the dialog then only appeared once a LATER load pushed more messages). Buffer
// until the renderer signals ready, then flush in order.
let rendererReady = false
const pendingMessages: Array<Record<string, unknown>> = []

// Figure HTML files written to tmpdir (served via spyde-fig://). Tracked so we
// can remove them on quit — otherwise a long session with many re-rendered
// figures leaves spyde_fig_*.html accumulating in the OS temp dir.
const figTmpFiles = new Set<string>()

function cleanupFigTmpFiles(): void {
  for (const p of figTmpFiles) {
    try { rmSync(p, { force: true }) } catch { /* best-effort temp cleanup */ }
  }
  figTmpFiles.clear()
}

function rendererAlive(): boolean {
  return !!win && !win.isDestroyed() && !win.webContents.isDestroyed()
}

function sendToRenderer(msg: Record<string, unknown>): void {
  if (rendererReady && rendererAlive()) {
    win!.webContents.send('spyde:message', msg)
  } else {
    pendingMessages.push(msg)
  }
}

function flushPendingMessages(): void {
  rendererReady = true
  if (!rendererAlive()) return
  for (const msg of pendingMessages.splice(0)) {
    win!.webContents.send('spyde:message', msg)
  }
}

// ── Window creation ──────────────────────────────────────────────────────────

// App icon (window/taskbar in dev + unpackaged runs; on packaged Win/macOS the
// OS instead shows the icon electron-builder baked into the exe/app bundle at
// build time — see electron-builder.yml's icon: fields — but Linux and dev
// mode both read this BrowserWindow option, so it's still needed here).
// Dev: __dirname is electron/out/main → three levels up is the repo root,
// where the icon lives (spyde/Spyde.png) — same navigation pythonEnv's
// projectRoot uses. Packaged: bundle-python.mjs stages the WHOLE spyde/
// source tree (icons included, only tests/__pycache__ excluded) into
// <app resources>/python/spyde/, so the identical file is right there too.
function resolveAppIcon(): string {
  const packaged = join(process.resourcesPath, 'python', 'spyde', 'Spyde.png')
  if (app.isPackaged && existsSync(packaged)) return packaged
  return join(__dirname, '..', '..', '..', 'spyde', 'Spyde.png')
}

function createWindow(): BrowserWindow {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    icon: resolveAppIcon(),
    // ONE custom dark title bar on every platform (no native bar stacked on top
    // of ours — the Windows "two header bars" bug came from 'hiddenInset', which
    // is macOS-only and left the native Windows frame in place).
    //  - macOS: 'hidden' keeps the traffic-light buttons overlaid top-left.
    //  - Windows/Linux: 'hidden' + titleBarOverlay draws native, DARK-THEMED
    //    min/max/close buttons top-right (OS handles hit-test/snap), so we don't
    //    hand-roll window controls.
    titleBarStyle: 'hidden',
    ...(process.platform !== 'darwin'
      ? { titleBarOverlay: { color: '#181825', symbolColor: '#cdd6f4', height: 38 } }
      : {}),
    backgroundColor: '#11111b',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      // webSecurity stays at its secure default (true). Figures load via the
      // dedicated `spyde-fig://` scheme (registered above), not raw file://, so
      // same-origin policy is NOT disabled app-wide anymore.
    },
  })

  // Once the renderer frame has loaded (and its ipcRenderer listener is live),
  // flush any messages the backend emitted during startup. A fresh reload resets
  // the gate so buffered messages aren't sent to a frame that's tearing down.
  win.webContents.on('did-finish-load', flushPendingMessages)
  win.webContents.on('did-start-navigation', (_e, _url, isInPlace, isMainFrame) => {
    if (isMainFrame && !isInPlace) rendererReady = false
  })

  if (process.env['ELECTRON_RENDERER_URL']) {
    win.loadURL(process.env['ELECTRON_RENDERER_URL'])
    // DevTools is opt-in (SPYDE_DEVTOOLS=1) — auto-opening it spams the console
    // with harmless Chromium "Autofill.enable" protocol errors.
    if (process.env['SPYDE_DEVTOOLS'] === '1') {
      win.webContents.openDevTools({ mode: 'detach' })
    }
  } else {
    win.loadFile(join(__dirname, '../renderer/index.html'))
  }

  return win
}

// ── App lifecycle ─────────────────────────────────────────────────────────────

app.whenReady().then(async () => {
  // The whole app chrome is dark; force prefers-color-scheme: dark so the
  // anyplotlib figures (which auto-theme off it) render in dark mode to match.
  nativeTheme.themeSource = 'dark'

  // Serve figure HTML/JS through the locked-down `spyde-fig://` scheme (see the
  // scheme registration near the top). Refuses anything outside the temp-dir
  // basename allowlist.
  protocol.handle(FIG_SCHEME, async (request) => {
    const host = (() => { try { return new URL(request.url).host } catch { return '' } })()
    const filePath = host === ICON_HOST
      ? resolveIconPath(request.url)
      : resolveFigPath(request.url)
    if (!filePath) return new Response('Not found', { status: 404 })
    return net.fetch(pathToFileURL(filePath).href)
  })

  // Tell the preload whether this is a packaged production app, so the renderer
  // can gate test-only hooks. `app.isPackaged` is only readable in the main
  // process; forward it via an env var the preload reads. Set BEFORE the window
  // (and thus the preload) loads.
  if (app.isPackaged) process.env.SPYDE_PACKAGED = '1'

  createWindow()

  // Wire autoUpdater events now that there's a window to report them to. Does
  // NOT check yet — the startup check below fires a few seconds later so it
  // doesn't compete with the Python sidecar coming up.
  initUpdater(win!, app.getPath('userData'))

  // Resolve (and on first packaged run, create via `uv sync`) the Python
  // sidecar env, then start the backend.
  // __dirname is electron/out/main → three levels up is the repo root (dev),
  // where spyde's pyproject.toml lives (so `uv run` resolves the right env).
  const projectRoot = join(__dirname, '..', '..', '..')
  const { cmd, cwd } = await resolvePythonEnv({
    isPackaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
    projectRoot,
    userData: app.getPath('userData'),
    onProgress: (line) => {
      process.stderr.write(`[uv] ${line}`)
      // Surface first-run env setup in the UI (the stream/log channel).
      win?.webContents.send('spyde:stream', line, 'stderr')
      sendToRenderer({ type: 'status', text: 'Setting up Python environment…' })
    },
  }).catch((err) => {
    const msg = `Python environment setup failed: ${err?.message ?? err}`
    console.error(`[spyde] ${msg}`)
    sendToRenderer({ type: 'error', text: msg })
    // Fall back to dev-style launch so a broken bundle is still diagnosable.
    return { cmd: ['uv', 'run', 'python', '-m', 'spyde'], cwd: projectRoot }
  })

  startSpyDE(cmd, {
    onMessage: (msg) => {
      // Figure HTML must be written to disk here in the main process (the
      // renderer is a browser sandbox with no fs). It's served back to the
      // iframe through the locked-down `spyde-fig://` scheme (NOT raw file://),
      // so app-wide webSecurity can stay enabled.
      if (msg.type === 'figure' && msg.html && msg.fig_id) {
        const figName = `spyde_fig_${String(msg.fig_id)}.html`
        const figPath = join(tmpdir(), figName)
        figTmpFiles.add(figPath)   // tracked for cleanup on quit
        try {
          // The Python side embeds a dynamic `import("file://…/spyde_figure_esm_*.js")`
          // for the shared JS bundle. A secure-scheme page can't import a
          // file:// module (cross-scheme), so rewrite that import to the SAME
          // spyde-fig:// origin (the bundle is served by the same handler). The
          // basename allowlist on the handler still gates which file is read.
          // The path separator before the filename is `/` on posix and an
          // escaped `\\` on Windows (Python JSON-escapes the backslashes), so
          // capture the basename irrespective of separators.
          const html = (msg.html as string).replace(
            /import\(\s*["']file:\/\/.*?(spyde_figure_esm_[0-9a-f]+\.js)["']\s*\)/g,
            (_m, name: string) => `import(${JSON.stringify(figUrl(name))})`,
          )
          writeFileSync(figPath, html, 'utf8')
          msg = { ...msg, file_url: figUrl(figName), html: undefined }
        } catch { /* leave msg as-is on failure */ }
      }
      // Echo key lifecycle messages to the dev terminal so backend health is
      // visible without opening devtools.
      if (msg.type === 'ready' || msg.type === 'dask_ready' || msg.type === 'error') {
        console.log(`[spyde backend] ${msg.type}: ${msg.text ?? msg.dashboard ?? ''}`)
      }
      sendToRenderer(msg)
    },
    onStream: (text, kind) => {
      // Forward to the renderer AND surface in the dev terminal.
      process[kind === 'stderr' ? 'stderr' : 'stdout'].write(`[spyde] ${text}`)
      // Guard: at teardown the backend stream can still emit a chunk after the
      // window/webContents is destroyed → "TypeError: Object has been destroyed".
      // Only forward to a live webContents.
      if (win && !win.isDestroyed() && !win.webContents.isDestroyed()) {
        win.webContents.send('spyde:stream', text, kind)
      }
    },
  }, cwd)

  buildMenu()

  // Startup check (locked decision: startup + manual, not silent background
  // auto-download). Delayed so it doesn't compete with the Python sidecar's
  // own startup work for network/CPU on a slow first launch.
  setTimeout(() => checkForUpdates(), 5000)

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  stopSpyDE()
  cleanupFigTmpFiles()
  if (process.platform !== 'darwin') app.quit()
})

// Tear the backend down on EVERY quit path, not just when the last window
// closes (e.g. macOS Cmd-Q, the File→Quit menu role, an app.quit() from IPC).
// stopSpyDE() is idempotent + null-safe, so overlapping with window-all-closed
// is harmless. Without this the Python sidecar (and its Dask workers) could
// outlive the UI on those paths.
app.on('before-quit', () => { stopSpyDE(); cleanupFigTmpFiles() })

// A console SIGINT/SIGTERM (Ctrl-C in `npm run dev`, or a parent killing us)
// bypasses the normal Electron quit events, so kill the backend explicitly then
// exit. Guard against double-registration under HMR with `.once` semantics via
// a flag is unnecessary here — main is evaluated once per process.
for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    stopSpyDE()
    cleanupFigTmpFiles()
    app.quit()
    // Give the graceful-quit write + tree-kill a moment, then hard-exit so the
    // signal isn't swallowed if Electron's own teardown stalls.
    setTimeout(() => process.exit(0), 2000)
  })
}

// ── Application menu ──────────────────────────────────────────────────────────

function buildMenu(): void {
  const menu = Menu.buildFromTemplate([
    {
      label: 'File',
      submenu: [
        {
          label: 'Open…',
          accelerator: 'CmdOrCtrl+O',
          click: async () => {
            const result = await dialog.showOpenDialog(win!, {
              properties: ['openFile', 'multiSelections'],
              filters: [
                { name: 'EM Data', extensions: ['hspy', 'zspy', 'mrc', 'tif', 'tiff', 'de5'] },
                { name: 'HyperSpy', extensions: ['hspy', 'zspy'] },
                { name: 'MRC', extensions: ['mrc'] },
                { name: 'TIFF', extensions: ['tif', 'tiff'] },
              ],
            })
            if (!result.canceled) {
              for (const p of result.filePaths) {
                sendAction('open_file', { path: p })
              }
            }
          },
        },
        {
          label: 'Open Zarr Folder (.zspy)…',
          // .zspy / .zarr are Zarr DIRECTORY stores, not files — the native
          // file picker can't select them (and on Windows a dialog can't mix
          // file + folder modes), so this uses a directory picker.
          click: async () => {
            const result = await dialog.showOpenDialog(win!, {
              title: 'Open a .zspy / .zarr dataset folder',
              properties: ['openDirectory', 'multiSelections'],
            })
            if (!result.canceled) {
              for (const p of result.filePaths) {
                sendAction('open_file', { path: p })
              }
            }
          },
        },
        {
          label: 'Load Stack…',
          // Opens the in-app StackDialog (reorderable list of datasets) rather
          // than the native picker — the user adds/reorders there, then confirms.
          click: () => win?.webContents.send('spyde:open_stack_dialog'),
        },
        { type: 'separator' },
        {
          label: 'Save Signal…',
          accelerator: 'CmdOrCtrl+S',
          click: async () => {
            const result = await dialog.showSaveDialog(win!, {
              // Default to .zspy (Zarr folder store — lazy, chunked, the format
              // SpyDE prefers); .hspy still available as a secondary option.
              defaultPath: 'signal.zspy',
              filters: [
                { name: 'Zarr (.zspy)', extensions: ['zspy'] },
                { name: 'HyperSpy (.hspy)', extensions: ['hspy'] },
              ],
            })
            if (!result.canceled && result.filePath) {
              sendAction('save_signal', { path: result.filePath })
            }
          },
        },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'Examples',
      submenu: [
        'mgo_nanocrystals',
        'small_ptychography',
        'zrnb_precipitate',
        'pdcusi_insitu',
        'sped_ag',
        'fe_multi_phase_grains',
      ].map((name) => ({
        label: name,
        click: () => sendAction('load_example', { name }),
      })),
    },
    { role: 'viewMenu' as const },
    {
      label: 'Window',
      submenu: [
        {
          label: 'Tile Windows',
          click: () => win?.webContents.send('spyde:tile'),
        },
        { role: 'minimize' as const },
      ],
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Guided Tour: Finding Diffraction Vectors',
          click: () => win?.webContents.send('spyde:start_guide', 'find-vectors'),
        },
        { type: 'separator' },
        {
          label: 'Dask Dashboard',
          click: () => win?.webContents.send('spyde:open_dashboard'),
        },
        {
          label: 'GitHub',
          click: () => shell.openExternal('https://github.com/cssfrancis/spyde'),
        },
        { type: 'separator' },
        {
          label: 'Check for Updates…',
          click: () => win?.webContents.send('spyde:open_update_dialog'),
        },
        {
          label: 'GPU Status…',
          click: () => win?.webContents.send('spyde:open_gpu_status_dialog'),
        },
      ],
    },
  ])
  Menu.setApplicationMenu(menu)
}

// ── IPC handlers (renderer → main → Python) ──────────────────────────────────

/** Forward a toolbar action to Python. */
ipcMain.on('spyde:action', (_, action: string, payload: Record<string, unknown>, windowId?: number) => {
  sendAction(action, payload, windowId)
})

/** Open a native file dialog and send path to Python. */
ipcMain.handle('spyde:open-file', async () => {
  const result = await dialog.showOpenDialog(win!, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'EM Data', extensions: ['hspy', 'zspy', 'mrc', 'tif', 'tiff', 'de5'] },
    ],
  })
  if (!result.canceled) {
    for (const p of result.filePaths) {
      sendAction('open_file', { path: p })
    }
  }
})

/** Open a .zspy/.zarr DIRECTORY store (folder picker → load). */
ipcMain.handle('spyde:open-zarr-folder', async () => {
  const result = await dialog.showOpenDialog(win!, {
    title: 'Open a .zspy / .zarr dataset folder',
    properties: ['openDirectory', 'multiSelections'],
  })
  if (!result.canceled) {
    for (const p of result.filePaths) sendAction('open_file', { path: p })
  }
})

/** Quit the app (custom-titlebar menu has no native File→Quit on Win/Linux). */
ipcMain.handle('spyde:quit', () => app.quit())

/** Multi-select file picker that RETURNS the chosen paths to the renderer (does
 *  NOT auto-open). Used by the StackDialog's "Add datasets…" tile. */
ipcMain.handle('spyde:pick-files', async (_e, opts?: { name?: string; extensions?: string[] }) => {
  const exts = (opts?.extensions ?? ['hspy', 'zspy', 'mrc', 'tif', 'tiff', 'de5']).map((e) =>
    e.replace(/^\./, ''),
  )
  const result = await dialog.showOpenDialog(win!, {
    title: 'Add datasets to the stack',
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: opts?.name ?? 'EM Data', extensions: exts }],
  })
  return result.canceled ? [] : result.filePaths
})

/** Multi-select DIRECTORY picker (for .zspy / .zarr Zarr stores, which are
 *  folders not files — Windows can't mix file + folder selection in one dialog). */
ipcMain.handle('spyde:pick-folders', async () => {
  const result = await dialog.showOpenDialog(win!, {
    title: 'Add .zspy / .zarr folders to the stack',
    properties: ['openDirectory', 'multiSelections'],
  })
  return result.canceled ? [] : result.filePaths
})

/** Pick a file and RETURN its path to the renderer (for action params, e.g. a
 *  .cif crystal structure) — does NOT auto-open it as a dataset. */
ipcMain.handle('spyde:pick-file', async (_e, opts: { name?: string; extensions?: string[] }) => {
  const exts = (opts?.extensions ?? []).map((e) => e.replace(/^\./, ''))
  const result = await dialog.showOpenDialog(win!, {
    properties: ['openFile'],
    filters: exts.length ? [{ name: opts?.name ?? 'Files', extensions: exts }] : [],
  })
  return result.canceled || !result.filePaths.length ? null : result.filePaths[0]
})

/** Save dialog. */
ipcMain.handle('spyde:save-dialog', async () => {
  const result = await dialog.showSaveDialog(win!, {
    // Default to .zspy (Zarr folder store); .hspy available as a fallback.
    defaultPath: 'signal.zspy',
    filters: [
      { name: 'Zarr (.zspy)', extensions: ['zspy'] },
      { name: 'HyperSpy (.hspy)', extensions: ['hspy'] },
    ],
  })
  if (!result.canceled && result.filePath) {
    sendAction('save_signal', { path: result.filePath })
  }
})

/** Forward figure interaction events to Python. */
ipcMain.on('spyde:figure-event', (_, figId: string, eventJson: string) =>
  sendFigureEvent(figId, eventJson)
)

/** Forward MDI resize to Python. */
ipcMain.on('spyde:resize', (_, figId: string, width: number, height: number) =>
  sendResize(figId, width, height)
)

// Only hand a URL to the OS if it parses AND uses a protocol we trust. Without
// this, the renderer (or any compromised iframe content posting through it)
// could ask the OS to open arbitrary `file:`, custom-scheme, or `javascript:`
// URLs — a classic shell.openExternal abuse. Allowlist web + mail only.
const OPEN_EXTERNAL_ALLOWED = new Set(['https:', 'http:', 'mailto:'])
ipcMain.on('open-external', (_, url: string) => {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    console.warn(`[spyde] open-external rejected unparseable URL: ${url}`)
    return
  }
  if (!OPEN_EXTERNAL_ALLOWED.has(parsed.protocol)) {
    console.warn(`[spyde] open-external rejected disallowed protocol: ${parsed.protocol}`)
    return
  }
  shell.openExternal(parsed.href)
})

// ── Update / GPU-status IPC ───────────────────────────────────────────────────

/** Manual "Check Now" from the update dialog. Result arrives async via the
 *  spyde:update-status push channel (checking -> available/not-available). */
ipcMain.on('spyde:check-for-updates', () => checkForUpdates())

/** User clicked "Download" in the update dialog. */
ipcMain.on('spyde:download-update', () => downloadUpdate())

/** User clicked "Restart to Install" once the update finished downloading. */
ipcMain.on('spyde:quit-and-install', () => quitAndInstall())

/** Current channel + whether this build even supports auto-update (dev/e2e
 *  builds have no app-update.yml, so the dialog can say so instead of
 *  silently doing nothing). */
ipcMain.handle('spyde:get-update-info', () => ({
  channel: readUpdateChannel(),
  supported: updatesSupported(),
  status: getLastUpdateStatus(),
  appVersion: app.getVersion(),
}))

/** Channel radio in the update dialog — persisted Electron-side (updater.ts)
 *  AND mirrored into ~/.spyde/settings.json via the Python action so it's
 *  visible/debuggable from that side too. */
ipcMain.on('spyde:set-update-channel', (_, channel: 'stable' | 'beta') => {
  setUpdateChannel(channel)
  sendAction('set_update_channel', { channel })
})
