/**
 * index.ts — Electron main process for SpyDE.
 */
import { app, BrowserWindow, dialog, ipcMain, Menu, shell, nativeTheme } from 'electron'
import { join } from 'path'
import { pathToFileURL } from 'url'
import { tmpdir } from 'os'
import { writeFileSync } from 'fs'
import {
  startSpyDE, sendAction, sendFigureEvent, sendResize,
  stopSpyDE,
} from './runner'
import { resolvePythonEnv } from './pythonEnv'

let win: BrowserWindow | null = null

// Messages from the Python backend can arrive before the renderer has finished
// loading and registered its ipcRenderer listener. webContents.send() drops
// anything sent before the frame is ready, which silently swallowed the FIRST
// message after a quiet period — e.g. the nav_shape_prompt when opening a file
// (the dialog then only appeared once a LATER load pushed more messages). Buffer
// until the renderer signals ready, then flush in order.
let rendererReady = false
const pendingMessages: Array<Record<string, unknown>> = []

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

function createWindow(): BrowserWindow {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
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
      webSecurity: false,   // needed so iframes can load file:// HTML
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

  // Tell the preload whether this is a packaged production app, so the renderer
  // can gate test-only hooks. `app.isPackaged` is only readable in the main
  // process; forward it via an env var the preload reads. Set BEFORE the window
  // (and thus the preload) loads.
  if (app.isPackaged) process.env.SPYDE_PACKAGED = '1'

  createWindow()

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
      // renderer is a browser sandbox with no fs). Forward a file:// URL.
      if (msg.type === 'figure' && msg.html && msg.fig_id) {
        const figPath = join(tmpdir(), `spyde_fig_${String(msg.fig_id)}.html`)
        try {
          writeFileSync(figPath, msg.html as string, 'utf8')
          // pathToFileURL produces a valid file URL on every OS — a bare
          // `file://${figPath}` is malformed on Windows (backslashes + drive
          // letter: `file://C:\…`), so the figure iframe never loaded there.
          msg = { ...msg, file_url: pathToFileURL(figPath).href, html: undefined }
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

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  stopSpyDE()
  if (process.platform !== 'darwin') app.quit()
})

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
