/**
 * index.ts — Electron main process for SpyDE.
 */
import { app, BrowserWindow, dialog, ipcMain, Menu, shell, nativeTheme } from 'electron'
import { join } from 'path'
import { tmpdir } from 'os'
import { writeFileSync } from 'fs'
import {
  startSpyDE, sendAction, sendFigureEvent, sendResize,
  stopSpyDE,
} from './runner'
import { resolvePythonEnv } from './pythonEnv'

let win: BrowserWindow | null = null

// ── Window creation ──────────────────────────────────────────────────────────

function createWindow(): BrowserWindow {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#11111b',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,   // needed so iframes can load file:// HTML
    },
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
      win?.webContents.send('spyde:message', { type: 'status', text: 'Setting up Python environment…' })
    },
  }).catch((err) => {
    const msg = `Python environment setup failed: ${err?.message ?? err}`
    console.error(`[spyde] ${msg}`)
    win?.webContents.send('spyde:message', { type: 'error', text: msg })
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
          msg = { ...msg, file_url: `file://${figPath}`, html: undefined }
        } catch { /* leave msg as-is on failure */ }
      }
      // Echo key lifecycle messages to the dev terminal so backend health is
      // visible without opening devtools.
      if (msg.type === 'ready' || msg.type === 'dask_ready' || msg.type === 'error') {
        console.log(`[spyde backend] ${msg.type}: ${msg.text ?? msg.dashboard ?? ''}`)
      }
      win?.webContents.send('spyde:message', msg)
    },
    onStream: (text, kind) => {
      // Forward to the renderer AND surface in the dev terminal.
      process[kind === 'stderr' ? 'stderr' : 'stdout'].write(`[spyde] ${text}`)
      win?.webContents.send('spyde:stream', text, kind)
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
        { type: 'separator' },
        {
          label: 'Save Signal…',
          accelerator: 'CmdOrCtrl+S',
          click: async () => {
            const result = await dialog.showSaveDialog(win!, {
              filters: [{ name: 'HyperSpy', extensions: ['hspy'] }],
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
    filters: [{ name: 'HyperSpy', extensions: ['hspy'] }],
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

ipcMain.on('open-external', (_, url: string) => shell.openExternal(url))
