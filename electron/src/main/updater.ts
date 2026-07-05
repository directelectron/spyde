/**
 * updater.ts — electron-updater wiring (check / download / install, stable vs
 * beta channel).
 *
 * The GitHub `publish` provider (electron-builder.yml) is what release.yml
 * populates: each tagged release gets latest.yml / latest-mac.yml /
 * latest-linux.yml alongside the installers, which is what autoUpdater reads
 * to detect a new version per platform. Plain vX.Y.Z tags are regular GitHub
 * releases; vX.Y.Z-rc.N/-beta.N tags are marked `prerelease` (release.yml's
 * `channel` job) — `allowPrerelease` is what gates whether autoUpdater will
 * offer those to a given install (see electron-updater's own default: it
 * auto-allows prereleases only when the CURRENTLY INSTALLED version already
 * has a prerelease component, so an explicit stable->beta opt-in needs us to
 * set this ourselves).
 *
 * autoDownload is OFF: check -> tell the renderer -> user clicks "Download" ->
 * we call downloadUpdate() -> "Restart to install" -> quitAndInstall(). This
 * matches the "click here to update" ask (not a silent background install).
 */
import { app, BrowserWindow } from 'electron'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { join } from 'path'
import { autoUpdater } from 'electron-updater'

export type UpdateChannel = 'stable' | 'beta'

export type UpdateStatus =
  | { state: 'idle' }
  | { state: 'checking' }
  | { state: 'available'; version: string; releaseNotes?: string }
  | { state: 'not-available' }
  | { state: 'downloading'; percent: number }
  | { state: 'downloaded'; version: string }
  | { state: 'error'; message: string }

let win: BrowserWindow | null = null
let channelFilePath = ''
let userDataDir = ''
let lastStatus: UpdateStatus = { state: 'idle' }

function sendStatus(status: UpdateStatus): void {
  lastStatus = status
  if (win && !win.isDestroyed() && !win.webContents.isDestroyed()) {
    win.webContents.send('spyde:update-status', status)
  }
}

export function getLastUpdateStatus(): UpdateStatus {
  return lastStatus
}

/** Read the persisted channel choice (defaults to 'stable'). Electron-side
 *  storage, separate from ~/.spyde/settings.json — updater.ts must be able to
 *  read this before the Python sidecar is necessarily up. */
export function readUpdateChannel(): UpdateChannel {
  try {
    const raw = readFileSync(channelFilePath, 'utf8').trim()
    return raw === 'beta' ? 'beta' : 'stable'
  } catch {
    return 'stable'
  }
}

export function setUpdateChannel(channel: UpdateChannel): void {
  autoUpdater.allowPrerelease = channel === 'beta'
  try {
    mkdirSync(userDataDir, { recursive: true })
    writeFileSync(channelFilePath, channel, 'utf8')
  } catch (err) {
    console.error('[updater] failed to persist update channel:', err)
  }
}

/** Wire autoUpdater events + do the initial channel read. Call once from
 *  app.whenReady() after the window exists (there's somewhere to show a
 *  result). Does NOT check for updates itself — see checkForUpdates(). */
export function initUpdater(mainWindow: BrowserWindow, userData: string): void {
  win = mainWindow
  userDataDir = userData
  channelFilePath = join(userData, 'update-channel.json')

  autoUpdater.autoDownload = false
  autoUpdater.autoInstallOnAppQuit = false
  autoUpdater.allowPrerelease = readUpdateChannel() === 'beta'

  autoUpdater.on('checking-for-update', () => sendStatus({ state: 'checking' }))

  autoUpdater.on('update-available', (info) =>
    sendStatus({ state: 'available', version: info.version, releaseNotes: releaseNotesText(info) }),
  )

  autoUpdater.on('update-not-available', () => sendStatus({ state: 'not-available' }))

  autoUpdater.on('download-progress', (progress) =>
    sendStatus({ state: 'downloading', percent: Math.round(progress.percent) }),
  )

  autoUpdater.on('update-downloaded', (info) =>
    sendStatus({ state: 'downloaded', version: info.version }),
  )

  autoUpdater.on('error', (err) =>
    sendStatus({ state: 'error', message: err?.message ?? String(err) }),
  )
}

function releaseNotesText(info: { releaseNotes?: string | Array<{ note?: string | null }> | null }): string | undefined {
  if (typeof info.releaseNotes === 'string') return info.releaseNotes
  if (Array.isArray(info.releaseNotes)) {
    return info.releaseNotes.map((n) => n.note ?? '').filter(Boolean).join('\n\n')
  }
  return undefined
}

/** Manual or startup check. Safe to call repeatedly (electron-updater no-ops
 *  a check already in flight). Not packaged (dev/e2e) -> no-op, since there's
 *  no installed app for electron-updater to reason about updating. */
export function checkForUpdates(): void {
  if (!app.isPackaged) {
    sendStatus({ state: 'not-available' })
    return
  }
  autoUpdater.checkForUpdates().catch((err) => sendStatus({ state: 'error', message: err?.message ?? String(err) }))
}

export function downloadUpdate(): void {
  autoUpdater.downloadUpdate().catch((err) => sendStatus({ state: 'error', message: err?.message ?? String(err) }))
}

export function quitAndInstall(): void {
  autoUpdater.quitAndInstall()
}

/** Whether this is a build electron-builder/electron-updater can actually
 *  act on (a real installed app, not `electron .` dev / the e2e harness). */
export function updatesSupported(): boolean {
  return app.isPackaged && existsSync(join(process.resourcesPath, 'app-update.yml'))
}
