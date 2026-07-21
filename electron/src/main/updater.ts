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
 * offer those to a given install.
 *
 * CHANNEL DEFAULT FOLLOWS THE RUNNING BUILD. The default channel is derived from
 * THIS build's own version: a prerelease build (…-rc.N) tracks beta, a plain
 * X.Y.Z tracks stable (defaultChannelForVersion in updater_errors.ts). So an rc
 * install checks the beta feed (where its updates actually are) instead of the
 * stable feed — which, with no stable release published yet, would 404 and error.
 * An explicit user choice (the dialog radio, persisted) always overrides the
 * default. And a "no release for this channel" result is reported as up-to-date,
 * not an error (isNoReleaseForChannel).
 *
 * autoDownload is OFF: check -> tell the renderer -> user clicks "Download" ->
 * we call downloadUpdate() -> "Restart to install" -> quitAndInstall(). This
 * matches the "click here to update" ask (not a silent background install).
 *
 * HARDENING (flaky GitHub can HANG, not just crash): every network step is
 * bounded by a timeout so a half-open connection can't wedge the UI in
 * 'checking'/'downloading' forever (with the "Check Now" button disabled in
 * exactly those states → unrecoverable). A stall detector watches the download
 * for silence. Any timeout/error leaves the updater RECHECKABLE (the in-flight
 * guard clears), raw electron-updater strings are mapped to friendly text, and
 * quitAndInstall() can no longer throw the process down.
 */
import { app, BrowserWindow } from 'electron'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { join } from 'path'
import { autoUpdater } from 'electron-updater'
import { friendlyError, defaultChannelForVersion } from './updater_errors'

export type UpdateChannel = 'stable' | 'beta'

export type UpdateStatus =
  | { state: 'idle' }
  | { state: 'checking' }
  | { state: 'available'; version: string; releaseNotes?: string }
  | { state: 'not-available' }
  | { state: 'downloading'; percent: number }
  | { state: 'downloaded'; version: string }
  | { state: 'error'; message: string }

// Bound the two network steps. A flaky/half-open GitHub connection otherwise
// leaves autoUpdater's promise pending forever (the 'error' event never fires),
// wedging the UI in 'checking'/'downloading'. Mirrors the PDF_EXPORT_TIMEOUT_MS
// race idiom in index.ts.
const CHECK_TIMEOUT_MS = 30_000
// The download reports periodic 'download-progress' events; if none arrives for
// this long while 'downloading', the transfer has stalled (a half-open socket
// mid-stream doesn't reject, it just goes quiet).
const DOWNLOAD_STALL_MS = 60_000

let win: BrowserWindow | null = null
let channelFilePath = ''
let userDataDir = ''
let lastStatus: UpdateStatus = { state: 'idle' }

// In-flight guards. `checkInFlight` prevents overlapping checks AND is what a
// timeout/error must CLEAR so a subsequent checkForUpdates() isn't blocked by a
// stale belief that a check is still running (the "guaranteed recovery" rule).
let checkInFlight = false
let checkTimer: ReturnType<typeof setTimeout> | null = null
let downloadStallTimer: ReturnType<typeof setTimeout> | null = null

function clearCheckTimer(): void {
  if (checkTimer) { clearTimeout(checkTimer); checkTimer = null }
}
function clearDownloadStallTimer(): void {
  if (downloadStallTimer) { clearTimeout(downloadStallTimer); downloadStallTimer = null }
}

// friendlyError lives in updater_errors.ts (pure, dependency-free) so it's unit-
// testable with node:test on every OS. Imported above; re-exported to keep the
// existing public API (some callers/tests import it from updater.ts).
export { friendlyError } from './updater_errors'

/** Every error/timeout path funnels through here so state stays consistent:
 *  friendly text out, in-flight guard + timers cleared so a retry works. */
function reportError(rawMessage: string): void {
  checkInFlight = false
  clearCheckTimer()
  clearDownloadStallTimer()
  sendStatus({ state: 'error', message: friendlyError(rawMessage) })
}

function sendStatus(status: UpdateStatus): void {
  lastStatus = status
  if (win && !win.isDestroyed() && !win.webContents.isDestroyed()) {
    win.webContents.send('spyde:update-status', status)
  }
}

export function getLastUpdateStatus(): UpdateStatus {
  return lastStatus
}

/** Read the effective update channel. An EXPLICIT persisted choice (the user
 *  flipped the radio) always wins; otherwise the default follows the RUNNING
 *  build's own version — a prerelease build (…-rc.N) tracks beta, a plain X.Y.Z
 *  tracks stable. This is what stops an rc install from fruitlessly checking the
 *  (non-existent) stable channel and erroring. Electron-side storage, separate
 *  from ~/.spyde/settings.json — readable before the Python sidecar is up. */
export function readUpdateChannel(): UpdateChannel {
  try {
    const raw = readFileSync(channelFilePath, 'utf8').trim()
    if (raw === 'beta' || raw === 'stable') return raw
    // Any other/empty content → fall through to the version-derived default.
  } catch {
    // No persisted choice yet → derive from this build's version.
  }
  return defaultChannelForVersion(appVersion())
}

/** The running app's version. Wrapped so it's overridable/testable and safe if
 *  called before `app` is ready (falls back to a stable-looking version). */
function appVersion(): string {
  try {
    return app.getVersion()
  } catch {
    return '0.0.0'
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

  autoUpdater.on('update-available', (info) => {
    // A definite result arrived → the check is no longer in flight and its
    // timeout must not later fire a spurious "timed out" over this state.
    checkInFlight = false
    clearCheckTimer()
    sendStatus({ state: 'available', version: info.version, releaseNotes: releaseNotesText(info) })
  })

  autoUpdater.on('update-not-available', () => {
    checkInFlight = false
    clearCheckTimer()
    sendStatus({ state: 'not-available' })
  })

  autoUpdater.on('download-progress', (progress) => {
    // Each progress tick proves the transfer is alive — re-arm the stall watch.
    armDownloadStall()
    sendStatus({ state: 'downloading', percent: Math.round(progress.percent) })
  })

  autoUpdater.on('update-downloaded', (info) => {
    clearDownloadStallTimer()
    sendStatus({ state: 'downloaded', version: info.version })
  })

  // electron-updater surfaces network + verification failures here. But a
  // "no release found for this channel" error (e.g. a STABLE build/channel when
  // only prereleases exist yet — no stable release published) is NOT a failure —
  // it means "you're up to date on your channel". Report THAT as not-available so
  // the user sees a calm "up to date", not a scary error. Everything else routes
  // through reportError (friendly + recheckable).
  autoUpdater.on('error', (err) => {
    const msg = err?.message ?? String(err)
    if (isNoReleaseForChannel(msg)) {
      checkInFlight = false
      clearCheckTimer()
      sendStatus({ state: 'not-available' })
      return
    }
    reportError(msg)
  })
}

/** Recognise electron-updater's "there is no release matching this channel" error
 *  — the feed / release simply doesn't exist for the channel we asked for (a
 *  stable build when only prereleases have shipped). That's "up to date", not a
 *  fault. Kept conservative: only the shapes electron-updater/GitHubProvider emit
 *  for a genuinely-absent release, so a real 404-from-network still surfaces. */
function isNoReleaseForChannel(msg: string): boolean {
  const s = String(msg || '')
  return /Unable to find latest version on GitHub|No published versions on GitHub|latest-mac\.yml.*not found|Cannot find channel|No version found/i.test(s)
}

/** (Re)arm the download stall watchdog: if no progress event lands within
 *  DOWNLOAD_STALL_MS, treat the transfer as failed (recheckable). */
function armDownloadStall(): void {
  clearDownloadStallTimer()
  downloadStallTimer = setTimeout(() => {
    downloadStallTimer = null
    // Only fires while we still believe we're downloading — a completed/errored
    // download clears the timer, so reaching here means genuine silence.
    if (lastStatus.state === 'downloading') {
      reportError('The download stalled — check your connection and try again.')
    }
  }, DOWNLOAD_STALL_MS)
}

function releaseNotesText(info: { releaseNotes?: string | Array<{ note?: string | null }> | null }): string | undefined {
  if (typeof info.releaseNotes === 'string') return info.releaseNotes
  if (Array.isArray(info.releaseNotes)) {
    return info.releaseNotes.map((n) => n.note ?? '').filter(Boolean).join('\n\n')
  }
  return undefined
}

/** Force the updater back to a neutral, checkable state. Exposed so any external
 *  recovery (e.g. a renderer "Retry" that wants a clean slate) can reset it; the
 *  error path already leaves it recheckable, so this is belt-and-braces. */
export function resetToIdle(): void {
  checkInFlight = false
  clearCheckTimer()
  clearDownloadStallTimer()
  sendStatus({ state: 'idle' })
}

/** Manual or startup check. Safe to call repeatedly — an in-flight check is a
 *  no-op (guarded), and a prior timed-out/errored check has already cleared the
 *  guard so a retry proceeds. Not packaged (dev/e2e) -> no-op, since there's no
 *  installed app for electron-updater to reason about updating. */
export function checkForUpdates(): void {
  if (!app.isPackaged) {
    sendStatus({ state: 'not-available' })
    return
  }
  if (checkInFlight) return
  checkInFlight = true

  // Bound the check: a half-open GitHub connection never rejects the promise nor
  // fires 'error', so without this the UI wedges in 'checking' forever with the
  // "Check Now" button disabled. On timeout report + clear the guard so a retry
  // works (mirrors index.ts's PDF_EXPORT_TIMEOUT_MS race).
  clearCheckTimer()
  checkTimer = setTimeout(() => {
    checkTimer = null
    if (checkInFlight) {
      reportError('Update check timed out — check your connection and try again.')
    }
  }, CHECK_TIMEOUT_MS)

  autoUpdater.checkForUpdates().catch((err) => reportError(err?.message ?? String(err)))
}

export function downloadUpdate(): void {
  // A download that never starts producing progress (immediate half-open) would
  // otherwise sit forever — arm the stall watch up front; each progress event
  // re-arms it, downloaded/error clears it.
  armDownloadStall()
  autoUpdater.downloadUpdate().catch((err) => reportError(err?.message ?? String(err)))
}

export function quitAndInstall(): void {
  // The one call that used to be able to throw the process down. If the installer
  // handoff fails (missing/locked installer, permissions), surface a friendly
  // recoverable error instead of an uncaught exception crashing the app.
  try {
    clearDownloadStallTimer()
    autoUpdater.quitAndInstall()
  } catch (err) {
    reportError(`Couldn't start the installer — please download manually. (${(err as Error)?.message ?? String(err)})`)
  }
}

/** Whether this is a build electron-builder/electron-updater can actually
 *  act on (a real installed app, not `electron .` dev / the e2e harness). */
export function updatesSupported(): boolean {
  return app.isPackaged && existsSync(join(process.resourcesPath, 'app-update.yml'))
}
