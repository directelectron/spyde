/**
 * @de/shell-main — the Electron main-process kernel shared by SpyDE,
 * de-groundcrew and de-autopilot.
 *
 * What lives here is everything that answers "how do I run a desktop app with a
 * Python brain?" — spawning and supervising the sidecar, bootstrapping its uv
 * environment on first launch, auto-update, and the identity plumbing that ties
 * those to a particular app.
 *
 * Call `configureShell()` FIRST, before any other export in this package: the
 * IPC channel prefix, settings directory, Python module name and packaged-app
 * env var all read from it, and they throw rather than guess.
 */
export { configureShell, shellConfig, channel, envPrefix } from './config'
export type { ShellConfig } from './config'

export {
  startBackend, stopBackend, sendAction, sendFigureEvent, sendResize,
  recentBackendOutput,
} from './backendProcess'

export { recordProblem, recordedProblems } from './problemLog'
export type { Problem } from './problemLog'

export {
  initErrorReporting, reportingConfigured, collectDiagnostics, submitReport,
} from './errorReport'
export type { Diagnostics, ReportResult } from './errorReport'

export { parseSentryDsn } from './sentryEnvelope'

export { registerShellDialogs } from './dialogs'
export type { FileFilter } from './dialogs'

export { createShellWindow } from './window'
export type { ShellWindow, ShellWindowOptions } from './window'
export type { BackendHandlers } from './backendProcess'

export {
  resolvePythonEnv, managedEnvPaths, venvPython, readLockedTorchVersion,
  installTorchPerMachine,
} from './pythonEnv'
export type { ResolvedPython, EnsureOptions } from './pythonEnv'

export { parseUvLine } from './envProgress'
export type { EnvPhase, EnvProgressEvent } from './envProgress'

export {
  initUpdater, checkForUpdates, downloadUpdate, quitAndInstall, resetToIdle,
  readUpdateChannel, setUpdateChannel, getLastUpdateStatus, updatesSupported,
  friendlyError,
} from './updater'
export type { UpdateChannel, UpdateStatus } from './updater'

export {
  isPrereleaseVersion, defaultChannelForVersion, truncateMessage,
} from './updaterErrors'
