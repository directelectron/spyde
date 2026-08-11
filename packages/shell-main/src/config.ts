/**
 * config.ts — the per-app identity the shell needs but must not assume.
 *
 * Everything in this package used to say "spyde" out loud: the IPC channel
 * prefix, the settings directory, the Python module to spawn, the wheel name,
 * the setuptools_scm env-var suffix. None of that is shell knowledge — it is the
 * one thing that differs between SpyDE, de-groundcrew and de-autopilot.
 *
 * An app calls `configureShell()` once, at the top of its main process, before
 * anything else in this package runs.
 */

export interface ShellConfig {
  /** Lowercase, filesystem- and URL-safe app id, e.g. 'spyde', 'groundcrew'.
   *  Drives the IPC channel prefix (`<appId>:action`), the settings directory
   *  (`~/.<appId>/settings.json`), the packaged-app env var
   *  (`<APPID>_PACKAGED`), and the custom figure scheme (`<appId>-fig://`). */
  appId: string
  /** Human-readable name for user-facing strings ("… exited with code 1"). */
  appName: string
  /** The Python module the backend runs as: `python -m <pythonModule>`. */
  pythonModule: string
  /** Distribution name of the Python package, when it differs from the module
   *  (e.g. module `spyde` / dist `spyde`). Used for the pre-built wheel prefix
   *  and the SETUPTOOLS_SCM_PRETEND_VERSION_FOR_<NAME> env var. */
  pythonDist?: string
}

let _config: ShellConfig | null = null

export function configureShell(config: ShellConfig): void {
  _config = { pythonDist: config.pythonModule, ...config }
}

export function shellConfig(): ShellConfig {
  if (_config === null) {
    throw new Error(
      'configureShell() must be called before any @de/shell-main API. ' +
      'Call it at the top of your main process entry point.',
    )
  }
  return _config
}

/** `<appId>:<name>` — the IPC channel namespace. */
export function channel(name: string): string {
  return `${shellConfig().appId}:${name}`
}

/** `<APPID>` — the env-var namespace (uppercased, non-alphanumerics to `_`). */
export function envPrefix(): string {
  return shellConfig().appId.toUpperCase().replace(/[^A-Z0-9]/g, '_')
}
