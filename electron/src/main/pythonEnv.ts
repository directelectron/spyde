/**
 * pythonEnv.ts — resolve (and on first run, create) the Python sidecar env.
 *
 * DISTRIBUTION_PLAN.md "Option A" (uv-managed): the installer ships a tiny
 * payload — the bundled `uv`, the project (`pyproject.toml` + `uv.lock`) and the
 * `spyde` source — under <resources>/python. On first launch we run
 * `uv sync` into a venv in the user's WRITABLE data dir (the app bundle is
 * read-only / code-signed), so the GPU-correct torch wheel is fetched per
 * machine and updates are a cheap incremental `uv sync`.
 *
 * In development (no bundled payload) we fall back to `uv run` from the repo
 * root — exactly the previous behaviour.
 */
import { spawn } from 'child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { createHash } from 'crypto'
import { join } from 'path'

export interface ResolvedPython {
  cmd: string[]      // argv to spawn, e.g. [pythonExe, '-m', 'spyde']
  cwd: string        // working directory for the spawn
}

export interface EnsureOptions {
  isPackaged: boolean
  resourcesPath: string   // process.resourcesPath (packaged) — holds <…>/python
  projectRoot: string     // repo root (dev) — holds pyproject.toml
  userData: string        // app.getPath('userData') — writable venv lives here
  onProgress?: (line: string) => void
}

const isWin = process.platform === 'win32'

function venvPython(envDir: string): string {
  return isWin ? join(envDir, 'Scripts', 'python.exe') : join(envDir, 'bin', 'python')
}

function uvBinaryName(): string {
  return isWin ? 'uv.exe' : 'uv'
}

/** Resolve the command to launch the SpyDE backend, creating the venv if needed. */
export async function resolvePythonEnv(opts: EnsureOptions): Promise<ResolvedPython> {
  const bundledProject = join(opts.resourcesPath, 'python')
  const bundledLock = join(bundledProject, 'uv.lock')

  // Development (or any build without the staged payload): use uv from PATH.
  if (!opts.isPackaged || !existsSync(bundledLock)) {
    return { cmd: ['uv', 'run', 'python', '-m', 'spyde'], cwd: opts.projectRoot }
  }

  const envDir = join(opts.userData, 'python-env')
  const pythonExe = venvPython(envDir)
  const stampFile = join(envDir, '.spyde-lock-hash')
  const lockHash = createHash('sha256').update(readFileSync(bundledLock)).digest('hex')

  // Skip the sync if the venv already matches the shipped lock.
  const upToDate =
    existsSync(pythonExe) &&
    existsSync(stampFile) &&
    readSafe(stampFile) === lockHash

  if (!upToDate) {
    await runUvSync(bundledProject, envDir, opts.onProgress)
    try {
      mkdirSync(envDir, { recursive: true })
      writeFileSync(stampFile, lockHash, 'utf8')
    } catch { /* a missing stamp just forces a re-sync next launch */ }
  }

  return { cmd: [pythonExe, '-m', 'spyde'], cwd: bundledProject }
}

function readSafe(p: string): string {
  try { return readFileSync(p, 'utf8').trim() } catch { return '' }
}

/**
 * `uv sync` the bundled project into `envDir`. UV_PROJECT_ENVIRONMENT redirects
 * the venv out of the read-only resources into the writable user dir;
 * --frozen installs the lock exactly (reproducible); torch-backend=auto (from
 * pyproject [tool.uv]) fetches the right GPU wheel.
 */
function runUvSync(
  projectDir: string,
  envDir: string,
  onProgress?: (line: string) => void,
): Promise<void> {
  const uv = join(projectDir, uvBinaryName())
  const uvCmd = existsSync(uv) ? uv : 'uv'   // fall back to PATH if not bundled
  return new Promise((resolve, reject) => {
    const proc = spawn(uvCmd, ['sync', '--frozen', '--no-dev'], {
      cwd: projectDir,
      env: {
        ...process.env,
        UV_PROJECT_ENVIRONMENT: envDir,
        // Keep uv's own cache/tools next to the env so an air-gapped re-run is
        // self-contained and we never write into the read-only bundle.
        UV_CACHE_DIR: join(envDir, '..', 'uv-cache'),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    const relay = (b: Buffer) => onProgress?.(b.toString())
    proc.stdout?.on('data', relay)
    proc.stderr?.on('data', relay)   // uv prints progress to stderr
    proc.on('error', reject)
    proc.on('close', (code) =>
      code === 0 ? resolve() : reject(new Error(`uv sync exited with code ${code}`)),
    )
  })
}
