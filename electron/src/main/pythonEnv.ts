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
 * torch is resolved PER MACHINE (win32/linux): the lock pins torch to the cu124
 * index (the dev box's backend), which would force the cu124 wheel onto every
 * user. Instead the first-run install is two-step:
 *   1. `uv sync --frozen --no-dev --no-install-package torch`  (lock-exact for
 *      everything EXCEPT torch)
 *   2. `uv pip install torch==<locked release> --torch-backend=auto
 *       --python <env python>`  (uv probes the machine's driver and picks the
 *      matching CUDA / CPU wheel — verified: `--python` is required, uv pip
 *      IGNORES UV_PROJECT_ENVIRONMENT; needs uv >= 0.5, the staged uv is 0.10.x)
 * Any step-2 failure falls back to the plain full `uv sync` (the old cu124
 * path), so the working install path can never regress. macOS keeps the plain
 * sync (no CUDA pin there; PyPI torch already carries MPS).
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

export function venvPython(envDir: string): string {
  return isWin ? join(envDir, 'Scripts', 'python.exe') : join(envDir, 'bin', 'python')
}

function uvBinaryName(): string {
  return isWin ? 'uv.exe' : 'uv'
}

/** Paths of the bundled project + managed env, and whether each exists — the
 *  single "is there a managed environment?" answer, shared by the first-run
 *  setup and the GPU-triage Fix handler (index.ts). In dev there is no bundled
 *  payload, so `bundled` is false and the triage UI goes report-only. */
export function managedEnvPaths(resourcesPath: string, userData: string): {
  projectDir: string
  envDir: string
  pythonExe: string
  bundled: boolean
  envExists: boolean
} {
  const projectDir = join(resourcesPath, 'python')
  const envDir = join(userData, 'python-env')
  const pythonExe = venvPython(envDir)
  return {
    projectDir,
    envDir,
    pythonExe,
    bundled: existsSync(join(projectDir, 'uv.lock')),
    envExists: existsSync(pythonExe),
  }
}

/**
 * The locked torch release for THIS platform, parsed from `<projectDir>/uv.lock`.
 *
 * The lock carries TWO torch entries (platform-conditional sources): the
 * `download.pytorch.org/whl/cu124` one (win32 per pyproject's source marker,
 * e.g. "2.6.0+cu124") and the PyPI one (everything else, e.g. "2.13.0"). Pick
 * the entry whose source matches this platform and strip any `+cuXXX` local
 * tag — `--torch-backend=auto` owns the backend variant; we only pin the
 * release so the resolved wheel stays lock-adjacent. torch is the ONLY
 * torch-family package in the lock (no torchvision/torchaudio), so one
 * `--no-install-package torch` + one pip install covers the family. Returns
 * null when the lock is missing/unparseable (caller uses the plain full sync).
 */
export function readLockedTorchVersion(projectDir: string): string | null {
  const lock = readSafe(join(projectDir, 'uv.lock'))
  if (!lock) return null
  // Each entry: [[package]] \n name = "torch" \n version = "…" \n source = { registry = "…" }
  const re = /\[\[package\]\]\s*\r?\nname = "torch"\s*\r?\nversion = "([^"]+)"\s*\r?\nsource = \{ registry = "([^"]+)" \}/g
  const entries: Array<{ version: string; registry: string }> = []
  for (let m = re.exec(lock); m; m = re.exec(lock)) {
    entries.push({ version: m[1], registry: m[2] })
  }
  if (!entries.length) return null
  const preferPytorchIndex = isWin   // pyproject pins the cu124 index for win32 only
  const pick =
    entries.find((e) => preferPytorchIndex === e.registry.includes('download.pytorch.org')) ??
    entries[0]
  return pick.version.split('+')[0]   // "2.6.0+cu124" → "2.6.0"
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
    await setupEnv(bundledProject, envDir, readSpydeVersion(bundledProject), opts.onProgress)
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
 * The version bundle-python.mjs baked into `<projectDir>/.spyde-version` (X.Y.Z,
 * resolved at CI bundle time when git history exists). Empty string in dev (no
 * marker) — the dev repo has `.git`, so setuptools_scm works normally there.
 */
function readSpydeVersion(projectDir: string): string {
  return readSafe(join(projectDir, '.spyde-version'))
}

/**
 * Build the env for a uv invocation against the managed env.
 *
 * `spydeVersion` (from the .spyde-version marker) is exported as
 * SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SPYDE so building the editable `spyde`
 * package succeeds even though the staged tree has NO `.git` — without it
 * setuptools_scm raises `LookupError: unable to detect version` and `uv sync`
 * exits 1 (the first-launch crash). Absent in dev (marker not present) → the
 * env is unchanged and setuptools_scm resolves from the repo's git history.
 */
function uvEnv(envDir: string, spydeVersion: string): NodeJS.ProcessEnv {
  // Pretend-version env for setuptools_scm. The dist-name–scoped var
  // (…_FOR_SPYDE — setuptools_scm normalises "spyde" to the env suffix "SPYDE")
  // is the precise one; the plain SETUPTOOLS_SCM_PRETEND_VERSION is a harmless
  // belt-and-braces fallback in case the dist name ever changes.
  const scmEnv: Record<string, string> = spydeVersion
    ? {
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SPYDE: spydeVersion,
        SETUPTOOLS_SCM_PRETEND_VERSION: spydeVersion,
      }
    : {}
  return {
    ...process.env,
    ...scmEnv,
    // UV_PROJECT_ENVIRONMENT redirects `uv sync`'s venv out of the read-only
    // resources into the writable user dir. (Ignored by `uv pip install`,
    // which targets via --python — verified; harmless to set for both.)
    UV_PROJECT_ENVIRONMENT: envDir,
    // Keep uv's own cache/tools next to the env so an air-gapped re-run is
    // self-contained and we never write into the read-only bundle.
    UV_CACHE_DIR: join(envDir, '..', 'uv-cache'),
  }
}

/** Spawn the bundled (or PATH) uv with `args`, streaming output to onProgress. */
function runUv(
  projectDir: string,
  envDir: string,
  args: string[],
  spydeVersion: string,
  onProgress?: (line: string) => void,
): Promise<void> {
  const uv = join(projectDir, uvBinaryName())
  const uvCmd = existsSync(uv) ? uv : 'uv'   // fall back to PATH if not bundled
  return new Promise((resolve, reject) => {
    const proc = spawn(uvCmd, args, {
      cwd: projectDir,
      env: uvEnv(envDir, spydeVersion),
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    const relay = (b: Buffer) => onProgress?.(b.toString())
    proc.stdout?.on('data', relay)
    proc.stderr?.on('data', relay)   // uv prints progress to stderr
    proc.on('error', reject)
    proc.on('close', (code) =>
      code === 0 ? resolve() : reject(new Error(`uv ${args[0]} exited with code ${code}`)),
    )
  })
}

/**
 * Install the locked torch release into the managed env with the backend
 * resolved for THIS machine (`--torch-backend=auto`: uv probes the NVIDIA
 * driver and picks the matching cuXXX wheel, or the much smaller CPU wheel when
 * no GPU is present). Shared by first-run setup (step 2) and the GPU-triage
 * "Fix PyTorch install" handler. Throws on failure (caller decides fallback).
 */
export function installTorchPerMachine(
  projectDir: string,
  envDir: string,
  onProgress?: (line: string) => void,
): Promise<void> {
  const torchVersion = readLockedTorchVersion(projectDir)
  if (!torchVersion) {
    return Promise.reject(new Error('could not read the locked torch version from uv.lock'))
  }
  onProgress?.(`[env-setup] installing torch==${torchVersion} with --torch-backend=auto\n`)
  return runUv(
    projectDir, envDir,
    ['pip', 'install', `torch==${torchVersion}`, '--torch-backend=auto',
     '--python', venvPython(envDir)],
    '',   // no scm pretend-version needed for a plain pip install
    onProgress,
  )
}

/**
 * Create/refresh the managed env. On win32/linux (the platforms where the lock
 * pins CUDA torch) this is the two-step per-machine install; on failure — or on
 * macOS, or when the lock has no readable torch pin — the plain full
 * `uv sync --frozen --no-dev` (the original, known-good path) runs instead.
 * Logs which path ran to the progress stream.
 */
async function setupEnv(
  projectDir: string,
  envDir: string,
  spydeVersion: string,
  onProgress?: (line: string) => void,
): Promise<void> {
  const twoStep =
    (process.platform === 'win32' || process.platform === 'linux') &&
    readLockedTorchVersion(projectDir) !== null

  if (twoStep) {
    try {
      onProgress?.('[env-setup] two-step install: uv sync (lock-exact, torch deferred) '
        + 'then torch via --torch-backend=auto\n')
      // Step 1: everything except torch, exactly as locked. torch is the only
      // torch-family package in the lock, so one exclusion covers it.
      await runUv(
        projectDir, envDir,
        ['sync', '--frozen', '--no-dev', '--no-install-package', 'torch'],
        spydeVersion, onProgress,
      )
      // Step 2: torch resolved for this machine.
      await installTorchPerMachine(projectDir, envDir, onProgress)
      onProgress?.('[env-setup] per-machine torch install complete\n')
      return
    } catch (err) {
      onProgress?.(`[env-setup] per-machine torch install failed (${(err as Error)?.message ?? err}); `
        + 'falling back to the full locked sync\n')
      // fall through to the plain sync
    }
  }

  onProgress?.('[env-setup] running full locked uv sync\n')
  await runUv(projectDir, envDir, ['sync', '--frozen', '--no-dev'], spydeVersion, onProgress)
}
