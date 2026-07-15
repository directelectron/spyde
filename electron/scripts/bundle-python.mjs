/**
 * bundle-python.mjs — stage the Python sidecar payload for electron-builder.
 *
 * Copies the project (pyproject.toml, uv.lock, the spyde package) and the
 * platform `uv` binary into electron/resources/python. electron-builder then
 * ships that dir as an extraResource (-> <app>/resources/python at runtime),
 * where pythonEnv.ts runs `uv sync` into the user's writable data dir on first
 * launch (DISTRIBUTION_PLAN.md "Option A" — tiny installer, GPU wheel by uv).
 *
 * The venv / torch are NOT bundled; only the lock + source + uv. Run before the
 * electron-builder step (npm run dist does this).
 *
 * uv binary: prefers a vendored copy at electron/vendor/uv/<platform>/uv[.exe];
 * otherwise copies the `uv` found on PATH. CI should populate vendor/ with the
 * pinned uv for each target OS.
 */
import { cpSync, existsSync, mkdirSync, rmSync, copyFileSync, chmodSync, readdirSync, readFileSync, writeFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { execSync } from 'child_process'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = join(__dirname, '..', '..')          // …/spyde
const pkgPath = join(__dirname, '..', 'package.json') // …/spyde/electron/package.json
const outDir = join(__dirname, '..', 'resources', 'python')
const isWin = process.platform === 'win32'
const uvName = isWin ? 'uv.exe' : 'uv'

function log(msg) { console.log(`[bundle-python] ${msg}`) }

/**
 * Resolve the spyde package version to bake into the staged payload.
 *
 * The bundled source tree has NO `.git`, and spyde's pyproject.toml derives its
 * version dynamically via setuptools_scm. Without git history, the first-launch
 * `uv sync` would raise `LookupError: unable to detect version` and exit 1 (the
 * packaged-app first-run crash). So we resolve the version HERE (in CI, git +
 * tags are present) and write a `.spyde-version` marker; pythonEnv.ts reads it
 * and exports SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SPYDE so the build succeeds.
 *
 * Prefer electron/package.json's `version` — release.yml verifies it equals the
 * pushed git tag (e.g. "0.1.0"), so it's the authoritative, already-validated
 * release version and a plain X.Y.Z is a valid PEP 440 version. Fall back to
 * setuptools_scm only if package.json is unreadable.
 */
function resolveSpydeVersion() {
  try {
    const v = JSON.parse(readFileSync(pkgPath, 'utf8')).version
    if (typeof v === 'string' && v.trim()) return v.trim()
  } catch { /* fall through to setuptools_scm */ }
  try {
    // no-guess-dev (per pyproject) — e.g. "0.1.0" on a tag, "0.1.1.dev3+g<sha>"
    // between tags. A PEP 440 dev version is still a valid pretend version.
    const v = execSync('uv run python -m setuptools_scm', {
      cwd: repoRoot, stdio: ['ignore', 'pipe', 'ignore'],
    }).toString().trim()
    if (v) return v
  } catch { /* no git / no uv — leave marker unwritten */ }
  return null
}

// Fresh staging dir.
rmSync(outDir, { recursive: true, force: true })
mkdirSync(outDir, { recursive: true })

// 1. Project metadata + lock (reproducible `uv sync`).
for (const f of ['pyproject.toml', 'uv.lock']) {
  const src = join(repoRoot, f)
  if (!existsSync(src)) throw new Error(`missing ${f} at repo root`)
  copyFileSync(src, join(outDir, f))
  log(`staged ${f}`)
}

// Pin the interpreter for the PACKAGED env only. WITHOUT this, first-launch
// `uv sync` picks the newest CPython (it grabbed 3.14), which has no CUDA torch
// wheels → setup fails. We write `.python-version` ONLY into the payload, never
// at the repo root: a root file would also hijack dev + the CI test matrix
// (every `uv run` re-creates the venv at this version, so `uv sync --python
// 3.10` then `uv run pytest` silently runs 3.12). Scoping it to the staged tree
// keeps 3.12 for shipped users while dev/CI stay free to pick their own Python.
const PAYLOAD_PYTHON = '3.12'
writeFileSync(join(outDir, '.python-version'), PAYLOAD_PYTHON + '\n', 'utf8')
log(`staged .python-version = ${PAYLOAD_PYTHON}`)

// 2. A PRE-BUILT spyde wheel — NOT the source tree.
//    Building `spyde` from source on first launch is impossible when the app is
//    installed read-only (e.g. C:\Program Files): setuptools' build_wheel runs
//    egg_info, which writes `spyde.egg-info` INTO the source dir → "could not
//    create 'spyde.egg-info': Access is denied", and `uv sync` exits 1. (This
//    bit rc.2 AND rc.3; --no-editable does NOT help — a wheel build still writes
//    egg_info into the tree.) So we build the wheel HERE, where the repo is
//    writable, and ship the .whl. pythonEnv.ts installs it with
//    `uv pip install --no-deps` after syncing the deps with --no-install-project,
//    so NOTHING is built on the user's machine. spyde is pure-Python → one
//    py3-none-any wheel works on every platform + Python we ship.
const wheelsDir = join(outDir, 'wheels')
mkdirSync(wheelsDir, { recursive: true })
// setuptools_scm can't see a version from the (git-less at build? no — CI has
// git) tree reliably across environments; pin the already-validated release
// version so the wheel's metadata matches the tag / package.json.
const wheelVersion = resolveSpydeVersion()
const scmEnv = wheelVersion
  ? { ...process.env, SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SPYDE: wheelVersion,
      SETUPTOOLS_SCM_PRETEND_VERSION: wheelVersion }
  : process.env
execSync(`uv build --wheel --out-dir "${wheelsDir}"`, {
  cwd: repoRoot, stdio: 'inherit', env: scmEnv,
})
const builtWheel = readdirSync(wheelsDir).find((f) => f.endsWith('.whl'))
if (!builtWheel) throw new Error('uv build produced no spyde wheel')
log(`staged spyde wheel: ${builtWheel}`)

// 3. The uv binary.
const vendored = join(__dirname, '..', 'vendor', 'uv', process.platform, uvName)
let uvSrc = vendored
if (!existsSync(vendored)) {
  try {
    const onPath = execSync(isWin ? 'where uv' : 'command -v uv').toString().trim().split('\n')[0]
    uvSrc = onPath
    log(`vendor/ uv not found; using uv from PATH: ${onPath}`)
  } catch {
    throw new Error(
      'no uv binary: add electron/vendor/uv/<platform>/uv or install uv on PATH')
  }
}
const uvDst = join(outDir, uvName)
copyFileSync(uvSrc, uvDst)
if (!isWin) chmodSync(uvDst, 0o755)
log(`staged ${uvName}`)

// 3b. Vendored git (portable). Some dependencies are `git+https://…` sources
//     (the cssfrancis/hyperspy fork), which `uv sync` resolves by shelling out
//     to `git clone`. A clean user machine often has NO git on PATH → "Git
//     operation failed" and first-launch setup dies. So ship a portable git in
//     the payload (like uv above); pythonEnv.ts prepends its bin dir to PATH for
//     the sync. Staged from electron/vendor/git/<platform>/ (a MinGit tree on
//     Windows). Optional in DEV — if absent we log and skip, and the sync falls
//     back to any git on PATH (the dev box has one). CI MUST populate vendor/git
//     for a shippable installer.
const vendoredGit = join(__dirname, '..', 'vendor', 'git', process.platform)
if (existsSync(vendoredGit)) {
  const gitDst = join(outDir, 'git')
  cpSync(vendoredGit, gitDst, { recursive: true })
  // Make the git executables runnable on posix (Windows ignores the mode).
  if (!isWin) {
    for (const sub of ['bin', 'libexec/git-core']) {
      const d = join(gitDst, sub)
      if (existsSync(d)) {
        for (const f of readdirSync(d)) {
          try { chmodSync(join(d, f), 0o755) } catch { /* best-effort */ }
        }
      }
    }
  }
  log('staged vendored git')
} else {
  log(`vendor/git/${process.platform} not found; NOT staging git — first-launch ` +
      '`uv sync` will need a git on the user\'s PATH for git+https deps. ' +
      'CI must populate electron/vendor/git/<platform>/ for a shippable build.')
}

// 4. Version marker: the staged tree has no `.git`, so setuptools_scm can't
//    detect a version at first-launch `uv sync` time (-> LookupError, exit 1).
//    Write the version resolved HERE (git present) so pythonEnv.ts can export
//    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SPYDE and the sync succeeds.
const spydeVersion = resolveSpydeVersion()
if (spydeVersion) {
  const markerPath = join(outDir, '.spyde-version')
  writeFileSync(markerPath, spydeVersion, 'utf8')
  log(`staged .spyde-version = ${spydeVersion}`)
} else {
  log('WARNING: could not resolve spyde version — .spyde-version NOT written; ' +
      'first-launch `uv sync` may fail with a setuptools_scm LookupError')
}

log(`done → ${outDir}`)
