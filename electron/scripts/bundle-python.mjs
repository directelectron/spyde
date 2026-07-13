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
import { cpSync, existsSync, mkdirSync, rmSync, copyFileSync, chmodSync, readFileSync, writeFileSync } from 'fs'
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

// 2. The spyde source package (installed into the venv by uv sync). Exclude the
//    test suite and caches to keep the payload small.
cpSync(join(repoRoot, 'spyde'), join(outDir, 'spyde'), {
  recursive: true,
  filter: (src) => {
    // Normalise separators: cpSync hands back backslash paths on Windows, so a
    // hard-coded "spyde/tests" check would silently fail to exclude the tests.
    const p = src.replaceAll('\\', '/')
    return (
      !p.includes('/spyde/tests') &&
      !p.includes('__pycache__') &&
      !p.endsWith('.pyc')
    )
  },
})
log('staged spyde/ source')

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
