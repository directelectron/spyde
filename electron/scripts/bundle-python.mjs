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
import { cpSync, existsSync, mkdirSync, rmSync, copyFileSync, chmodSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { execSync } from 'child_process'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = join(__dirname, '..', '..')          // …/spyde
const outDir = join(__dirname, '..', 'resources', 'python')
const isWin = process.platform === 'win32'
const uvName = isWin ? 'uv.exe' : 'uv'

function log(msg) { console.log(`[bundle-python] ${msg}`) }

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
  filter: (src) =>
    !src.includes(`${'spyde'}/tests`) &&
    !src.includes('__pycache__') &&
    !src.endsWith('.pyc'),
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

log(`done → ${outDir}`)
