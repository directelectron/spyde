// shell-link.mjs — link the shell's TypeScript into this project.
//
// The shell ships its TypeScript INSIDE the de-shell wheel (de_shell/js: one
// folder per Electron target), so the JavaScript that speaks the sidecar
// protocol and the Python that speaks it move together — one `pip install -U`
// upgrades both, and two versions in one app cannot happen. The bundler and
// tsc need a fixed path, though, and site-packages is not one: it moves with
// the platform and the Python version, and an editable install points at a
// checkout instead. So this makes `electron/shell` a junction (a symlink off
// Windows) to wherever the installed package's `de_shell/js` is, and every
// alias and tsconfig path goes through it.
//
// Run by npm's postinstall, by `npm run shell:link` after a `uv sync` that
// changed the shell, and by electron.vite.config.ts on every build, which is
// what keeps it honest: a stale link is re-pointed, never trusted.
import { execFileSync } from 'node:child_process'
import {
  existsSync, lstatSync, realpathSync, rmdirSync, rmSync, symlinkSync, unlinkSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const DEFAULT_ELECTRON_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..')

/** The project's Python: the venv's own interpreter when it exists (fast, and
 *  no uv needed), else `uv run` from the locked environment. */
function pythonCmd(projectDir) {
  const venv = process.platform === 'win32'
    ? join(projectDir, '.venv', 'Scripts', 'python.exe')
    : join(projectDir, '.venv', 'bin', 'python')
  return existsSync(venv) ? [venv] : ['uv', 'run', '--frozen', 'python']
}

/**
 * Where the installed de-shell keeps its TypeScript, by asking it — or, when
 * there is no Python environment to ask (a TypeScript-only CI job), the
 * sibling checkout `../de-shell` that the uv path source points at anyway.
 */
export function shellJsDir(electronDir = DEFAULT_ELECTRON_DIR) {
  const projectDir = resolve(electronDir, '..')
  const [cmd, ...args] = pythonCmd(projectDir)
  let out = ''
  let why = ''
  try {
    out = execFileSync(cmd, [...args, '-m', 'de_shell.js'],
                       { cwd: projectDir, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
  } catch (err) {
    why = `${cmd}: ${err.message.split('\n')[0]}`
  }
  const dir = out.trim().split(/\r?\n/).pop()
  if (dir && existsSync(join(dir, 'main', 'index.ts'))) return dir
  const sibling = resolve(projectDir, '..', 'de-shell', 'de_shell', 'js')
  if (existsSync(join(sibling, 'main', 'index.ts'))) return sibling
  throw new Error(
    (why ? `could not ask the project's Python where de-shell is (${why})`
         : `de_shell.js answered ${JSON.stringify(dir)}, which has no main/index.ts`) +
    `, and there is no checkout at ${sibling}. Run \`uv sync --extra tests\` first.`)
}

/** Remove whatever sits at the link path without ever following it: a
 *  junction is unlinked (rmdir on the platforms that insist), a real directory
 *  someone copied into place is removed for real. */
function clearLink(link) {
  let st
  try { st = lstatSync(link) } catch { return }
  if (st.isSymbolicLink()) {
    try { unlinkSync(link) } catch { rmdirSync(link) }
  } else if (st.isDirectory()) {
    rmSync(link, { recursive: true, force: true })
  } else {
    unlinkSync(link)
  }
}

/**
 * Make `<electronDir>/shell` point at the installed shell's TypeScript, and
 * return the link path. Idempotent: an up-to-date link is left alone.
 */
export function ensureShellLink(electronDir = DEFAULT_ELECTRON_DIR, { quiet = false } = {}) {
  const link = join(electronDir, 'shell')
  const target = realpathSync(shellJsDir(electronDir))
  let current = null
  try {
    if (lstatSync(link).isSymbolicLink()) current = realpathSync(link)
  } catch { /* nothing there yet */ }
  if (current === target) return link
  clearLink(link)
  symlinkSync(target, link, process.platform === 'win32' ? 'junction' : 'dir')
  if (!quiet) console.log(`[shell-link] electron/shell -> ${target}`)
  return link
}

// As a script: `node electron/scripts/shell-link.mjs [--soft]`. `--soft` (what
// postinstall passes) warns instead of failing when the Python environment is
// not there yet — an `npm install` before the first `uv sync` is a normal order
// of events, and the vite config re-tries with a hard error at build time.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    ensureShellLink()
  } catch (err) {
    if (!process.argv.includes('--soft')) throw err
    console.warn(`[shell-link] not linked yet: ${err.message}`)
  }
}
