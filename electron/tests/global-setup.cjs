/**
 * global-setup.cjs — settings isolation for EVERY spec, not just harness users.
 *
 * Two things bite only on a machine that has never RUN SpyDE, i.e. every CI
 * runner, and neither is visible on a dev box:
 *
 *  1. `tutorial_seen` is unset, so FirstRunGate auto-opens the welcome tour —
 *     which loads a tutorial dataset and floats a callout bubble over the UI a
 *     spec is trying to drive. (It no longer swallows every pointer event: the
 *     overlay is `pointerEvents:none` apart from the bubble, see Tour.tsx. But
 *     the bubble is still a real hit-target, and the auto-loaded dataset still
 *     adds subwindows.) This is how action_scoping and center_zero_beam failed
 *     on CI while passing locally.
 *
 *  2. `shell.openPath` shells out to xdg-open, which has no file manager to
 *     reach and leaves the app unable to exit (examples_menu's afterAll timed
 *     out for 120s on app.close()).
 *
 * _harness.cjs already handles both per launch, but only ~half the suite uses
 * it — 30 specs call `_electron.launch()` directly. They all spread
 * `...process.env`, so setting these here is what actually reaches all of them.
 * A spec that wants the real behaviour (first_run.spec.ts wants a genuine first
 * launch) passes its own SPYDE_SETTINGS_DIR, which still wins.
 *
 * SPYDE_SETTINGS_DIR redirects settings.json ONLY — never Electron's own
 * profile, which Chromium refuses to launch without.
 */
const { mkdtempSync, mkdirSync, writeFileSync } = require('fs')
const { tmpdir } = require('os')
const { join } = require('path')

module.exports = async () => {
  // One run-scoped root for every per-launch scratch dir (_clean_slate.cjs
  // creates them here), so global-teardown can remove the whole lot instead of
  // leaving a profile per launch behind in the system temp.
  const root = mkdtempSync(join(tmpdir(), 'spyde-e2e-'))
  process.env.SPYDE_E2E_TMP_ROOT = root

  const dir = join(root, 'shared', '.spyde')
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'settings.json'), JSON.stringify({ tutorial_seen: true }))
  // This stays as the FALL-BACK only. _clean_slate.cjs replaces it with a fresh
  // dir per launch; it recognises the inherited default by comparing against
  // SPYDE_E2E_SHARED_SETTINGS_DIR below. Keeping it set means that if the patch
  // ever fails to install, the suite degrades to the old shared-dir behaviour
  // rather than to the developer's real ~/.spyde (which would auto-open the
  // welcome tour and break dozens of specs in a way that looks like anything
  // but a settings problem).
  process.env.SPYDE_SETTINGS_DIR = dir
  process.env.SPYDE_E2E_SHARED_SETTINGS_DIR = dir
  process.env.SPYDE_NO_SHELL_OPEN = '1'
}
