/**
 * _clean_slate.cjs — a FRESH app profile for every Electron launch in the suite.
 *
 * The problem this fixes
 * ---------------------
 * Two kinds of state used to survive from one spec into the next, for the whole
 * length of a shard:
 *
 *  1. **Electron's Chromium profile** (`userData`) — and with it localStorage,
 *     sessionStorage and IndexedDB. NOTHING isolated this: every one of the ~60
 *     spec files in a shard launched into the SAME profile. The renderer
 *     persists real state there — per-cell report image widths
 *     (`ReportImageCell`), console command history (`ConsoleBar`), CIF recents
 *     (`CifRecents`) — so what a spec saw depended on which specs had run
 *     before it.
 *
 *  2. **settings.json** — `global-setup.cjs` minted ONE scratch `.spyde` for the
 *     entire run and exported it as `SPYDE_SETTINGS_DIR`. The ~30 specs that
 *     call `_electron.launch()` directly (rather than `launchApp`) all spread
 *     `...process.env`, so they shared that one file and accumulated each
 *     other's `recent_files`. (`_harness.cjs` already minted a fresh dir per
 *     launch, so harness users were fine.)
 *
 * Sharing state across tests makes them order-dependent, which is what turns a
 * re-shard or a re-order into a pile of unexplainable failures: the test that
 * fails is not the test that is broken.
 *
 * How it works
 * ------------
 * `playwright.config.ts` is evaluated in EVERY worker process (verified: the
 * config's top level runs with `TEST_WORKER_INDEX` set), so wrapping
 * `_electron.launch` there reaches every spec — harness users and direct
 * launchers alike — with no per-spec edits, and covers specs written later
 * without anyone having to remember this.
 *
 * Each launch gets:
 *   - `--user-data-dir=<fresh>`  → clean localStorage/IndexedDB/cache
 *   - `SPYDE_SETTINGS_DIR=<fresh>` seeded with `tutorial_seen: true`
 *
 * Opting out
 * ----------
 * A spec that means to test persistence ACROSS launches passes its own
 * `SPYDE_SETTINGS_DIR` (first_run.spec.ts launches twice with one dir on
 * purpose) or its own `--user-data-dir`; both are honoured untouched.
 *
 * Distinguishing "the caller asked for this dir" from "the caller merely
 * inherited it via ...process.env" is why `global-setup.cjs` also exports
 * `SPYDE_E2E_SHARED_SETTINGS_DIR`: a value equal to that one is the inherited
 * default and gets replaced; anything else is a deliberate choice and is kept.
 *
 * FAIL-SAFE: global-setup still sets the shared `SPYDE_SETTINGS_DIR`. If this
 * patch ever fails to install, every launch falls back to exactly the old
 * behaviour (one shared scratch dir) — never to the developer's real ~/.spyde,
 * which would silently re-open the welcome tour and break dozens of specs.
 */
const { _electron } = require('@playwright/test')
const { mkdtempSync, mkdirSync, writeFileSync } = require('fs')
const { tmpdir } = require('os')
const { join } = require('path')

/** Where per-launch scratch dirs live. global-setup points this at one
 *  run-scoped root so global-teardown can delete the lot in one go; without it
 *  (a spec run directly, no globalSetup) fall back to the system temp. */
function scratchRoot() {
  return process.env.SPYDE_E2E_TMP_ROOT || tmpdir()
}

function freshProfileDir() {
  return mkdtempSync(join(scratchRoot(), 'profile-'))
}

function freshSettingsDir() {
  const dir = join(mkdtempSync(join(scratchRoot(), 'settings-')), '.spyde')
  mkdirSync(dir, { recursive: true })
  // Same seed global-setup used: a runner that has never run SpyDE would
  // otherwise auto-open the welcome tour over whatever the spec is driving.
  writeFileSync(join(dir, 'settings.json'), JSON.stringify({ tutorial_seen: true }))
  return dir
}

let installed = false

function installCleanSlate() {
  if (installed) return
  installed = true

  const orig = _electron.launch.bind(_electron)

  _electron.launch = function cleanSlateLaunch(opts = {}) {
    const args = Array.isArray(opts.args) ? opts.args.slice() : []
    // Respect an explicit profile (a spec testing profile persistence).
    if (!args.some((a) => String(a).startsWith('--user-data-dir'))) {
      args.push(`--user-data-dir=${freshProfileDir()}`)
    }

    // `opts.env` undefined means "inherit process.env"; preserve that meaning
    // while still being able to override the one key we care about.
    const env = { ...(opts.env || process.env) }
    const inherited = process.env.SPYDE_E2E_SHARED_SETTINGS_DIR
    if (!env.SPYDE_SETTINGS_DIR ||
        (inherited && env.SPYDE_SETTINGS_DIR === inherited)) {
      env.SPYDE_SETTINGS_DIR = freshSettingsDir()
    }

    return orig({ ...opts, args, env })
  }
}

module.exports = { installCleanSlate, freshSettingsDir, freshProfileDir }
