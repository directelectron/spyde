/**
 * first_run.spec.ts — Phase 4 of the docs overhaul: the first-run welcome
 * walkthrough.
 *
 * On a genuine first launch (no `tutorial_seen` key in ~/.spyde/settings.json)
 * the app should auto-open the "welcome" guide once the backend is ready, which
 * in turn auto-loads the tiny `navigation` tutorial dataset (2 subwindows) —
 * see guides/welcome.ts. Opening the tour immediately persists tutorial_seen
 * (session.mark_tutorial_seen, spyde/backend/session.py), so a SECOND launch
 * with the same settings must NOT auto-open it again.
 *
 * SETTINGS ISOLATION: the harness (_harness.cjs) has no built-in settings-dir
 * isolation, and redirecting the whole-process USERPROFILE/HOME so Electron
 * itself launches under a scratch profile does NOT work — Electron/Chromium
 * refuses to start without a real user profile dir (`electron.launch: Process
 * failed to launch!`). So Session honors a dedicated SPYDE_SETTINGS_DIR env
 * override (spyde/backend/session.py) that only redirects settings.json, and
 * launchApp({env}) threads it through (_electron.launch({env}) → runner.ts
 * spreads process.env to the Python subprocess) — isolating ~/.spyde/settings.json
 * from the real one without touching Electron's own profile. This gives us a
 * REAL clean-then-dirty two-launch simulation (not just a forced-flag substitute).
 *
 * NO pre-kill (the user runs their own SpyDE on this box; Playwright manages
 * its own instances).
 */
import { test, expect } from '@playwright/test'
import { mkdirSync, mkdtempSync, rmSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
const { launchApp, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'first_run_shots')

test.describe('first-run welcome walkthrough', () => {
  test('genuine first launch auto-opens the welcome tour + autoloads the tutorial dataset', async () => {
    mkdirSync(SHOTS, { recursive: true })
    const home = mkdtempSync(join(tmpdir(), 'spyde-first-run-'))
    const settingsDir = join(home, '.spyde')
    const ctx = await launchApp({ dask: true, env: { SPYDE_SETTINGS_DIR: settingsDir } })
    const { page } = ctx
    try {
      // Settle so the backend-ready stdin pump is live (same guard the
      // tutorial/lazy specs use) before FirstRunGate's get_first_run fires.
      await page.waitForTimeout(1500)

      // The welcome tour should auto-open — no manual Help click.
      await expect(page.getByTestId('tour-overlay')).toBeVisible({ timeout: 30_000 })
      const bubble = page.getByTestId('tour-bubble')
      await expect(bubble).toBeVisible()
      await expect(bubble).toContainText('Welcome to SpyDE')
      await page.screenshot({ path: join(SHOTS, '01-first-launch-welcome-tour.png') })

      // Its autoload loads the navigation tutorial dataset -> 2 subwindows.
      await waitForSubwindowCount(page, 2, 60_000)
      await expect(page.getByTestId('tour-autoload-loading')).toHaveCount(0, { timeout: 60_000 })
      await expect(page.getByTestId('tour-autoload-error')).toHaveCount(0)
      await page.screenshot({ path: join(SHOTS, '02-welcome-tour-dataset-loaded.png') })

      ctx.assertNoJsErrors()
    } finally {
      await ctx.app.close()
      rmSync(home, { recursive: true, force: true })
    }
  })

  test('second launch with the same settings dir does NOT auto-open the tour', async () => {
    const home = mkdtempSync(join(tmpdir(), 'spyde-first-run-'))
    const settingsDir = join(home, '.spyde')
    try {
      // --- first launch: consumes the first-run flag ---------------------------
      const ctx1 = await launchApp({ dask: true, env: { SPYDE_SETTINGS_DIR: settingsDir } })
      try {
        await ctx1.page.waitForTimeout(1500)
        await expect(ctx1.page.getByTestId('tour-overlay')).toBeVisible({ timeout: 30_000 })
        // Give mark_tutorial_seen's fire-and-forget write time to land before we
        // close (it's sent the instant the tour opens — see FirstRunGate.tsx).
        await ctx1.page.waitForTimeout(1000)
      } finally {
        await ctx1.app.close()
      }

      // --- second launch: same settings dir, flag now persisted ----------------
      const ctx2 = await launchApp({ dask: true, env: { SPYDE_SETTINGS_DIR: settingsDir } })
      try {
        await ctx2.page.waitForTimeout(3000)
        await expect(ctx2.page.getByTestId('tour-overlay')).toHaveCount(0)
        await ctx2.page.screenshot({ path: join(SHOTS, '03-second-launch-no-auto-tour.png') })

        // Still re-launchable manually from Help.
        await ctx2.page.getByTestId('help-button').click()
        await expect(ctx2.page.getByTestId('help-menu')).toBeVisible()
        await expect(ctx2.page.getByTestId('help-guide-welcome')).toBeVisible()
        await ctx2.page.getByTestId('help-guide-welcome').click()
        await expect(ctx2.page.getByTestId('tour-overlay')).toBeVisible()
        await expect(ctx2.page.getByTestId('tour-bubble')).toContainText('Welcome to SpyDE')
        await ctx2.page.screenshot({ path: join(SHOTS, '04-second-launch-manual-relaunch.png') })

        ctx2.assertNoJsErrors()
      } finally {
        await ctx2.app.close()
      }
    } finally {
      rmSync(home, { recursive: true, force: true })
    }
  })
})
