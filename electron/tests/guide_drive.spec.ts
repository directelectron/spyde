/**
 * guide_drive.spec.ts — the in-app coachmark Tour (descriptive walkthrough).
 *
 * The tour is DESCRIPTIVE ONLY (the "Show me ▶" auto-drive was removed): it loads
 * its tutorial dataset ONCE on open, spotlights each step, and closes the dataset
 * again on exit. This spec verifies, in the RUNNING app:
 *   • opening the Find Vectors guide auto-loads exactly ONE dataset (a navigator +
 *     signal window — NOT a double/triple copy),
 *   • there is NO "Show me" button (purely Back/Next/Done), and
 *   • closing the tour tears the tutorial dataset back down (no lingering windows).
 *
 * Needs a real Dask client (the tutorial dataset loads through the backend) so it
 * runs on the `electron` project with `launchApp({dask:true})`. NO pre-kill (the
 * user runs their own SpyDE on this box; Playwright manages its instance).
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { join } from 'path'
const {
  launchApp, waitForSubwindowCount,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'guide_drive_shots')

test('Find Vectors tour loads ONE dataset, has no Show me, and closes it on exit', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const ctx = await launchApp({ dask: true })
  const { page } = ctx
  try {
    // Settle so backend-ready's stdin pump is live before the tour fires the
    // autoload action (same guard the tutorial/lazy specs use).
    await page.waitForTimeout(1500)

    const subwindows = page.getByTestId('subwindow')

    // --- open the Find Vectors guide from the Help "?" menu ------------------
    await page.getByTestId('help-button').click()
    await expect(page.getByTestId('help-menu')).toBeVisible()
    await page.getByTestId('help-guide-find-vectors').click()
    await expect(page.getByTestId('tour-overlay')).toBeVisible()
    const bubble = page.getByTestId('tour-bubble')
    await expect(bubble).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '01-tour-open.png') })

    // --- auto-load ran: EXACTLY the navigator + signal subwindows (one copy) --
    await waitForSubwindowCount(page, 2, 60_000)
    await expect(page.getByTestId('tour-autoload-loading')).toHaveCount(0, { timeout: 60_000 })
    await expect(page.getByTestId('tour-autoload-error')).toHaveCount(0)
    // Let any (buggy) extra loads settle, then assert we still have exactly 2 —
    // no double/triple-load stacking. (find-vectors = navigator + signal = 2.)
    await page.waitForTimeout(1500)
    expect(await subwindows.count()).toBe(2)
    await page.screenshot({ path: join(SHOTS, '02-one-dataset-2-windows.png') })

    // --- the tour is descriptive: NO "Show me" anywhere in the walkthrough ----
    await expect(page.getByTestId('tour-show-me')).toHaveCount(0)
    // Walk every step; the count must never appear and window count must not grow.
    for (let i = 0; i < 6; i++) {
      const next = page.getByTestId('tour-next')
      if (await next.count() === 0) break
      await next.click()
      await expect(page.getByTestId('tour-show-me')).toHaveCount(0)
    }
    expect(await subwindows.count()).toBe(2)
    await page.screenshot({ path: join(SHOTS, '03-stepped-through-no-showme.png') })

    // --- closing the tour (Done) tears the tutorial dataset back down ---------
    const done = page.getByTestId('tour-done')
    if (await done.count() > 0) await done.click()
    else await page.getByTestId('tour-close').click()
    await expect(page.getByTestId('tour-overlay')).toHaveCount(0)
    // tutorial_close_all fired → the dummy dataset's windows are gone.
    await expect(subwindows).toHaveCount(0, { timeout: 30_000 })
    await page.screenshot({ path: join(SHOTS, '04-closed-dataset-gone.png') })

    ctx.assertNoJsErrors()
  } finally {
    await ctx.app.close()
  }
})
