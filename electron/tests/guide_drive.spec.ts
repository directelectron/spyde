/**
 * guide_drive.spec.ts — Phase 2: the in-app guide DRIVE engine.
 *
 * Verifies the coachmark Tour can auto-load its tutorial dataset and run a step
 * on demand ("Show me ▶") in the RUNNING app — the renderer-side port of the
 * `guide_screenshots.spec.ts` drive semantics (guideDriver.ts). We open the
 * Find Vectors guide from the Help "?" menu; its `autoload`
 * (`tutorial_load {name:'find_vectors'}`) must open a navigator + signal window,
 * and "Show me" on an autoDrive step (hover the toolbar → reveal, click Find
 * Diffraction Vectors → open the wizard) must perform the action and advance.
 *
 * Needs a real Dask client (the tutorial dataset loads through the backend) so
 * it runs on the `electron` project with `launchApp({dask:true})`. NO pre-kill
 * (the user runs their own SpyDE on this box; Playwright manages its instance).
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { join } from 'path'
const {
  launchApp, waitForSubwindowCount,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'guide_drive_shots')

test('Find Vectors tour auto-loads its dataset and "Show me" drives a step', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const ctx = await launchApp({ dask: true })
  const { page } = ctx
  try {
    // Settle so backend-ready's stdin pump is live before the tour fires the
    // autoload action (same guard the tutorial/lazy specs use).
    await page.waitForTimeout(1500)

    // --- open the Find Vectors guide from the Help "?" menu ------------------
    await page.getByTestId('help-button').click()
    await expect(page.getByTestId('help-menu')).toBeVisible()
    await page.getByTestId('help-guide-find-vectors').click()
    await expect(page.getByTestId('tour-overlay')).toBeVisible()
    const bubble = page.getByTestId('tour-bubble')
    await expect(bubble).toBeVisible()
    await expect(bubble).toContainText('What you’ll do')

    // The autoload state should be shown while the tutorial dataset loads.
    // (It may already be done by the time we assert — accept either.)
    await page.screenshot({ path: join(SHOTS, '01-tour-open-autoload.png') })

    // --- auto-load ran: navigator + signal subwindows opened -----------------
    await waitForSubwindowCount(page, 2, 60_000)
    // The "Loading tutorial data…" note clears once autoload resolves.
    await expect(page.getByTestId('tour-autoload-loading')).toHaveCount(0, { timeout: 60_000 })
    await expect(page.getByTestId('tour-autoload-error')).toHaveCount(0)
    await page.screenshot({ path: join(SHOTS, '02-autoload-done-2-windows.png') })

    // --- advance to an autoDrive step and click "Show me" --------------------
    // Steps: 0 intro, 1 "two linked windows" (autoDrive backend load), 2 "plot
    // toolbar" (autoDrive hover), 3 "Open Find Diffraction Vectors" (autoDrive
    // click → opens the wizard). Walk to step 3 (index 3 → "4 / 7").
    for (let i = 0; i < 3; i++) await page.getByTestId('tour-next').click()
    await expect(bubble).toContainText('4 / 7')
    await expect(bubble).toContainText('Open Find Diffraction Vectors')

    // This step is autoDrive → the "Show me ▶" button is offered.
    const showMe = page.getByTestId('tour-show-me')
    await expect(showMe).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '03-show-me-button.png') })

    // Run it: hover reveals the toolbar, click opens the wizard, waitFor
    // (visible: find-vectors-wizard) resolves, and the tour advances to step 5.
    await showMe.click()
    // The wizard must actually open (the drive's waitFor).
    await expect(page.getByTestId('find-vectors-wizard')).toBeVisible({ timeout: 60_000 })
    // And the tour auto-advanced past the click step.
    await expect(bubble).toContainText('5 / 7', { timeout: 60_000 })
    await expect(page.getByTestId('tour-drive-error')).toHaveCount(0)
    await page.screenshot({ path: join(SHOTS, '04-wizard-open-advanced.png') })

    ctx.assertNoJsErrors()
  } finally {
    await ctx.app.close()
  }
})
