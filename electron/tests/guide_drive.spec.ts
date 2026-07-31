/**
 * guide_drive.spec.ts — the in-app guided tutorials, end to end in the real app.
 *
 * The tour is DESCRIPTIVE and CLICK-THROUGH: it loads its tutorial dataset once
 * on open, spotlights each step over the LIVE UI, ends on a "More info" step,
 * and closes everything it opened on exit. This spec pins the behaviours that
 * were actually broken, in the running app:
 *
 *   1. DISMISSAL — clicking outside the tour must NOT exit it. Only ✕ (or Done)
 *      does. This is the headline fix: the overlay used to be a full-screen
 *      `onClick={onClose}` hit-target, so a stray click killed the walkthrough
 *      mid-way AND swallowed the very click each step asked the user to make.
 *   2. CLICK-THROUGH — a click while the tour is open reaches the real UI
 *      underneath (the tour is not modal), which is what makes step 1 fixable
 *      rather than just "not dismissing".
 *   3. SELF-CONTAINED — opening a guide auto-loads exactly ONE dataset.
 *   4. MORE INFO — the walkthrough ends on a step with the technique's further
 *      reading.
 *   5. CLEANUP — ✕ tears the example data back down (no lingering windows).
 *   6. HELP BAR — one row per technique with an Info / Guided-tour subtoolbar,
 *      and Info opens the technique dialog.
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

test('guided tutorial: click-through, ✕-only exit, More info, and full cleanup', async () => {
  mkdirSync(SHOTS, { recursive: true })
  const ctx = await launchApp({ dask: true })
  const { page } = ctx
  try {
    // Settle so backend-ready's stdin pump is live before the tour fires the
    // autoload action (same guard the tutorial/lazy specs use).
    await page.waitForTimeout(1500)

    const subwindows = page.getByTestId('subwindow')
    const overlay = page.getByTestId('tour-overlay')
    const bubble = page.getByTestId('tour-bubble')

    // --- 6a. the help bar lists TECHNIQUES with an Info/tour subtoolbar ------
    await page.getByTestId('help-button').click()
    await expect(page.getByTestId('help-menu')).toBeVisible()
    await expect(page.getByTestId('help-technique-find-vectors')).toBeVisible()
    await expect(page.getByTestId('help-info-find-vectors')).toBeVisible()
    await expect(page.getByTestId('help-guide-find-vectors')).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '01-help-bar-subtoolbar.png') })

    // --- 6b. Info opens the technique dialog with real external links --------
    await page.getByTestId('help-info-find-vectors').click()
    const infoDialog = page.getByTestId('guide-info-dialog')
    await expect(infoDialog).toBeVisible()
    await expect(page.getByTestId('guide-info-links')).toBeVisible()
    // The links are the curated upstream docs (pyxem for this technique).
    await expect(
      infoDialog.locator('[data-testid^="guide-info-link-https://pyxem.org"]').first(),
    ).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '02-info-dialog.png') })
    await page.getByTestId('guide-info-close').click()
    await expect(infoDialog).toHaveCount(0)

    // --- 3. open the tour; it auto-loads EXACTLY one dataset ----------------
    await page.getByTestId('help-button').click()
    await page.getByTestId('help-guide-find-vectors').click()
    await expect(overlay).toBeVisible()
    await expect(bubble).toBeVisible()

    await waitForSubwindowCount(page, 2, 60_000)
    await expect(page.getByTestId('tour-autoload-loading')).toHaveCount(0, { timeout: 60_000 })
    await expect(page.getByTestId('tour-autoload-error')).toHaveCount(0)
    // Let any (buggy) extra loads settle, then assert we still have exactly 2 —
    // no double/triple-load stacking. (find-vectors = navigator + signal = 2.)
    await page.waitForTimeout(1500)
    expect(await subwindows.count()).toBe(2)
    await page.screenshot({ path: join(SHOTS, '03-tour-open-one-dataset.png') })

    // --- 1. THE HEADLINE: clicking outside the tour must NOT exit it --------
    // Click the MDI backdrop, a subwindow titlebar, and the app bar — the three
    // places a user's hand naturally lands mid-tour. The tour survives all of
    // them; only ✕ closes it.
    const mdi = page.getByTestId('mdi-area')
    const mdiBox = await mdi.boundingBox()
    // A far corner of the MDI area, clear of the bubble and of any subwindow.
    await page.mouse.click(mdiBox!.x + 12, mdiBox!.y + mdiBox!.height - 12)
    await expect(overlay).toBeVisible()
    await page.getByTestId('subwindow-titlebar').first().click()
    await expect(overlay).toBeVisible()
    await page.getByTestId('app-bar').click({ position: { x: 400, y: 10 } })
    await expect(overlay).toBeVisible()
    // …and the step did not change either (a click is not a Next).
    await expect(bubble).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '04-clicked-away-still-open.png') })

    // --- 2. CLICK-THROUGH: the app underneath still responds ---------------
    // The control-panel toggle is a plain renderer-state button in the app bar.
    // If the tour overlay were still eating clicks, this would not flip.
    const dock = page.getByTestId('plot-control-dock')
    const dockWasVisible = await dock.count() > 0
    await page.getByTestId('toggle-sidebar').click()
    await expect
      .poll(async () => (await dock.count()) > 0, { timeout: 5_000 })
      .toBe(!dockWasVisible)
    await expect(overlay).toBeVisible()   // …and STILL did not dismiss the tour
    // Put it back so the later screenshots match the default layout.
    await page.getByTestId('toggle-sidebar').click()
    await page.screenshot({ path: join(SHOTS, '05-click-through-works.png') })

    // --- 2b. the actual point of the fix: DO the step the tour is pointing at
    // The Find Vectors walkthrough tells you to click the peak-finding tool.
    // With the old modal overlay that click hit the overlay instead — it closed
    // the tour and never reached the button. Now it opens the real wizard AND
    // the tour stays up beside it, which is what a coachmark tour is for.
    // Walk to that step first, so this also pins that the callout bubble does
    // not come to rest ON TOP of the control it is spotlighting.
    for (let i = 0; i < 10; i++) {
      const title = await bubble.locator('h3').innerText()
      if (/Find Diffraction Vectors/i.test(title)) break
      await page.getByTestId('tour-next').click()
    }
    await expect(bubble.locator('h3')).toContainText('Find Diffraction Vectors')
    const signalWin = page.getByTestId('subwindow')
      .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
      .first()
    await signalWin.getByTestId('subwindow-titlebar').hover()
    await signalWin.getByTestId('action-btn-Find Diffraction Vectors').click()
    await expect(page.getByTestId('find-vectors-wizard')).toBeVisible({ timeout: 30_000 })
    await expect(overlay).toBeVisible()
    await page.screenshot({ path: join(SHOTS, '05b-performed-the-step.png') })
    // Close the wizard again (its own ✕ — NOT Escape, which would exit the tour).
    await page.getByTestId('fv-close').click()
    await expect(page.getByTestId('find-vectors-wizard')).toHaveCount(0)
    await expect(overlay).toBeVisible()

    // --- 4. walk to the end: the last step is "More info" -------------------
    for (let i = 0; i < 20; i++) {
      const next = page.getByTestId('tour-next')
      if (await next.count() === 0) break
      await next.click()
    }
    await expect(page.getByTestId('tour-done')).toBeVisible()
    const moreInfo = page.getByTestId('tour-more-info')
    await expect(moreInfo).toBeVisible()
    await expect(
      moreInfo.locator('[data-testid^="tour-info-link-"]').first(),
    ).toBeVisible()
    expect(await subwindows.count()).toBe(2)   // walking never opened windows
    await page.screenshot({ path: join(SHOTS, '06-more-info-final-step.png') })

    // --- 5. ✕ closes the tour AND tears the example data back down ----------
    await page.getByTestId('tour-close').click()
    await expect(overlay).toHaveCount(0)
    await expect(subwindows).toHaveCount(0, { timeout: 30_000 })
    await page.screenshot({ path: join(SHOTS, '07-closed-example-data-gone.png') })

    ctx.assertNoJsErrors()
  } finally {
    await ctx.app.close()
  }
})
