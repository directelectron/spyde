/**
 * fit_all_spectra.spec.ts — what "Fit all Spectra" is supposed to deliver.
 *
 * Four things, all of which used to be missing or wrong:
 *
 *   1. the button says "Fit all Spectra", not "Run scan";
 *   2. it opens a PLOT of the results — one map per component plus chi
 *      squared — rather than leaving the answer only reachable by scrubbing
 *      the navigator one pixel at a time;
 *   3. moving the navigator afterwards shows THAT position's fit, for every
 *      position and not only the converged ones;
 *   4. the positions that fit worse than their neighbours can be refit, from
 *      their best neighbour's answer.
 *
 * Run: npx playwright test tests/fit_all_spectra.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, raiseWindowOwning, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fit_all_spectra_shots'

test('Fit all Spectra fits everything, plots the maps, and can refit the poor', async () => {
  test.setTimeout(600_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'tutorial_load', { name: 'spectroscopy' })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_500)
    const windowsBefore = await page.getByTestId('subwindow').count()

    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Fit').click()
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })

    for (let i = 0; i < 2; i++) {
      await page.locator('[data-testid="fit-add-toggle"]').click()
      await expect(page.locator('[data-testid="fit-add-Gaussian"]')).toBeVisible({ timeout: 30_000 })
      await page.locator('[data-testid="fit-add-Gaussian"]').click()
      await page.waitForTimeout(800)
    }

    // ── 0. the maps window exists BEFORE anything is fitted ──────────────
    // The parameters are held per position from the moment the caret opens,
    // so the maps are there to watch fill in rather than appearing at the end.
    expect(
      await page.getByTestId('subwindow').count(),
      'no Fit-components window opened with the caret',
    ).toBeGreaterThan(windowsBefore)
    const chipsAtOpen = await page.getByTestId('subwindow').last()
      .locator('[data-testid^="view-chip-"]').allTextContents()
    console.log('chips before fitting:', JSON.stringify(chipsAtOpen))
    expect(
      chipsAtOpen.join('|'),
      'the empty maps window has no per-component chips',
    ).toMatch(/Gaussian/)

    // ── 1. the name ──────────────────────────────────────────────────────
    // The live-preview window opened by adding components can sit over the caret
    // by now; raise its own window first (the click a user makes without noticing).
    await raiseWindowOwning(page, "fit-wizard")
    await page.getByTestId('fit-tab-Run').click()
    await expect(page.locator('[data-testid="fit-run"]')).toHaveText(/Fit all Spectra/i)
    await page.screenshot({ path: `${SHOTS}/01-run-tab.png`, fullPage: true })

    await page.locator('[data-testid="fit-run"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/converged/i, { timeout: 300_000 })
    await page.waitForTimeout(3_000)
    await page.screenshot({ path: `${SHOTS}/02-fitted.png`, fullPage: true })

    // ── 2. the maps opened on their own ──────────────────────────────────
    // chi squared must be one of them: a component map read without it can be
    // a picture of the fit falling over rather than of the sample.
    const chips = await page.getByTestId('subwindow').last()
      .locator('[data-testid^="view-chip-"]').allTextContents()
      .catch(() => [] as string[])
    const bodyText = await page.locator('body').innerText()
    expect(
      chips.join('|').toLowerCase().includes('chi') ||
      /chi\s*squared/i.test(bodyText),
      `no chi-squared map among the results (chips: ${JSON.stringify(chips)})`,
    ).toBeTruthy()

    // ── 3. every position is stored, not only the converged ones ─────────
    await page.getByTestId('fit-tab-Model').click()
    const coverage = await page.locator('[data-testid="fit-coverage"]').textContent()
    const [done, total] = (coverage ?? '').match(/(\d+)\s*\/\s*(\d+)/)!
      .slice(1).map(Number)
    expect(
      done,
      `only ${done} of ${total} positions were stored — moving the navigator ` +
      `to the rest shows a stale model instead of their fit`,
    ).toBe(total)
    console.log(`coverage after the run: ${done}/${total}`)

    // ── 4. refit the poor ────────────────────────────────────────────────
    // The live-preview window opened by adding components can sit over the caret
    // by now; raise its own window first (the click a user makes without noticing).
    await raiseWindowOwning(page, "fit-wizard")
    await page.getByTestId('fit-tab-Run').click()
    const refit = page.locator('[data-testid="fit-refit-poor"]')
    await expect(refit).toBeVisible()
    const label = await refit.textContent()
    console.log('refit button says:', label)
    const windowsWithMaps = await page.getByTestId('subwindow').count()
    await refit.click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/poor/i, { timeout: 180_000 })
    await page.waitForTimeout(2_500)
    await page.screenshot({ path: `${SHOTS}/03-after-refit.png`, fullPage: true })

    // The automatic maps window is a PREVIEW of the fit as it stands, so a
    // refit REPLACES it. Three fits used to leave three identical-looking
    // windows stacked on each other with no way to tell which was current.
    expect(
      await page.getByTestId('subwindow').count(),
      'refitting stacked another result window instead of replacing it',
    ).toBe(windowsWithMaps)

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-20).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
