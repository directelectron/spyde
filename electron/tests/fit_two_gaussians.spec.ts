/**
 * fit_two_gaussians.spec.ts — "Fit spectrum" on the two-gaussians tutorial.
 *
 * This exists because "Fit spectrum" reported CONVERGED while drawing a model
 * about half the height of the data. The status was true and the result was
 * wrong: the fit was running against the mean over navigation instead of the
 * spectrum on screen, because the spectrum was reconstructed from
 * `signal.data` + a navigator index rather than read from `plot.current_data`.
 *
 * Nothing in the handler tests could see that — they set up their own data and
 * never had a navigator. It needs the real app, a real navigator position, and
 * an assertion about the FITTED VALUES rather than about a status string.
 *
 * Dataset: `tutorial_spectroscopy` = hyperspy's `two_gaussians`, which is the
 * case this is meant to handle — two overlapping peaks on one axis.
 *
 * Run: npx playwright test tests/fit_two_gaussians.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fit_two_gaussians_shots'
const SQRT_2PI = Math.sqrt(2 * Math.PI)

test('Fit spectrum fits the displayed spectrum, not the navigation mean', async () => {
  test.setTimeout(420_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'tutorial_load', { name: 'spectroscopy' })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_500)
    await page.screenshot({ path: `${SHOTS}/01-loaded.png`, fullPage: true })

    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Fit').click()
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })

    const addGaussian = async () => {
      await page.locator('[data-testid="fit-add-toggle"]').click()
      await expect(page.locator('[data-testid="fit-palette"]')).toBeVisible({ timeout: 10_000 })
      await page.locator('[data-testid="fit-add-Gaussian"]').click()
      await page.waitForTimeout(900)
    }
    await addGaussian()
    await addGaussian()
    await expect(page.locator('[data-testid="fit-comp-Gaussian"]')).toBeVisible()
    await expect(page.locator('[data-testid="fit-comp-Gaussian 2"]')).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/02-two-gaussians.png`, fullPage: true })

    // ── fit the displayed spectrum ───────────────────────────────────────
    await page.locator('[data-testid="fit-spectrum"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/converged/i, { timeout: 60_000 })
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/03-fitted.png`, fullPage: true })

    // ── the assertion that actually catches the bug ──────────────────────
    // A HyperSpy Gaussian's A is the AREA, so the peak height it contributes
    // is A / (sigma * sqrt(2*pi)). Summed over both components that is the
    // model's height at the (coincident) centres.
    //
    // two_gaussians peaks at ~1050. Fitting the navigation mean produced ~460
    // — and still said "converged". Anything above ~700 can only come from a
    // fit against the real displayed spectrum, so this separates the two
    // outcomes without pinning an exact fitted value.
    const num = async (tid: string) =>
      Number(await page.locator(`[data-testid="${tid}"]`).inputValue())

    const peak =
      (await num('fit-p-Gaussian-A')) / ((await num('fit-p-Gaussian-sigma')) * SQRT_2PI) +
      (await num('fit-p-Gaussian 2-A')) / ((await num('fit-p-Gaussian 2-sigma')) * SQRT_2PI)

    expect(peak, `model peak ${peak.toFixed(0)} — too low for the displayed ` +
      `spectrum; this is what fitting the navigation mean looks like`)
      .toBeGreaterThan(700)

    // Both components must be doing work. A degenerate fit that drives one to
    // zero and lets the other carry everything would still clear the peak
    // check above.
    expect(await num('fit-p-Gaussian-A')).toBeGreaterThan(0)
    expect(await num('fit-p-Gaussian 2-A')).toBeGreaterThan(0)

    const errs = backend.logBuffer.filter((l: string) => /Traceback|CRITICAL/.test(l))
    expect(errs, `backend errors:\n${errs.join('\n')}`).toHaveLength(0)
    assertNoJsErrors()
  } finally {
    const tail = backend.logBuffer.slice(-30).join('\n')
    console.log(`\n──── backend log (tail) ────\n${tail}\n──────────────────────────`)
    await ctx.app.close().catch(() => {})
  }
})
