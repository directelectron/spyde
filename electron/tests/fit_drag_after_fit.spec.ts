/**
 * fit_drag_after_fit.spec.ts — "once you fit a spectrum you can't move either
 * component".
 *
 * Reported from the app and invisible to the Python tests, which have no
 * renderer and therefore never fire the events that come back at the backend
 * while a drag is in progress.
 *
 * The suspicion this is built to confirm or kill: `fit_current` REMEMBERS the
 * position, and the caret's navigator listener sends `fit_navigated` on any
 * figure pointer_up it does not recognise as its own. `fit_navigated` recalls
 * a stored fit — so after a fit, every drag could be immediately overwritten
 * by the stored values and the handle would snap straight back.
 *
 * Run: npx playwright test tests/fit_drag_after_fit.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fit_drag_after_fit_shots'

test('a component can still be dragged after Fit spectrum', async () => {
  test.setTimeout(420_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'tutorial_load', { name: 'spectroscopy' })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_500)

    const figId = await page.evaluate(() => {
      for (const s of Array.from(document.querySelectorAll('[data-testid="subwindow"]'))) {
        const tid = s.querySelector('iframe')?.getAttribute('data-testid') ?? ''
        const crumb = s.querySelector('[data-testid="window-breadcrumb"]')?.textContent ?? ''
        if (tid.startsWith('figure-') && !crumb.startsWith('N-')) return tid.slice('figure-'.length)
      }
      return ''
    })
    expect(figId).toBeTruthy()

    const widgets = async () =>
      page.evaluate((f) => (window as any)._spyde_test_widgets(f), figId)

    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Fit').click()
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })
    await page.locator('[data-testid="fit-add-toggle"]').click()
    await expect(page.locator('[data-testid="fit-add-Gaussian"]')).toBeVisible({ timeout: 30_000 })
    await page.locator('[data-testid="fit-add-Gaussian"]').click()
    await page.waitForTimeout(800)
    await page.locator('[data-testid="fit-add-toggle"]').click()
    await page.locator('[data-testid="fit-add-Gaussian"]').click()
    await page.waitForTimeout(800)

    const centre = (n: string) =>
      page.locator(`[data-testid="fit-p-${n}-centre"]`).inputValue().then(Number)

    // ── drag BEFORE any fit: the control case ────────────────────────────
    const post = (w: any, type: string, fields: Record<string, number>) =>
      page.evaluate(({ f, panel, id, t, fs }) => {
        window.postMessage({
          type: 'awi_event', figId: f,
          data: JSON.stringify({
            source: 'js', panel_id: panel, widget_id: id, event_type: t, ...fs,
          }),
        }, '*')
      }, { f: figId, panel: w.panel_id, id: w.id, t: type, fs: fields })

    const peakOf = async (name: string) => {
      const ws = await widgets()
      const points = ws.filter((w: any) => w.type === 'point')
      // The two gaussians' handles, in the order their components were added.
      return points[name === 'Gaussian' ? 0 : 1]
    }

    let w = await peakOf('Gaussian')
    await post(w, 'pointer_up', { x: 35.0, y: Number(w.data.y) })
    await page.waitForTimeout(1_200)
    expect(await centre('Gaussian'),
      'the drag did not work even BEFORE a fit').toBeCloseTo(35.0, 1)
    await page.screenshot({ path: `${SHOTS}/01-dragged-before-fit.png`, fullPage: true })

    // ── fit, then drag again ─────────────────────────────────────────────
    await page.locator('[data-testid="fit-spectrum"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/converged/i, { timeout: 60_000 })
    await page.waitForTimeout(1_500)
    const afterFit = await centre('Gaussian')
    await page.screenshot({ path: `${SHOTS}/02-fitted.png`, fullPage: true })

    w = await peakOf('Gaussian')
    const target = afterFit > 50 ? 20.0 : 80.0
    await post(w, 'pointer_up', { x: target, y: Number(w.data.y) })
    await page.waitForTimeout(2_000)
    await page.screenshot({ path: `${SHOTS}/03-dragged-after-fit.png`, fullPage: true })

    const moved = await centre('Gaussian')
    expect(
      moved,
      `after Fit spectrum the handle was dragged to ${target} but the model ` +
      `says ${moved} (it was ${afterFit} before the drag) — the component is ` +
      `frozen`,
    ).toBeCloseTo(target, 1)

    // And it must STAY moved — a stored fit must not creep back in.
    await page.waitForTimeout(2_500)
    expect(await centre('Gaussian'),
      'the component snapped back after the drag').toBeCloseTo(target, 1)

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-25).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
