/**
 * fit_quality.spec.ts — after fitting, does the drawn model match the drawn data?
 *
 * Every other fit spec asserts on parameters, coverage or window counts. None
 * of them would notice the thing actually reported: a model curve sitting well
 * below the spectrum it is supposed to describe. This reads BOTH curves out of
 * the same panel state and compares them, which is exactly what the eye does.
 *
 * It also checks the follow-up: at a position that has already been fitted,
 * pressing "Fit spectrum" should have nothing left to do. If it visibly
 * improves the curve then what was on screen was not that position's fit.
 *
 * Run: npx playwright test tests/fit_quality.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fit_quality_shots'

test('the fitted model matches the spectrum on screen', async () => {
  test.setTimeout(600_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'tutorial_load', { name: 'spectroscopy' })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_500)

    const figIds = await page.evaluate(() => {
      const out: Record<string, string> = {}
      for (const s of Array.from(document.querySelectorAll('[data-testid="subwindow"]'))) {
        const tid = s.querySelector('iframe')?.getAttribute('data-testid') ?? ''
        const crumb = s.querySelector('[data-testid="window-breadcrumb"]')?.textContent ?? ''
        if (!tid.startsWith('figure-')) continue
        out[crumb.startsWith('N-') ? 'nav' : 'sig'] = tid.slice('figure-'.length)
      }
      return out
    })

    /** The panel's PRIMARY line (the spectrum) and its `model` overlay. */
    const curves = async () => page.evaluate((f) => {
      const dec = (b64?: string) => {
        if (!b64) return null
        const bin = atob(b64)
        const bytes = new Uint8Array(bin.length)
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
        return Array.from(new Float64Array(bytes.buffer))
      }
      const hook = (window as any)._spyde_test_panel_json
      for (const raw of hook ? hook(f) : []) {
        const d = JSON.parse(raw)
        const data = dec(d.data_b64)
        if (!data) continue
        const model = (d.extra_lines ?? [])
          .filter((l: any) => l.label === 'model')
          .map((l: any) => dec(l.data_b64))[0] ?? null
        return { data, model }
      }
      return null
    }, figIds.sig)

    /** Residual as a fraction of the data's own range — what the eye judges. */
    const misfit = (c: { data: number[]; model: number[] | null }) => {
      expect(c.model, 'no model curve is drawn').toBeTruthy()
      const n = Math.min(c.data.length, c.model!.length)
      let se = 0
      for (let i = 0; i < n; i++) se += (c.data[i] - c.model![i]) ** 2
      const rms = Math.sqrt(se / n)
      const span = Math.max(...c.data) - Math.min(...c.data)
      return rms / (span || 1)
    }

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

    await page.getByTestId('fit-tab-Run').click()
    await page.locator('[data-testid="fit-run"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/converged/i, { timeout: 300_000 })
    await page.getByTestId('fit-tab-Model').click()
    await page.waitForTimeout(2_000)
    await page.screenshot({ path: `${SHOTS}/01-after-fit-all.png`, fullPage: true })

    const afterAll = misfit((await curves())!)
    console.log(`misfit right after Fit all Spectra: ${(afterAll * 100).toFixed(1)}% of range`)

    // ── sweep the navigator: ONE position proves nothing ─────────────────
    // The report is "if I go through all of the positions" — so walk a grid
    // of them and find the worst, rather than trusting whichever the
    // navigator happens to start on.
    const cross = await page.evaluate((f) => {
      const ws = (window as any)._spyde_test_widgets(f)
      return ws.find((w: any) => w.type === 'crosshair')
    }, figIds.nav)
    expect(cross, 'the navigator has no crosshair').toBeTruthy()

    const goTo = async (cx: number, cy: number) => {
      await page.evaluate(({ f, panel, id, x, y }) => {
        window.postMessage({
          type: 'awi_event', figId: f,
          data: JSON.stringify({
            source: 'js', panel_id: panel, widget_id: id,
            event_type: 'pointer_up', cx: x, cy: y,
          }),
        }, '*')
      }, { f: figIds.nav, panel: cross.panel_id, id: cross.id, x: cx, y: cy })
      await page.waitForTimeout(1_200)
    }

    let worst = { m: 0, at: [0, 0] as number[] }
    const swept: number[] = []
    for (const cy of [2, 10, 16, 24, 30]) {
      for (const cx of [2, 10, 16, 24, 30]) {
        await goTo(cx, cy)
        const m = misfit((await curves())!)
        swept.push(m)
        if (m > worst.m) worst = { m, at: [cx, cy] }
        if (swept.length <= 3) {
          console.log(`  at [${cx},${cy}] misfit ${(m * 100).toFixed(1)}% ` +
            `status="${await page.locator('[data-testid="fit-status"]').textContent()}"` +
            ` coverage="${await page.locator('[data-testid="fit-coverage"]').textContent()}"`)
        }
      }
    }
    swept.sort((a, b) => a - b)
    console.log(`swept ${swept.length} positions — median ` +
      `${(swept[swept.length >> 1] * 100).toFixed(1)}%, worst ` +
      `${(worst.m * 100).toFixed(1)}% at [${worst.at}]`)

    await goTo(worst.at[0], worst.at[1])
    await page.screenshot({ path: `${SHOTS}/03-worst-position.png`, fullPage: true })
    const worstBefore = misfit((await curves())!)
    await page.locator('[data-testid="fit-spectrum"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/chi2/i, { timeout: 60_000 })
    await page.waitForTimeout(1_500)
    const worstAfter = misfit((await curves())!)
    await page.screenshot({ path: `${SHOTS}/04-worst-refitted.png`, fullPage: true })
    console.log(`worst position: ${(worstBefore * 100).toFixed(1)}% -> ` +
      `${(worstAfter * 100).toFixed(1)}% after Fit spectrum`)

    expect(
      worstBefore,
      `the WORST swept position misses the spectrum by ` +
      `${(worstBefore * 100).toFixed(1)}% of its range; fitting it alone ` +
      `reaches ${(worstAfter * 100).toFixed(1)}% — so the whole-scan fit did ` +
      `not leave that position at its own answer`,
    ).toBeLessThan(Math.max(worstAfter * 1.5, 0.10))

    // ── the reported symptom: press Fit spectrum at an ALREADY fitted
    //    position. If it improves the curve, what was drawn was not the fit
    //    that was made here.
    await goTo(16, 16)
    await page.waitForTimeout(1_000)
    await page.locator('[data-testid="fit-spectrum"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/chi2/i, { timeout: 60_000 })
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/02-after-fit-spectrum.png`, fullPage: true })
    const afterOne = misfit((await curves())!)
    console.log(`misfit after Fit spectrum here    : ${(afterOne * 100).toFixed(1)}% of range`)

    expect(
      afterAll,
      `the model drawn after Fit all Spectra misses the spectrum by ` +
      `${(afterAll * 100).toFixed(1)}% of its range — fitting one spectrum ` +
      `here gets to ${(afterOne * 100).toFixed(1)}%`,
    ).toBeLessThan(0.10)

    expect(
      afterAll,
      `"Fit spectrum" improved an already-fitted position from ` +
      `${(afterAll * 100).toFixed(1)}% to ${(afterOne * 100).toFixed(1)}% — ` +
      `the model on screen was not this position's fit`,
    ).toBeLessThan(afterOne * 1.5 + 0.01)

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-20).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
