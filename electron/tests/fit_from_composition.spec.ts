/**
 * fit_from_composition.spec.ts — build the model from the elements present.
 *
 * `fit_from_composition` existed in the backend and in the staged registry but
 * nothing in the renderer ever sent it: an EELS model was reachable only by
 * adding gaussians by hand. This drives the button that now does.
 *
 * On an EELS signal `create_model()` auto-populates a background plus one
 * `EELSCLEdge` per element subshell; the edges are then TABULATED, because
 * `EELSCLEdge` has no batched port and without that step the fit falls back to
 * HyperSpy's one-pixel-at-a-time path — seconds against minutes on a real scan.
 * After that it is the ordinary caret, which is what the last assertion here
 * checks: the model it built actually fits.
 *
 * Data: the bundled synthetic EELS SI (`spyde.data.eels_si`) — power-law
 * background + C/N/O K edges at their real energies, no download.
 *
 * Run: npx playwright test tests/fit_from_composition.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fit_from_composition_shots'

test('the Fit caret builds an EELS model from the composition', async () => {
  test.setTimeout(600_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'load_test_data_eels', { nav: [8, 8] })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_500)

    const sigFig = await page.evaluate(() => {
      for (const s of Array.from(document.querySelectorAll('[data-testid="subwindow"]'))) {
        const tid = s.querySelector('iframe')?.getAttribute('data-testid') ?? ''
        const crumb = s.querySelector('[data-testid="window-breadcrumb"]')?.textContent ?? ''
        if (tid.startsWith('figure-') && !crumb.startsWith('N-')) return tid.slice('figure-'.length)
      }
      return ''
    })

    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Fit').click()
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })
    await page.waitForTimeout(1_000)

    // With no elements set there is nothing to build from, so no button.
    await expect(page.locator('[data-testid="fit-from-composition"]')).toHaveCount(0)
    await page.screenshot({ path: `${SHOTS}/01-no-elements.png`, fullPage: true })

    // Set the composition the way Plot Control's panel does.
    await backendAction(page, 'set_composition', { elements: ['C', 'N', 'O'] })
    const fromComp = page.locator('[data-testid="fit-from-composition"]')
    await expect(fromComp).toBeVisible({ timeout: 20_000 })
    await expect(fromComp).toHaveText(/C, N, O/)
    await page.screenshot({ path: `${SHOTS}/02-button-appeared.png`, fullPage: true })

    await fromComp.click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/Built \d+ components/i, { timeout: 60_000 })
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/03-model-built.png`, fullPage: true })

    // One component per edge plus a background — and they must be REAL EELS
    // components, not three gaussians someone placed by hand.
    const names = await page.locator('[data-testid^="fit-comp-"]')
      .evaluateAll((els) => els.map((e) =>
        e.getAttribute('data-testid')!.replace('fit-comp-', '')))
    console.log('components built:', JSON.stringify(names))
    expect(names.length, 'no components were built').toBeGreaterThan(3)
    for (const el of ['C', 'N', 'O']) {
      expect(names.join('|'), `no component for ${el}`).toContain(el)
    }

    // ── and the model it built has to actually fit ───────────────────────
    const curves = async () => page.evaluate((f) => {
      const dec = (b64?: string) => {
        if (!b64) return null
        const bin = atob(b64)
        const b = new Uint8Array(bin.length)
        for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i)
        return Array.from(new Float64Array(b.buffer))
      }
      const hook = (window as any)._spyde_test_panel_json
      for (const raw of hook ? hook(f) : []) {
        const d = JSON.parse(raw)
        const data = dec(d.data_b64)
        if (!data) continue
        const model = (d.extra_lines ?? [])
          .filter((l: any) => l.label === 'model').map((l: any) => dec(l.data_b64))[0]
        return { data, model }
      }
      return null
    }, sigFig)

    await page.locator('[data-testid="fit-spectrum"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/chi2/i, { timeout: 120_000 })
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/04-fitted.png`, fullPage: true })

    const c = (await curves())!
    expect(c.model, 'the built model draws no curve').toBeTruthy()
    const n = Math.min(c.data.length, c.model!.length)
    let se = 0
    for (let i = 0; i < n; i++) se += (c.data[i] - c.model![i]) ** 2
    const misfit = Math.sqrt(se / n) /
      ((Math.max(...c.data) - Math.min(...c.data)) || 1)
    console.log(`misfit of the composition model: ${(misfit * 100).toFixed(1)}%`)
    expect(
      misfit,
      `the model built from C, N, O misses the spectrum by ` +
      `${(misfit * 100).toFixed(1)}% of its range`,
    ).toBeLessThan(0.15)

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-20).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
