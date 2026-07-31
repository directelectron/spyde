/**
 * eels_edge_component.spec.ts — add ONE exspy core-loss edge from the picker.
 *
 * The bug: the Fit caret's `+ Component` palette offered nine analytic shapes
 * and no way to reach `EELSCLEdge` at all. The only route to an edge was
 * "From <elements>", which REPLACES the whole model — so adding a single O-K
 * edge onto a background you had already tuned was impossible.
 *
 * The backend tests (test_eels_edge_component.py) cover the handler contract.
 * They cannot see whether the edge section actually RENDERS in the palette,
 * whether the composition seeds it, or whether the added edge draws a curve on
 * the spectrum — which is exactly what a headless-green-but-broken UI looks
 * like. Every stage is screenshotted (CLAUDE.md: the screenshot IS the test).
 *
 * Needs the `eels` extra. Without exspy the section renders the install line
 * instead of edges, and this spec skips rather than failing — CI runs one
 * Electron job with the extras and this must not break the one without.
 *
 * Run: npx playwright test tests/eels_edge_component.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'eels_edge_shots'

test('the component picker offers EELS edges on an EELS signal', async () => {
  test.setTimeout(420_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    // The bundled synthetic EELS SI: 200-800 eV, C/N/O K edges, and the
    // microscope parameters already stamped on it — an edge cannot be built
    // without those, so a dataset that lacked them would test the error path.
    await backendAction(page, 'load_test_data_eels', { nav: [6, 6], n_channels: 512 })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_000)

    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Fit').click()
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })
    // The palette (and the edge list with it) arrives asynchronously — the
    // backend samples it on a worker so the caret opens immediately.
    await page.waitForTimeout(3_000)

    await page.locator('[data-testid="fit-add-toggle"]').click()
    await expect(page.locator('[data-testid="fit-palette"]')).toBeVisible()

    // The section only exists on an EELS signal at all.
    const section = page.locator('[data-testid="fit-edge-section"]')
    await expect(section).toBeVisible({ timeout: 20_000 })
    await page.screenshot({ path: `${SHOTS}/01-picker-with-edges.png`, fullPage: true })

    if (await page.locator('[data-testid="fit-edge-no-exspy"]').count()) {
      test.skip(true, 'the eels extra (exspy) is not installed in this backend')
    }
    // The microscope parameters ship with the synthetic data, so this warning
    // must NOT be showing — if it is, the edge list would be empty for a
    // reason that has nothing to do with the picker.
    await expect(page.locator('[data-testid="fit-edge-no-microscope"]')).toHaveCount(0)

    // ── the edges the window actually contains ───────────────────────────
    const offered = await page.locator('[data-testid^="fit-add-edge-"]')
      .evaluateAll((els) => els.map((e) =>
        e.getAttribute('data-testid')!.replace('fit-add-edge-', '')))
    console.log(`edges offered: ${offered.length} — ${offered.slice(0, 12).join(', ')}`)
    expect(offered.length, 'the picker offered no EELS edges').toBeGreaterThan(0)

    // ── the filter is the way in when no composition is set ──────────────
    // A 200-800 eV window holds ~64 major edges, so the list is a catalogue,
    // not a shortlist, until the user narrows it.
    await page.locator('[data-testid="fit-edge-filter"]').fill('O_K')
    await expect(page.locator('[data-testid="fit-add-edge-O_K"]'))
      .toBeVisible({ timeout: 10_000 })
    const filtered = await page.locator('[data-testid^="fit-add-edge-"]').count()
    console.log(`edges after filtering to O_K: ${filtered}`)
    expect(filtered, 'the filter did not narrow the list').toBeLessThan(offered.length)
    await page.screenshot({ path: `${SHOTS}/01b-filtered.png`, fullPage: true })
    await page.locator('[data-testid="fit-edge-filter"]').fill('')

    // ── the composition seeds the suggestions ────────────────────────────
    // `metadata.Sample.elements` is what Plot Control's Composition panel
    // writes, so setting it must lead the list rather than needing more wiring.
    await backendAction(page, 'set_composition', { elements: ['C', 'N', 'O'] })
    await page.waitForTimeout(500)
    // Reopen the caret so the catalogue is rebuilt against the new metadata.
    await page.locator('[data-testid="fit-close"]').click()
    await page.waitForTimeout(500)
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Fit').click()
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })
    await page.waitForTimeout(3_000)
    await page.locator('[data-testid="fit-add-toggle"]').click()
    const okEdge = page.locator('[data-testid="fit-add-edge-O_K"]')
    await expect(okEdge).toBeVisible({ timeout: 20_000 })
    await page.screenshot({ path: `${SHOTS}/02-composition-seeded.png`, fullPage: true })

    // ── a background FIRST, so each edge JOINS the model instead of
    // replacing it — the whole point against "From <elements>" ───────────
    await page.locator('[data-testid="fit-add-PowerLaw"]').click()
    await page.waitForTimeout(1_500)

    // One edge per element in the composition, added one at a time. Building
    // the whole model this way is the workflow that did not exist before.
    for (const sub of ['C_K', 'N_K', 'O_K']) {
      await page.locator('[data-testid="fit-add-toggle"]').click()
      const btn = page.locator(`[data-testid="fit-add-edge-${sub}"]`)
      await expect(btn).toBeVisible({ timeout: 20_000 })
      await btn.click()
      await expect(page.locator(`[data-testid="fit-comp-${sub}"]`))
        .toBeVisible({ timeout: 60_000 })
    }
    await expect(page.locator('[data-testid="fit-comp-PowerLaw"]')).toBeVisible()
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/03-edges-added.png`, fullPage: true })

    // The parameters are the EDGE's, not a gaussian's — the component that
    // landed has to actually be an EELSCLEdge.
    await expect(page.locator('[data-testid="fit-p-O_K-intensity"]')).toBeVisible()
    const onset = await page.locator('[data-testid="fit-p-O_K-onset_energy"]')
      .inputValue()
    console.log(`O_K onset as shown: ${onset}`)
    expect(Math.abs(Number(onset) - 532), 'the edge did not land on its onset')
      .toBeLessThan(2)

    // ── and the model FITS ───────────────────────────────────────────────
    // Not decoration: an edge whose GOS curves never got attached silently
    // drops the whole model off the batched engine, and an edge that arrived
    // at intensity 1 against counts of 1e5 draws as a flat line on the axis.
    await page.locator('[data-testid="fit-spectrum"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/chi2/i, { timeout: 120_000 })
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/04-edges-fitted.png`, fullPage: true })

    const intensities: Record<string, number> = {}
    for (const sub of ['C_K', 'N_K', 'O_K']) {
      intensities[sub] = Number(await page
        .locator(`[data-testid="fit-p-${sub}-intensity"]`).inputValue())
    }
    console.log(`edge intensities after the fit: ${JSON.stringify(intensities)}`)
    // `intensity` is bounded at 0, so an edge the solver cannot use pins there.
    // The claim under test is that the edges PARTICIPATE — which they cannot
    // if the batched engine never got their curves.
    expect(Object.values(intensities).some((v) => v > 0),
      'every edge pinned at zero — the edges are not participating in the fit')
      .toBe(true)

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-25).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
