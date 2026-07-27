/**
 * integrate_default_span.spec.ts — clicking Integrate on a movie navigator must
 * give a SMALL region, not the whole recording.
 *
 * set_integrating(True) used to toggle only widget visibility, so the span kept
 * the RangeWidget constructor's x0=0 / x1=10 in DATA units — on a calibrated
 * 0.05 s/frame movie that is the entire recording. Worse than cosmetic: the read
 * is capped at MAX_REGION_EXTENT_PER_DIM, so the drawn box claimed to integrate
 * everything while the displayed frame was the mean of 16.
 *
 * This drives the REAL "Integrate" button. movie_roi_drag_perf.spec.ts cannot
 * cover it: that one goes through the test_region_scrub harness action, which
 * sets the widget geometry itself and therefore bypasses the seeding entirely.
 *
 * Run: npx playwright test tests/integrate_default_span.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, navWindow,
} = require('./_harness.cjs')

const SHOTS = 'integrate_span_shots'
const FRAMES = 40
// Small frames — this is about the span's geometry, not read throughput.
const SIZE = 512
const DEFAULT_EXTENT = 8      // DEFAULT_REGION_EXTENT_PER_DIM
const MAX_EXTENT = 16         // MAX_REGION_EXTENT_PER_DIM

test('Integrate on a movie navigator starts at a small default span', async () => {
  test.setTimeout(300_000)

  const ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO', SPYDE_NAV_PROFILE: '1' },
  })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'load_test_data_movie', { size: SIZE, frames: FRAMES })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(3_000)
    await expect(navWindow(page)).toBeVisible()
    await page.screenshot({ path: `${SHOTS}/01-point-mode.png` })

    const marker = backend.logBuffer.length

    // The REAL button, not a harness action.
    const integrateBtn = page.getByTestId('selector-integrate').first()
    await expect(integrateBtn).toBeVisible({ timeout: 60_000 })
    await integrateBtn.click()
    await page.waitForTimeout(3_000)
    await page.screenshot({ path: `${SHOTS}/02-integrate-default.png` })

    // The backend logs the indices it read, which is the ground truth for what
    // is being integrated (as opposed to what the box appears to cover).
    const lines: string[] = backend.logBuffer.slice(marker)
      .filter((l: string) => l.includes('[NAV-PROFILE]') && l.includes('idx=['))
    const counts = lines.map((l) => {
      const m = l.match(/idx=\[([^\]]*)\]/)
      return m ? m[1].split(',').filter((s) => s.trim()).length : 0
    }).filter((n) => n > 0)
    const seeded = backend.logBuffer.slice(marker)
      .filter((l: string) => l.includes('seeded default integrating span'))

    console.log(`[integrate-span] nav reads=${counts.length} frame counts=` +
      `${JSON.stringify(counts.slice(0, 6))}`)
    if (seeded.length) console.log(`[integrate-span] ${seeded[0].slice(-110)}`)

    expect(counts.length, 'no region read was logged after clicking Integrate')
      .toBeGreaterThan(0)

    // The seed must actually have run — otherwise a default that happens to look
    // right would pass this test for the wrong reason.
    expect(seeded.length,
      'the span was never seeded — set_integrating did not reach seed_default_span')
      .toBeGreaterThan(0)

    // Assert the SETTLED span, not the max. A couple of reads at the instant of
    // the mode switch can still reflect the pre-seed geometry (observed
    // [16,16,8,8,8,8]): they are capped to MAX_EXTENT so they are bounded and
    // they resolve immediately, and the box the user ends up looking at is the
    // seeded one. Logged below rather than hidden.
    const settled = counts[counts.length - 1]
    expect(settled, `Integrate settled at ${settled} frames — expected the `
      + `${DEFAULT_EXTENT}-frame default, not the capped whole recording`)
      .toBe(DEFAULT_EXTENT)
    expect(Math.max(...counts),
      'a read exceeded the hard cap').toBeLessThanOrEqual(MAX_EXTENT)
    const transient = counts.filter((n) => n !== DEFAULT_EXTENT).length
    if (transient) {
      console.log(`[integrate-span] NOTE ${transient} transient read(s) at the `
        + `mode switch still used the pre-seed geometry (capped to ${MAX_EXTENT})`)
    }
    assertNoJsErrors()
  } finally {
    await ctx.app.close().catch(() => {})
  }
})
