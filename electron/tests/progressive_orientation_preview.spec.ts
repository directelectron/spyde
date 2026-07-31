/**
 * progressive_orientation_preview.spec.ts — the Orientation-Mapping half of the
 * "a progressive fill must show the SIGNAL too" work.
 *
 * Orientation's progressive window is NOT a navigator + signal pair: it is the
 * IPF map alone, one 2-D plot with no navigator (so
 * `live_signal.attach_signal_preview` is a documented no-op there and the map
 * already fills live). The navigator + signal pair during an OM run is the
 * SOURCE window, and its diffraction pattern is where a live orientation result
 * shows up — so the fix is that the matched-template overlay is attached BEFORE
 * the whole-field match instead of after it, making the whole (minutes-long)
 * fill navigable.
 *
 * This spec drives the real thing on the real SPED-Ag scan and captures it:
 * the IPF map filling in while the source DP stays navigable. The ordering
 * itself is pinned exactly (and fast) by
 * test_orientation_port.py::test_source_overlay_attaches_before_the_batch.
 *
 * Screenshots land in electron/progressive_orientation_shots/ (gitignored).
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'progressive_orientation_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  test.setTimeout(600_000)
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_sped_ag')
  await waitForSubwindowCount(ctx.page, 2, 420_000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test.setTimeout(600_000)

test('the IPF map fills in while the source DP stays navigable', async () => {
  const { page } = ctx
  const before = await page.getByTestId('subwindow').count()

  // Phase "ag" matches the real SPED-Ag scan (a mismatched phase gives a black
  // IPF and nothing to look at).
  await backendAction(page, 'run_test_orientation', { phase: 'ag' })

  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 600_000, message: 'the orientation IPF window never opened',
  }).toBeGreaterThan(before)
  const ipf = page.getByTestId('subwindow').filter({ hasText: /Orientation/ }).first()
  await expect(ipf).toBeVisible({ timeout: 30_000 })
  await page.screenshot({ path: join(SHOTS, '01-ipf-window-opened.png') })

  // While the map fills, the SOURCE navigator still drives the source DP — the
  // window pair that was inert for the whole run before this change.
  const srcNav = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-test/ }) })
    .first()
  const navBox = await srcNav.locator('iframe').first().boundingBox()
  expect(navBox).not.toBeNull()
  for (let i = 0; i < 4; i++) {
    const fx = 0.25 + 0.15 * i
    await page.mouse.move(navBox!.x + navBox!.width * fx, navBox!.y + navBox!.height * 0.5)
    await page.mouse.down()
    await page.mouse.move(navBox!.x + navBox!.width * fx + 3,
                          navBox!.y + navBox!.height * 0.5 + 3, { steps: 4 })
    await page.mouse.up()
    await page.waitForTimeout(1200)
    await page.screenshot({ path: join(SHOTS, `${String(i + 2).padStart(2, '0')}-during-fill.png`) })
  }

  // Deliberately NOT waiting out the whole dense match (many minutes on 13k
  // patterns): everything this spec is about happens in the first blocks, and
  // tearing down mid-compute is itself a supported path
  // (close_cancels_compute.spec.ts). Give it a short grace so the shot is
  // representative, then finish.
  await ctx.backend.waitForLog('Orientation map complete', 60_000).catch(() => {})
  await page.screenshot({ path: join(SHOTS, '90-finished.png') })
  ctx.assertNoJsErrors()
})
