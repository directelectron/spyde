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
  launchApp, backendAction, waitForSubwindowCount, dragCrosshair,
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

/** Checksum of one subwindow's figure canvases — "is this a different frame?" */
async function figureSignature(win: any): Promise<number> {
  const ifel = await win.locator('iframe').first().elementHandle()
  if (!ifel) return -1
  const frame = await ifel.contentFrame()
  if (!frame) return -1
  try {
    return await frame.evaluate(() => {
      let sum = 0
      for (const c of Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]) {
        const g = c.getContext('2d')
        if (!g || !c.width || !c.height) continue
        const d = g.getImageData(0, 0, c.width, c.height).data
        for (let p = 0; p < d.length; p += 4) {
          sum = (sum + (d[p] + d[p + 1] + d[p + 2]) * (1 + (p % 7))) % 2147483647
        }
      }
      return sum
    })
  } catch { return -1 }
}

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
  const srcSig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-test/ }) })
    .first()

  const dpSigs: number[] = []
  const walk = await dragCrosshair(page, srcNav, {
    dx: -30, steps: 4, settleMs: 1200,
    onStep: async (i: number) => {
      dpSigs.push(await figureSignature(srcSig))
      await page.screenshot({
        path: join(SHOTS, `${String(i + 1).padStart(2, '0')}-during-fill.png`),
      })
    },
  })
  // eslint-disable-next-line no-console
  console.log('source nav during OM fill: crosshair moved', walk.moved, 'px',
              'DP signatures', JSON.stringify(dpSigs))

  // The point of the change: the SOURCE window is navigable throughout the
  // (minutes-long) match. Without the crosshair-moved guard this would pass
  // even if the press missed the widget entirely and the DP never changed.
  expect(walk.moved,
    'the source navigator crosshair never moved during the orientation fill',
  ).toBeGreaterThan(20)
  expect(new Set(dpSigs).size,
    `the source diffraction pattern did not follow the navigator while the
     orientation map was filling: ${JSON.stringify(dpSigs)}`,
  ).toBeGreaterThan(1)

  // Deliberately NOT waiting out the whole dense match (many minutes on 13k
  // patterns): everything this spec is about happens in the first blocks, and
  // tearing down mid-compute is itself a supported path
  // (close_cancels_compute.spec.ts). Give it a short grace so the shot is
  // representative, then finish.
  await ctx.backend.waitForLog('Orientation map complete', 60_000).catch(() => {})
  await page.screenshot({ path: join(SHOTS, '90-finished.png') })
  ctx.assertNoJsErrors()
})
