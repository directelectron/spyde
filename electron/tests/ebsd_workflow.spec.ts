/**
 * ebsd_workflow.spec.ts — EBSD Indexing, end-to-end, driven through the CARET
 * exactly as a user drives it, on the bundled synthetic EBSD scan.
 *
 * The point of this spec is the thing headless tests cannot see: that the
 * matched orientation's Kikuchi BAND LINES actually draw on the pattern. The
 * backend suite already proves the geometry and the handlers; only pixels prove
 * the overlay. So every stage screenshots to ebsd_shots/ and the band stage
 * asserts green line pixels on the pattern canvas.
 *
 * Modelled on find_vectors_workflow.spec.ts: real Dask, bundled synthetic data,
 * signal-based waits (never a fixed sleep), renderer JS errors fail the run.
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, raiseWindow, backendAction, waitForSubwindowCount, countColorPixels, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'ebsd_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_ebsd', { nav: [16, 16], detector: [60, 60] })
  await waitForSubwindowCount(ctx.page, 2, 120_000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

test.setTimeout(300_000)

/** Open the EBSD Indexing caret on the pattern (signal) window. */
async function openCaret() {
  const { page } = ctx
  const sig = sigWindow(page)
  // Focus-raise then hover the TITLEBAR — hovering the figure itself puts the
  // cursor inside the iframe and the toolbar never mounts.
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  const button = sig.getByTestId('action-btn-EBSD Indexing')
  await expect(button, 'the EBSD Indexing toolbar button never appeared — the '
    + 'signal_types: [EBSD] gate did not match').toBeVisible({ timeout: 30_000 })
  await button.click()
  await expect(page.getByTestId('ebsd-wizard')).toBeVisible({ timeout: 10_000 })
}

test('the caret opens on an EBSD scan and builds a dictionary', async () => {
  const { page } = ctx
  await page.screenshot({ path: `${SHOTS}/01-loaded.png` })

  await openCaret()
  await page.screenshot({ path: `${SHOTS}/02-caret-load-tab.png` })

  // 2 Library → a coarse step so the dictionary builds quickly.
  await page.getByTestId('ebsd-tab-Library').click()
  await page.getByTestId('ebsd-step').fill('6')
  await page.screenshot({ path: `${SHOTS}/03-caret-library-tab.png` })
  await page.getByTestId('ebsd-build').click()

  // The backend acks with ebsd_dictionary_ready → the caret's status line.
  await expect(page.getByTestId('ebsd-status'))
    .toContainText(/Dictionary ready/i, { timeout: 180_000 })
  await page.screenshot({ path: `${SHOTS}/04-dictionary-ready.png` })
  ctx.assertNoJsErrors()
})

test('the matched orientation draws Kikuchi band lines on the pattern', async () => {
  const { page } = ctx

  // The Refine tab shows the live match; the overlay is already attached.
  await page.getByTestId('ebsd-tab-Refine').click()
  await expect(page.getByTestId('ebsd-match'))
    .toContainText(/NCC/i, { timeout: 60_000 })

  await expect.poll(() => countColorPixels(page, 'green'), {
    timeout: 60_000,
    message: 'no green band lines drew on the EBSD pattern',
  }).toBeGreaterThan(0)

  await page.screenshot({ path: `${SHOTS}/05-bands-overlay.png` })
  await sigWindow(page).screenshot({ path: `${SHOTS}/06-bands-closeup.png` })
  ctx.assertNoJsErrors()
})

/** Zone-axis markers are #fab387 (orange) — a colour the shared green/red
 *  probes deliberately do not match, so they are counted separately here. */
async function countOrange(page): Promise<number> {
  let total = 0
  for (const frame of page.frames()) {
    try {
      total += await frame.evaluate(() => {
        let n = 0
        for (const c of Array.from(document.querySelectorAll('canvas'))) {
          const ctx2 = (c as HTMLCanvasElement).getContext('2d')
          if (!ctx2 || !(c as HTMLCanvasElement).width) continue
          const d = ctx2.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                      (c as HTMLCanvasElement).height).data
          for (let p = 0; p < d.length; p += 4) {
            const r = d[p], g = d[p + 1], b = d[p + 2]
            if (r > 200 && g > 130 && g < 210 && b > 90 && b < 180) n++
          }
        }
        return n
      })
    } catch { /* detached frame */ }
  }
  return total
}

test('the band count and zone-axis toggle change what is drawn', async () => {
  const { page } = ctx
  const many = await countColorPixels(page, 'green')
  expect(many, 'no bands were drawn to begin with').toBeGreaterThan(0)

  await page.getByTestId('ebsd-nbands').fill('3')
  await expect.poll(() => countColorPixels(page, 'green'), {
    timeout: 30_000, message: 'reducing the band count did not redraw',
  }).toBeLessThan(many)
  await sigWindow(page).screenshot({ path: `${SHOTS}/07-three-bands.png` })
  const few = await countColorPixels(page, 'green')

  // Raising it again must redraw MORE lines — without this the test would pass
  // on a control that only ever removes them.
  await page.getByTestId('ebsd-nbands').fill('14')
  await expect.poll(() => countColorPixels(page, 'green'), {
    timeout: 30_000, message: 'raising the band count did not redraw',
  }).toBeGreaterThan(few)

  expect(await countOrange(page), 'zone axes are off but drew anyway').toBe(0)
  await page.getByTestId('ebsd-zone-axes').check()
  await expect.poll(() => countOrange(page), {
    timeout: 30_000, message: 'the zone-axis markers never drew',
  }).toBeGreaterThan(0)
  await sigWindow(page).screenshot({ path: `${SHOTS}/08-zone-axes.png` })
  ctx.assertNoJsErrors()
})

test('indexing the scan opens the IPF orientation map', async () => {
  const { page } = ctx
  const before = await page.getByTestId('subwindow').count()

  await page.getByTestId('ebsd-tab-Run').click()
  await page.getByTestId('ebsd-run').click()

  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 240_000, message: 'indexing never opened the IPF window',
  }).toBeGreaterThan(before)
  const ipf = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title')
      .filter({ hasText: 'Orientation (IPF-Z)' }) }).first()
  await expect(ipf, 'the IPF-Z result window never opened')
    .toBeVisible({ timeout: 30_000 })

  // The completion status goes to the app status bar (emit_status), not the
  // caret — the caret's own line only tracks its own stages.
  await expect(page.locator('body'))
    .toContainText(/orientation map complete/i, { timeout: 240_000 })
  await page.screenshot({ path: `${SHOTS}/09-ipf-map.png` })
  await ipf.screenshot({ path: `${SHOTS}/10-ipf-closeup.png` })
  ctx.assertNoJsErrors()
})

test('the quality maps ride along as chip views on the IPF window', async () => {
  const { page } = ctx
  // Pick the IPF window by its TITLE chip, not by hasText — the EBSD caret
  // itself contains the word "orientations", so a text filter matches the
  // pattern window first.
  const ipf = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('subwindow-title')
      .filter({ hasText: 'Orientation (IPF-Z)' }) }).first()
  await expect(ipf).toBeVisible({ timeout: 15_000 })

  // An IPF map cannot show where it is WRONG; these are the maps that can.
  for (const label of ['IPF-Z', 'NCC', 'Similarity', 'ADP']) {
    await expect(ipf.getByTestId(new RegExp(`^view-chip-${label}-`)),
      `the ${label} map was not registered as a view`)
      .toBeVisible({ timeout: 15_000 })
  }
  await raiseWindow(ipf)
  await ipf.getByTestId(/^view-chip-Similarity-/).click()
  await expect.poll(() => countColorPixels(page, 'bright'), {
    timeout: 30_000, message: 'the Similarity view painted nothing',
  }).toBeGreaterThan(0)
  await ipf.screenshot({ path: `${SHOTS}/11-similarity-view.png` })
  ctx.assertNoJsErrors()
})
