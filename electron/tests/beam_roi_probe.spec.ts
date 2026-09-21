/** The zero-beam search region: drawn, draggable, and on the enlarged panel.
 *
 * Synthetic members, because this is an INTERACTION test — the region's effect
 * on a solve is measured in the Python suite, where it can be compared against
 * a known answer instead of eyeballed.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'

const { launchApp } = require('./_harness.cjs')
const SHOTS = join(__dirname, '..', 'beam_roi_shots')
const MEMBER_DIRECTORY = join(require('os').tmpdir(), 'spyde-multiangle-test')
const TILTS = [1.0, 1.0, 1.0, 0.5, 0.5]
const SCAN = 28
const paths = () => TILTS.map((_t, i) =>
  join(MEMBER_DIRECTORY, `angle${String(i).padStart(2, '0')}.mrc`))

let ctx: any
test.setTimeout(10 * 60_000)

const send = (action: string, payload: Record<string, unknown> = {}) =>
  ctx.page.evaluate(([a, p]: any) => (window as any).electron.action(a, p),
    [action, payload] as const)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await ctx.page.waitForTimeout(1500)
  await send('write_test_multiangle_files', { nav: SCAN, sig: 64 })
  await ctx.page.waitForTimeout(2500)
})
test.afterAll(async () => { await ctx?.app?.close() })

test('the region is drawn, moves, and is placeable on the enlarged panel', async () => {
  const { page } = ctx
  await page.getByTestId('menu-file').click()
  await page.getByTestId('menu-load-multiangle').click()
  const dialog = page.getByTestId('multiangle-loader')
  await expect(dialog).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(1500)

  await send('maped_add_files', { paths: paths() })
  await page.waitForTimeout(3000)
  await send('maped_set_scan_shape', { scan_shape: [SCAN, SCAN] })
  await expect(dialog.getByTestId('maped-summary'))
    .toContainText('5 angles', { timeout: 120_000 })
  for (let i = 0; i < TILTS.length; i += 1) {
    await send('maped_set_member', { index: i, tilt: TILTS[i], azimuth: i * 70 })
    await page.waitForTimeout(200)
  }
  await expect.poll(() => dialog.locator('img').count(), { timeout: 240_000 })
    .toBeGreaterThanOrEqual(4)

  await send('maped_align_real', { params: { max_shift: 16 } })
  await expect(dialog.getByTestId('maped-tab-reciprocal'))
    .toBeEnabled({ timeout: 300_000 })
  // The real-space evidence belongs IN the dialog: the dialog is a full-screen
  // modal, so a window behind it cannot be looked at while it is up. It lives
  // on the tab that solves it, so stand there.
  await dialog.getByTestId('maped-tab-real').click()
  await expect(dialog.getByTestId('maped-real-evidence'))
    .toBeVisible({ timeout: 300_000 })
  await page.waitForTimeout(1200)
  await dialog.screenshot({ path: join(SHOTS, '00-real-evidence.png') })
  console.log('sharpness:', await dialog.getByTestId('maped-real-gain')
    .textContent().catch(() => 'none'))
  console.log('verdict  :', await dialog.getByTestId('maped-real-verdict')
    .textContent().catch(() => 'none'))

  await dialog.getByTestId('maped-tab-reciprocal').click()
  await page.waitForTimeout(2500)
  await dialog.screenshot({ path: join(SHOTS, '01-reciprocal-with-roi.png') })

  const boxes = dialog.getByTestId('maped-beam-roi')
  console.log('region boxes drawn:', await boxes.count())
  expect(await boxes.count(), 'no search region drawn').toBeGreaterThan(0)

  // Enlarge first: a tableau tile is far too coarse to aim on.
  const probe = await page.evaluate(() => {
    const panel = document.querySelector('[data-testid="maped-corner-0-0"]')
    if (!panel) return { found: false }
    const r = panel.getBoundingClientRect()
    const atCentre = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)
    return {
      found: true,
      rect: [Math.round(r.width), Math.round(r.height)],
      atCentre: (atCentre as HTMLElement)?.dataset?.testid
        ?? (atCentre as HTMLElement)?.tagName ?? null,
      zoomsBefore: document.querySelectorAll('[data-testid="maped-zoom"]').length,
    }
  })
  console.log('PROBE', JSON.stringify(probe))

  await page.getByTestId('maped-corner-0-0').dblclick()
  await page.waitForTimeout(800)
  console.log('zooms after playwright dblclick:', await page.getByTestId('maped-zoom').count())

  if (await page.getByTestId('maped-zoom').count() === 0) {
    const dispatched = await page.evaluate(() => {
      const panel = document.querySelector('[data-testid="maped-corner-0-0"]')
      panel?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))
      return document.querySelectorAll('[data-testid="maped-zoom"]').length
    })
    await page.waitForTimeout(600)
    console.log('zooms after a dispatched dblclick:', dispatched,
      '->', await page.getByTestId('maped-zoom').count())
  }
  await expect(page.getByTestId('maped-zoom')).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(1000)
  await page.screenshot({ path: join(SHOTS, '02-zoomed-with-roi.png') })

  const zoomBox = page.getByTestId('maped-zoom').getByTestId('maped-beam-roi')
  expect(await zoomBox.count(), 'no region in the enlarged panel').toBe(1)

  const before = await dialog.getByTestId('maped-status').textContent()
  const zb = await zoomBox.boundingBox()
  expect(zb, 'the enlarged region has no box').not.toBeNull()
  await page.mouse.move(zb!.x + zb!.width / 2, zb!.y + zb!.height / 2)
  await page.mouse.down()
  await page.mouse.move(zb!.x + zb!.width / 2 + 40, zb!.y + zb!.height / 2 - 25,
    { steps: 12 })
  await page.mouse.up()
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '03-zoom-after-drag.png') })
  const after = await dialog.getByTestId('maped-status').textContent()
  console.log('status before:', before)
  console.log('status after :', after)
  expect(after, 'dragging in the enlarged panel did not move the region')
    .not.toEqual(before)
  expect(after).toMatch(/Zero beam searched within/)
})
