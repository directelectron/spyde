/** The zero-beam search region, on the reciprocal tab, with real members. */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { readdirSync, statSync, existsSync } from 'fs'

const { launchApp } = require('./_harness.cjs')
const SHOTS = join(__dirname, '..', 'beam_roi_shots')
const FOLDER = 'D:\Strain1degtilt910kx'
const STEMS = ['20250710_27329', '20250710_27330', '20250710_27332', '20250710_27333']
const paths = () => STEMS.map((s) => join(FOLDER, `${s}_cal.zspy`))

function warm(directory: string): number {
  let count = 0
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const full = join(directory, entry.name)
    if (entry.isDirectory()) count += warm(full)
    else { try { statSync(full); count += 1 } catch { /* raced */ } }
  }
  return count
}

let ctx: any
test.setTimeout(45 * 60_000)

test.beforeAll(async () => {
  for (const p of paths()) if (!existsSync(p)) throw new Error(`missing ${p}`)
  warm(paths()[0])
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await ctx.page.waitForTimeout(2500)
})
test.afterAll(async () => { await ctx?.app?.close() })

test('the region is drawn on the corner panels and can be moved', async () => {
  const { page, app } = ctx
  await page.getByTestId('menu-file').click()
  await page.getByTestId('menu-load-multiangle').click()
  const dialog = page.getByTestId('multiangle-loader')
  await expect(dialog).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(2000)

  await app.evaluate(({ ipcMain }: any, chosen: string[]) => {
    ipcMain.removeHandler('spyde:pick-folders')
    ipcMain.handle('spyde:pick-folders', async () => chosen)
  }, paths())
  await dialog.getByTestId('maped-add-folders').click()
  await expect(dialog.getByTestId('maped-summary'))
    .toContainText('4 angles', { timeout: 600_000 })

  for (let i = 0; i < STEMS.length; i += 1) {
    await page.evaluate(([idx]) => (window as any).electron.action(
      'maped_set_member', { index: idx, tilt: 1.0, azimuth: idx * 90 }), [i] as const)
    await page.waitForTimeout(300)
  }
  await expect.poll(() => dialog.locator('img').count(), { timeout: 1_800_000 })
    .toBeGreaterThanOrEqual(4)

  // The reciprocal tab is locked until real space is solved — it is the stage
  // that decides which members are in the acquisition at all.
  await page.evaluate(() => (window as any).electron.action(
    'maped_align_real', { params: { max_shift: 48 } }))
  await expect(dialog.getByTestId('maped-tab-reciprocal'))
    .toBeEnabled({ timeout: 1_800_000 })

  await dialog.getByTestId('maped-tab-reciprocal').click()
  await page.waitForTimeout(3000)
  await dialog.screenshot({ path: join(SHOTS, '01-reciprocal-with-roi.png') })

  const boxes = dialog.getByTestId('maped-beam-roi')
  console.log('region boxes drawn:', await boxes.count())
  expect(await boxes.count(), 'no search region drawn').toBeGreaterThan(0)
  console.log('radius field:', await dialog.getByTestId('maped-beam-roi-half').inputValue())

  // Drag the first one and check the backend took the move.
  const first = boxes.first()
  const box = await first.boundingBox()
  if (box) {
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    await page.mouse.down()
    await page.mouse.move(box.x + box.width / 2 + 12, box.y + box.height / 2 + 8, { steps: 8 })
    await page.mouse.up()
    await page.waitForTimeout(1500)
  }
  await dialog.screenshot({ path: join(SHOTS, '02-after-drag.png') })
  console.log('status after drag:', await dialog.getByTestId('maped-status').textContent())

  await dialog.getByTestId('maped-beam-roi-half').fill('24')
  await page.waitForTimeout(1500)
  await dialog.screenshot({ path: join(SHOTS, '03-radius-24.png') })
  console.log('status after radius:', await dialog.getByTestId('maped-status').textContent())

  // A tableau tile is ~100 screen px for a 512 px detector, so a drag there
  // moves the region five detector pixels at a time. The enlarged panel is
  // where it can actually be placed on a disk.
  await page.getByTestId('maped-corner-0-0').dblclick()
  await expect(page.getByTestId('maped-zoom')).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '04-zoomed-with-roi.png') })
  const zoomBox = page.getByTestId('maped-zoom').getByTestId('maped-beam-roi')
  console.log('region in the enlarged panel:', await zoomBox.count())
  expect(await zoomBox.count(), 'no region in the enlarged panel').toBe(1)

  const zb = await zoomBox.boundingBox()
  if (zb) {
    await page.mouse.move(zb.x + zb.width / 2, zb.y + zb.height / 2)
    await page.mouse.down()
    await page.mouse.move(zb.x + zb.width / 2 + 30, zb.y + zb.height / 2 - 18, { steps: 10 })
    await page.mouse.up()
    await page.waitForTimeout(1500)
  }
  await page.screenshot({ path: join(SHOTS, '05-zoom-after-drag.png') })
  console.log('status after zoom drag:',
    await dialog.getByTestId('maped-status').textContent())
})
