/**
 * ipf_two_window_vom.spec.ts — the two-window IPF layout on the VECTOR
 * orientation-mapping path.
 *
 * `ipf_two_window.spec.ts` proves it for the DENSE (raw) 4D-STEM OM. Vector-OM
 * reaches the same display through the same plumbing point
 * (`ipf_view.attach_ipf_3d(..., session=…)`, called from
 * vector_orientation_om._build_ipf_heatmap) but via a completely different
 * producer — `commit_result_tree`'s `on_tree` hook instead of a progressive
 * `open_result_tree` — so it gets its own proof on real pixels.
 *
 * The drive is lifted verbatim from vector_om_lazy.spec.ts (load_test_vectors →
 * Vector Orientation Mapping wizard → pick the real Silver .cif, mocked →
 * Generate Library), which is the point the live IPF window opens.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'

const CIF = join(__dirname, '..', '..', 'spyde', 'tests', 'Silver__0011135.cif')
const SHOTS = join(__dirname, '..', 'ipf_two_window_vom_shots')

let app: ElectronApplication
let page: Page
let ipfId = ''
let mapId = ''

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  // A file-scope `test.setTimeout` applies to TESTS, not to hooks — a hook keeps
  // the config default (120 s) unless it raises its own budget from inside. This
  // setup boots Electron, spins a real LocalCluster, finds vectors and runs the
  // vector-OM fit, which does not fit in 120 s on a loaded machine: the hook was
  // timing out before the first test ever ran.
  test.setTimeout(300_000)
  mkdirSync(SHOTS, { recursive: true })
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env },   // real LocalCluster + client
  })
  let daskReady = false
  app.process().stdout?.on('data', (d: Buffer) => {
    if (String(d).includes('Dask cluster ready')) daskReady = true
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  for (let i = 0; i < 80 && !daskReady; i++) await page.waitForTimeout(500)
  await app.evaluate(({ ipcMain }, cif) => {
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => cif)
  }, CIF)
  await page.evaluate(() => window.electron.action('load_test_vectors', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 4,
    { timeout: 60_000 },
  )
  // Find Vectors finishes on a background thread — Generate Library races it.
  await expect(page.getByTestId('status-text'))
    .toContainText('Found', { timeout: 60_000 })
  await page.waitForTimeout(1500)
})

test.afterAll(async () => { await app?.close() })

async function shotStats(buf: Buffer) {
  return await page.evaluate(async (b64: string) => {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image(); i.onload = () => res(i); i.onerror = rej
      i.src = 'data:image/png;base64,' + b64
    })
    const cv = document.createElement('canvas')
    cv.width = img.width; cv.height = img.height
    const c2 = cv.getContext('2d')!
    c2.drawImage(img, 0, 0)
    const d = c2.getImageData(0, 0, cv.width, cv.height).data
    let colorful = 0
    for (let p = 0; p < d.length; p += 4) {
      if (Math.max(d[p], d[p + 1], d[p + 2]) - Math.min(d[p], d[p + 1], d[p + 2]) > 40) colorful++
    }
    return { colorful, total: cv.width * cv.height }
  }, buf.toString('base64'))
}

test('vector-OM opens the map window AND the IPF explorer window', async () => {
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Vector Orientation Mapping') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()
  await vsig.getByTestId('subwindow-titlebar').hover()
  await vsig.getByTestId('action-btn-Vector Orientation Mapping').click()
  await expect(page.getByTestId('vector-orientation-wizard')).toBeVisible()

  await page.getByTestId('vom-pick-cif').click()
  await expect(page.getByTestId('vom-pick-cif')).toHaveText('Silver__0011135.cif')
  await page.getByTestId('vom-tab-Library').click()
  await page.getByTestId('vom-generate').click()

  // The explorer window is the one that owns the toggle group.
  const toggle = page.getByTestId(/^ipf-view-toggle-/).first()
  await expect(toggle, 'vector-OM never opened the IPF explorer window')
    .toBeAttached({ timeout: 200_000 })
  ipfId = (await toggle.getAttribute('data-testid'))!.replace('ipf-view-toggle-', '')

  // The map window is the one carrying the X/Y/Z projection chips.
  const chip = page.getByTestId(/^view-chip-IPF-X-/).first()
  await expect(chip, 'vector-OM never registered the X/Y/Z projection chips')
    .toBeAttached({ timeout: 60_000 })
  mapId = (await chip.getAttribute('data-testid'))!.replace('view-chip-IPF-X-', '')
  expect(mapId, 'the maps and the IPF must be SEPARATE windows').not.toBe(ipfId)
  for (const d of ['X', 'Y', 'Z']) {
    await expect(page.getByTestId(`view-chip-IPF-${d}-${mapId}`)).toBeAttached()
  }
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '01-vom-both-windows.png') })
})

test('all four toggle states render on the vector-OM IPF window', async () => {
  const win = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(`ipf-view-toggle-${ipfId}`) }).first()
  // The result windows CASCADE — raise this one so its box is on top.
  await win.getByTestId('subwindow-titlebar').click()
  await page.waitForTimeout(500)
  const states: Array<[string, string, string]> = [
    ['2d', 'points', '02-vom-2d-points.png'],
    ['2d', 'heatmap', '03-vom-2d-heatmap.png'],
    ['3d', 'points', '04-vom-3d-points.png'],
    ['3d', 'heatmap', '05-vom-3d-heatmap.png'],
  ]
  for (const [dim, style, file] of states) {
    await page.getByTestId(`ipf-view-${dim}-${ipfId}`).click({ force: true })
    await page.getByTestId(`ipf-style-${style}-${ipfId}`).click({ force: true })
    await page.waitForTimeout(3500)
    const bb = (await win.boundingBox())!
    const st = await shotStats(await page.screenshot({ clip: bb, path: join(SHOTS, file) }))
    console.log(`[vom two-window] ${dim}/${style} =`, JSON.stringify(st))
    expect(st.colorful, `${dim} · ${style} rendered nothing chromatic`).toBeGreaterThan(100)
  }
})

/**
 * NB rotation is NOT asserted here — `load_test_vectors` is a synthetic field
 * in which EVERY scan position has the SAME lattice, so vector-OM fits one
 * identical orientation everywhere (the IPF-X map is a single flat colour and
 * the whole sphere cloud collapses onto one point — see 01/04 in the shots).
 * A camera that "rotates to face the picked orientation" therefore CANNOT move
 * between two picks on this data, and asserting that it does would only be
 * testing the fixture. The rotation itself is proven on real grain variation in
 * ipf_two_window.spec.ts (si_grains, dense OM). What this test pins is that the
 * vector-OM pick path runs end-to-end and leaves the explorer rendering.
 */
test('picking on the vector-OM map drives the explorer cleanly', async () => {
  const win = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId(`ipf-view-toggle-${ipfId}`) }).first()
  await page.getByTestId(`ipf-view-3d-${ipfId}`).click({ force: true })
  await page.getByTestId(`ipf-style-points-${ipfId}`).click({ force: true })
  await page.waitForTimeout(3000)
  const bb = (await win.boundingBox())!

  const pick = (iy: number, ix: number) => page.evaluate(
    (p) => window.electron.action('test_ipf_pick', p), { iy, ix })
  await pick(0, 0)
  await page.waitForTimeout(2000)
  await page.screenshot({ clip: bb, path: join(SHOTS, '06-vom-pick-0-0.png') })
  await pick(5, 5)
  await page.waitForTimeout(2000)
  const st = await shotStats(
    await page.screenshot({ clip: bb, path: join(SHOTS, '07-vom-pick-5-5.png') }))
  console.log('[vom two-window] after pick =', JSON.stringify(st))
  expect(st.colorful, 'the sphere went blank after a pick').toBeGreaterThan(100)
})
