/**
 * ipf_key.spec.ts — the IPF colour-key triangle legend (the stereographic
 * fundamental-sector key matplotlib/pyxem show) is pinned in the corner of an
 * IPF map window, and hides when the window is flipped to the 3-D explorer.
 *
 * Self-contained (Node 23 + Playwright break on cross-file .ts imports).
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

const map = (bg: string) =>
  `<!doctype html><html><body style="margin:0;height:100vh;background:${bg}"></body></html>`

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
})
test.afterAll(async () => { await app?.close() })

test.beforeEach(async () => {
  await page.reload()
  await page.waitForSelector('[data-testid="mdi-area"]')
})

async function inject(msg: Record<string, unknown>) {
  await page.evaluate((m) => { (window as Window & { _spyde_test_inject?: (m: unknown) => void })._spyde_test_inject?.(m) }, msg)
}

test('IPF colour-key triangle shows on the 2-D map, hides in 3-D', async () => {
  // An IPF window: the 2-D map + a 3-D explorer figure (so the 2D/3D toggle shows).
  await inject({
    type: 'figure', window_id: 1, fig_id: 'ipfmap',
    html: map('linear-gradient(#d23 0%, #2d2 50%, #23d 100%)'),
    title: 'Sig — Orientation (IPF-Z)', is_navigator: false,
  })
  await inject({
    type: 'figure', window_id: 1, fig_id: 'ipf3d',
    html: map('#111'), title: 'IPF (3D)', is_navigator: false, view: '3d',
  })
  // The backend emits the colour-key triangle as a native anyplotlib figure
  // (view="ipf_key") for this window.
  await inject({
    type: 'figure', window_id: 1, fig_id: 'ipfkey',
    html: map('#222'), title: 'IPF colour key', is_navigator: false,
    view: 'ipf_key',
  })

  // Legend pinned over the 2-D map.
  await expect(page.getByTestId('ipf-key-1')).toBeVisible()
  await page.screenshot({ path: join(__dirname, '..', 'ipf_key_overlay.png') })

  // Switch to the 3-D explorer → the 2-D legend hides.
  await page.getByTestId('ipf-view-3d-1').click()
  await expect(page.getByTestId('ipf-key-1')).toBeHidden()

  // Back to 2-D → it returns.
  await page.getByTestId('ipf-view-2d-1').click()
  await expect(page.getByTestId('ipf-key-1')).toBeVisible()
})
