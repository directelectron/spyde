/**
 * view_bar.spec.ts — the unified per-window VIEW selector (chip strip + tiling).
 *
 * A result window can hold several NAMED views of one navigation field — strain
 * εxx/εyy/εxy, virtual images, IPF directions — each emitted as a figure tagged
 * with `view_label` (chip text) + `view_kind`. The frontend renders ONE chip
 * strip per window: click a chip to show that view, ⌘-click to TILE several with
 * a shared grid. This drives that UI directly via the test-inject hook (no slow
 * OM pipeline) and asserts: chips appear, single-select swaps the visible
 * figure, ⌘-click tiles two figures side by side.
 *
 * Self-contained (Node 23 + Playwright break on cross-file .ts imports).
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

// Minimal coloured figure HTML — distinct per view so a tiled screenshot reads
// clearly. We test chip selection / tiling layout, not canvas pixels.
const fig = (label: string, bg: string) =>
  `<!doctype html><html><body style="margin:0;height:100vh;display:flex;` +
  `align-items:center;justify-content:center;background:${bg};color:#fff;` +
  `font:600 28px sans-serif">${label}</body></html>`

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

// Emit three strain views into one window (window_id 900).
async function emitStrainViews() {
  const views: [string, string][] = [
    ['εxx', '#1f4d7a'], ['εyy', '#7a1f4d'], ['εxy', '#4d7a1f'],
  ]
  for (let i = 0; i < views.length; i++) {
    const [label, bg] = views[i]
    await inject({
      type: 'figure', window_id: 900, fig_id: `strain-${label}`,
      html: fig(label, bg), title: 'Sig — Strain',
      is_navigator: false, view_label: label, view_kind: '2d',
    })
  }
}

test('strain window shows a chip strip; click swaps, ⌘-click shows the side-by-side figure', async () => {
  await emitStrainViews()

  // The view bar + one chip per strain component appear.
  await expect(page.getByTestId('view-bar-900')).toBeVisible()
  await expect(page.getByTestId('view-chip-εxx-900')).toBeVisible()
  await expect(page.getByTestId('view-chip-εyy-900')).toBeVisible()
  await expect(page.getByTestId('view-chip-εxy-900')).toBeVisible()

  const strainWin = page.getByTestId('subwindow').filter({ has: page.getByTestId('view-bar-900') }).first()

  // Default: exactly the first view (εxx) is shown.
  await expect.poll(() => strainWin.locator('iframe:visible').count()).toBe(1)
  await expect(page.getByTestId('figure-strain-εxx')).toBeVisible()

  // Plain-click εyy → still one visible figure, now εyy.
  await page.getByTestId('view-chip-εyy-900').click()
  await expect(page.getByTestId('figure-strain-εyy')).toBeVisible()
  await expect(page.getByTestId('figure-strain-εxx')).toBeHidden()

  // ⌘-click εxx → selects {εyy, εxx}. The backend rebuilds ONE figure with the
  // selected views as side-by-side axes (view_label "__tiled__"); the frontend
  // swaps to it. (Injected here to stand in for the backend, which has no data
  // for this synthetic window.)
  await page.getByTestId('view-chip-εxx-900').click({ modifiers: ['Meta'] })
  await expect(page.getByTestId('view-chip-εxx-900')).toHaveCSS('background-color', 'rgb(137, 180, 250)')  // both chips active
  await inject({
    type: 'figure', window_id: 900, fig_id: 'strain-tiled',
    html: fig('εxx | εyy', '#2d2d44'), title: 'εxx / εyy',
    is_navigator: false, view_label: '__tiled__', view_kind: 'tiled',
  })
  // Exactly the one combined figure is shown — not two iframes.
  await expect(page.getByTestId('figure-strain-tiled')).toBeVisible()
  await expect.poll(() => strainWin.locator('iframe:visible').count()).toBe(1)
  await expect(page.getByTestId('figure-strain-εxx')).toBeHidden()
  await expect(page.getByTestId('figure-strain-εyy')).toBeHidden()

  // Click a single chip → back to that lone view (combined figure hidden).
  await page.getByTestId('view-chip-εxy-900').click()
  await expect(page.getByTestId('figure-strain-εxy')).toBeVisible()
  await expect(page.getByTestId('figure-strain-tiled')).toBeHidden()

  await page.screenshot({ path: join(__dirname, '..', 'view_bar_chips.png') })
})
