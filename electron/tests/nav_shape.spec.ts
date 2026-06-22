/**
 * nav_shape.spec.ts — the scan-shape / step-size confirm dialog.
 *
 * Injecting a `nav_shape_prompt` (as the backend emits when opening a navigated
 * dataset) must show the dialog pre-filled with the inferred shape; confirming
 * dispatches `confirm_nav_shape` with the chosen grid + step size. Renderer-only.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page

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
  await page.evaluate((m) => { (window as any)._spyde_test_inject?.(m) }, msg)
}

async function trackActions() {
  await app.evaluate(({ ipcMain }) => {
    ;(globalThis as any).__sent = []
    ipcMain.removeAllListeners('spyde:action')
    ipcMain.on('spyde:action', (_e, action, payload) => {
      ;(globalThis as any).__sent.push({ action, payload })
    })
  })
}
const sentActions = () => app.evaluate(() => (globalThis as any).__sent)

const PROMPT = {
  type: 'nav_shape_prompt',
  nav_shape: [12, 1],
  n_patterns: 12,
  signal_shape: [8, 8],
  scale: 1.0,
  units: 'nm',
  filename: 'scan.mrc',
}

test('nav_shape_prompt shows the dialog pre-filled', async () => {
  await inject(PROMPT)
  await expect(page.getByTestId('nav-shape-dialog')).toBeVisible()
  await expect(page.getByTestId('nav-shape-dialog')).toContainText('scan.mrc')
  expect(await page.getByTestId('nav-shape-x').inputValue()).toBe('12')
  expect(await page.getByTestId('nav-step').inputValue()).toBe('1')
})

test('factor presets fold a flat stack into a 2-D grid', async () => {
  await inject(PROMPT)
  // 12 frames → a 4×3 preset is offered.
  await page.getByTestId('nav-preset-4x3').click()
  expect(await page.getByTestId('nav-shape-x').inputValue()).toBe('4')
  expect(await page.getByTestId('nav-shape-y').inputValue()).toBe('3')
})

test('a shape whose product ≠ frame count warns and disables Open', async () => {
  await inject(PROMPT)
  await page.getByTestId('nav-shape-x').fill('5')
  await page.getByTestId('nav-shape-y').fill('3')   // 15 ≠ 12
  await expect(page.getByTestId('nav-shape-warn')).toBeVisible()
  await expect(page.getByTestId('nav-open')).toBeDisabled()
})

test('confirm dispatches confirm_nav_shape with grid + step size', async () => {
  await trackActions()
  await inject(PROMPT)
  await page.getByTestId('nav-preset-4x3').click()
  await page.getByTestId('nav-step').fill('3')
  await page.getByTestId('nav-units').fill('nm')
  await page.getByTestId('nav-open').click()

  expect(await sentActions()).toContainEqual({
    action: 'confirm_nav_shape',
    payload: { nav_shape: [4, 3], step_size: 3, units: 'nm' },
  })
  // Dialog closes after confirm.
  await expect(page.getByTestId('nav-shape-dialog')).toHaveCount(0)
})

test('cancel opens as-loaded (empty confirm payload)', async () => {
  await trackActions()
  await inject(PROMPT)
  await page.getByTestId('nav-cancel').click()
  expect(await sentActions()).toContainEqual({
    action: 'confirm_nav_shape',
    payload: {},
  })
  await expect(page.getByTestId('nav-shape-dialog')).toHaveCount(0)
})
