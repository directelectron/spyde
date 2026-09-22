/**
 * export_numpy.spec.ts — File → Export to NumPy writes the focused window's
 * maps as a .npz.
 *
 * The native Save dialog is replaced by one that answers with a temp path;
 * everything else is the real chain: the in-app File menu → the main process
 * → the backend's export_numpy → a file on disk. The file is checked to be a
 * zip archive (what np.savez writes) and the menu is screenshotted, so a
 * missing row would be visible, not just a failed locator.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'
import { tmpdir } from 'os'
import { existsSync, mkdirSync, readFileSync, rmSync } from 'fs'
import { raiseWindow, sigWindow } from './_harness.cjs'

let app: ElectronApplication
let page: Page
const target = join(tmpdir(), `spyde-export-${process.pid}.npz`)
const shots = join(__dirname, '..', 'export_numpy_shots')

test.describe.configure({ mode: 'serial' })
test.setTimeout(120_000)

// A .npz is a zip archive: every member starts with the local-file-header
// signature, and np.savez writes the first member at byte 0.
const ZIP_SIGNATURE = [0x50, 0x4b, 0x03, 0x04]

const dialogCalls = () =>
  app.evaluate(() => (globalThis as any).__saveDialogs as Array<{ defaultPath: string }>)

test.beforeAll(async () => {
  rmSync(target, { force: true })
  mkdirSync(shots, { recursive: true })
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  await app.evaluate(({ dialog }, path) => {
    ;(globalThis as any).__saveDialogs = []
    dialog.showSaveDialog = (async (_owner: unknown, options: { defaultPath: string }) => {
      ;(globalThis as any).__saveDialogs.push({ defaultPath: options.defaultPath })
      return { canceled: false, filePath: path }
    }) as typeof dialog.showSaveDialog
  }, target)
  await page.evaluate(() => window.electron.action('load_test_data', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 60_000 })
})

test.afterAll(async () => {
  await app?.close()
  rmSync(target, { force: true })
})

test('Export to NumPy writes the focused window through the Save dialog', async () => {
  // Focus the signal window: the menu names no window, the backend exports
  // whichever is active.
  await raiseWindow(sigWindow(page))

  await page.getByTestId('menu-file').click()
  const row = page.getByTestId('menu-export-numpy')
  await expect(row).toBeVisible()
  await page.screenshot({ path: join(shots, '01-file-menu.png') })
  await row.click()

  await expect.poll(() => existsSync(target), { message: 'no file was written', timeout: 30_000 })
    .toBe(true)
  await expect.poll(() => readFileSync(target).length, { timeout: 10_000 }).toBeGreaterThan(100)
  expect([...readFileSync(target).subarray(0, 4)]).toEqual(ZIP_SIGNATURE)
  expect((await dialogCalls()).map((call) => call.defaultPath)).toEqual(['maps.npz'])

  // The status bar says what went out.
  await expect(page.getByText(/Exported \d+ arrays? to /)).toBeVisible({ timeout: 10_000 })
  await page.screenshot({ path: join(shots, '02-exported.png') })
})
