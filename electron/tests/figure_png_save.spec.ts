/**
 * figure_png_save.spec.ts — a figure's own "Save PNG…" saves a file in SpyDE.
 *
 * A figure lives in an iframe, where anyplotlib cannot download: it posts the
 * image to the host instead. SpyDE announces that it saves PNGs itself, takes
 * the image from its own figure frames only, and writes it through a native
 * Save dialog (replaced here by one that answers with a temp path).
 */
import { test, expect, _electron as electron, ElectronApplication, Page, Frame } from '@playwright/test'
import { join } from 'path'
import { tmpdir } from 'os'
import { existsSync, readFileSync, rmSync } from 'fs'

let app: ElectronApplication
let page: Page
const target = join(tmpdir(), `spyde-figure-save-${process.pid}.png`)

test.describe.configure({ mode: 'serial' })
test.setTimeout(120_000)

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

async function signalFrame(): Promise<Frame> {
  const signal = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) }).first()
  const frame = await (await signal.locator('iframe').first().elementHandle())?.contentFrame()
  if (!frame) throw new Error('no signal figure frame')
  return frame
}

/** Open the figure's export menu and click "Save PNG…"; return the menu rows. */
async function saveFromMenu(frame: Frame): Promise<string[]> {
  await frame.evaluate(() =>
    (document.querySelector('[aria-label="Copy or save this figure"]') as HTMLElement).click())
  const rows = await frame.evaluate(() =>
    Array.from(document.querySelectorAll('[data-apl-menu] *'))
      .filter((element) => element.children.length === 0)
      .map((element) => element.textContent ?? ''))
  await frame.evaluate(() => {
    const row = Array.from(document.querySelectorAll('[data-apl-menu] *'))
      .find((element) => element.textContent === 'Save PNG…') as HTMLElement
    row.click()
  })
  return rows
}

const dialogCalls = () => app.evaluate(() => (globalThis as any).__saveDialogs as Array<{ defaultPath: string }>)

test.beforeAll(async () => {
  rmSync(target, { force: true })
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
  const frame = await signalFrame()
  await frame.waitForFunction(() => typeof (globalThis as any).__aplExportPNG === 'function')
})

test.afterAll(async () => {
  await app?.close()
  rmSync(target, { force: true })
})

test('Save PNG writes the figure through the Save dialog', async () => {
  await saveFromMenu(await signalFrame())
  await expect.poll(() => existsSync(target), { message: 'no file was written' }).toBe(true)
  expect([...readFileSync(target).subarray(0, 8)]).toEqual(PNG_SIGNATURE)
  const calls = await dialogCalls()
  expect(calls).toHaveLength(1)
  expect(calls[0].defaultPath).toMatch(/apl-.*\.png$/)
  await expect(page.getByTestId('status-text')).toContainText(`Saved ${target}`)
})

test('an image that does not come from a figure frame is not saved', async () => {
  const before = (await dialogCalls()).length
  const dataUrl = readFileSync(target).toString('base64')
  await page.evaluate((png) => window.postMessage({
    type: 'anyplotlib_export_png_result', requestId: null,
    dataUrl: `data:image/png;base64,${png}`, filename: 'spoofed.png',
  }, '*'), dataUrl)
  await page.waitForTimeout(500)
  expect((await dialogCalls()).length).toBe(before)
})

test('the main process only writes PNG bytes, under a plain file name', async () => {
  const notPng = await page.evaluate(() =>
    window.electron.savePng(`data:image/png;base64,${btoa('not a png at all')}`, 'x.png'))
  expect(notPng).toMatchObject({ ok: false, error: 'not a PNG image' })

  const png = readFileSync(target).toString('base64')
  const saved = await page.evaluate((data) =>
    window.electron.savePng(`data:image/png;base64,${data}`, '../../outside/evil'), png)
  expect(saved.ok).toBe(true)
  const lastDefault = (await dialogCalls()).at(-1)!.defaultPath
  expect(lastDefault.endsWith('evil.png')).toBe(true)
  expect(lastDefault).not.toContain('outside')
})

// Needs the anyplotlib release with the host announcement (anyplotlib PR #78).
test('the figure shows no right-click preview and no second save entry', async () => {
  const frame = await signalFrame()
  const rows = await saveFromMenu(frame)
  expect(rows.some((row) => row.includes('choose folder'))).toBe(false)
  await page.waitForTimeout(300)
  expect(await frame.evaluate(() => document.body.innerText.includes('Save image as'))).toBe(false)
})
