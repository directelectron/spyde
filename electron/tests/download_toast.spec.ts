/**
 * download_toast.spec.ts — the Examples-download notification cards.
 *
 * Renderer-only (SPYDE_NO_DASK + injected PLOTAPP messages, the
 * signal_type.spec idiom): `download_progress` shows a bottom-right toast with
 * a determinate progress bar and a Cancel button; Cancel dispatches the
 * `download_cancel` action with the download's token; `download_done` removes
 * the card. The backend side (pooch hook, cancel abort) is covered by
 * spyde/tests/migrated/test_example_download.py.
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
    ipcMain.on('spyde:action', (_e, action, payload, windowId) => {
      ;(globalThis as any).__sent.push({ action, payload, windowId })
    })
  })
}
const sentActions = () => app.evaluate(() => (globalThis as any).__sent)

const TOKEN = 'example:zrnb_precipitate'

test('progress message shows a toast with a moving bar; done removes it', async () => {
  // No toast before any download.
  await expect(page.getByTestId('download-toasts')).toHaveCount(0)

  await inject({ type: 'download_progress', token: TOKEN,
    label: 'zrnb_precipitate', done: 0, total: 668_000_000 })
  const toast = page.getByTestId('download-toast-zrnb_precipitate')
  await expect(toast).toBeVisible()
  await expect(toast).toContainText('Downloading zrnb_precipitate')
  await expect(toast).toContainText('0.00 / 668 MB (0%)')

  // Progress advances the bar + readout.
  await inject({ type: 'download_progress', token: TOKEN,
    label: 'zrnb_precipitate', done: 334_000_000, total: 668_000_000 })
  await expect(toast).toContainText('334 / 668 MB (50%)')
  // The fill animates via a 200ms width transition — poll until it has moved.
  await expect.poll(() => page.getByTestId('download-bar-zrnb_precipitate')
    .locator('div').evaluate((el) => (el as HTMLElement).getBoundingClientRect().width),
  { timeout: 3_000, message: 'progress bar never filled' }).toBeGreaterThan(10)
  await page.screenshot({ path: 'download_toast_shots/01-progress.png' })

  // Done → card disappears.
  await inject({ type: 'download_done', token: TOKEN, ok: true, cancelled: false })
  await expect(page.getByTestId('download-toasts')).toHaveCount(0)
})

test('Cancel dispatches download_cancel with the token', async () => {
  await trackActions()
  await inject({ type: 'download_progress', token: TOKEN,
    label: 'zrnb_precipitate', done: 100_000_000, total: 668_000_000 })
  const cancel = page.getByTestId('download-cancel-zrnb_precipitate')
  await expect(cancel).toBeVisible()
  await cancel.click()
  await expect(cancel).toBeDisabled()          // fire-once: greys out
  await expect(cancel).toContainText('Cancelling')
  await expect.poll(async () => (await sentActions()).map((s: any) => s.action))
    .toContain('download_cancel')
  const sent = (await sentActions()).find((s: any) => s.action === 'download_cancel')
  expect(sent.payload).toEqual({ token: TOKEN })
  await page.screenshot({ path: 'download_toast_shots/02-cancelling.png' })

  // Backend confirms (cancelled) → card disappears.
  await inject({ type: 'download_done', token: TOKEN, ok: false, cancelled: true })
  await expect(page.getByTestId('download-toasts')).toHaveCount(0)
})
