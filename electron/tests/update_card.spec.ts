/**
 * update_card.spec.ts — the bottom-left "new version available" UpdateCard.
 *
 * Renderer-only (SPYDE_NO_DASK). e2e can't reach real GitHub, so we drive the
 * card by pushing update-status over the REAL IPC channel the updater uses:
 * `webContents.send('spyde:update-status', …)` from the main process via
 * app.evaluate (the same channel updater.ts's sendStatus writes to). That
 * exercises preload's onUpdateStatus → UpdateCard exactly like production.
 *
 * Asserts: available → card bottom-left with Download; error → friendly message
 * + Retry (a state the OLD banner dropped entirely); downloading → progress bar;
 * downloaded → Restart; dismiss hides it; a NEW status re-shows after dismiss.
 * Button clicks are verified against the ipcRenderer channels they fire on.
 *
 * The Help → Check-for-Updates dialog + channel persistence live in
 * update_gpu_dialogs.spec.ts (kept passing separately).
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

/** Push an update-status over the real IPC channel (updater.ts's sendStatus
 *  writes to this exact channel). Reaches preload's onUpdateStatus → UpdateCard. */
async function pushStatus(status: Record<string, unknown>) {
  await app.evaluate(({ BrowserWindow }, s) => {
    const w = BrowserWindow.getAllWindows()[0]
    w?.webContents.send('spyde:update-status', s)
  }, status)
}

/** Track the raw ipcRenderer channels the card's buttons fire on (download /
 *  install / re-check are ipcRenderer.send, NOT spyde:action). */
async function trackUpdateChannels() {
  await app.evaluate(({ ipcMain }) => {
    ;(globalThis as any).__upd = []
    for (const ch of ['spyde:download-update', 'spyde:quit-and-install', 'spyde:check-for-updates']) {
      ipcMain.removeAllListeners(ch)
      ipcMain.on(ch, () => { (globalThis as any).__upd.push(ch) })
    }
  })
}
const firedChannels = () => app.evaluate(() => (globalThis as any).__upd ?? [])

test('available → card bottom-left with the version and a Download button', async () => {
  await expect(page.getByTestId('update-card')).toHaveCount(0)

  await pushStatus({ state: 'available', version: '9.9.9' })
  const card = page.getByTestId('update-card')
  await expect(card).toBeVisible()
  await expect(card).toContainText('SpyDE 9.9.9 is available')

  // Pinned bottom-left (mirror of the bottom-right DownloadToasts). Electron's
  // page.viewportSize() is null (no emulated viewport), so read the window size
  // from the DOM and use the card's own rect.
  const { x, bottom, w, h } = await card.evaluate((el) => {
    const r = el.getBoundingClientRect()
    return { x: r.left, bottom: r.bottom, w: window.innerWidth, h: window.innerHeight }
  })
  expect(x).toBeLessThan(w / 2)          // left half
  expect(bottom).toBeGreaterThan(h / 2)  // bottom half

  await expect(page.getByTestId('update-card-download')).toBeVisible()
  await page.screenshot({ path: 'update_card_shots/01-available.png' })
})

test('Download button fires the download IPC', async () => {
  await trackUpdateChannels()
  await pushStatus({ state: 'available', version: '9.9.9' })
  await page.getByTestId('update-card-download').click()
  await expect.poll(firedChannels).toContain('spyde:download-update')
})

test('downloading → progress bar + percent', async () => {
  await pushStatus({ state: 'downloading', percent: 42 })
  const card = page.getByTestId('update-card')
  await expect(card).toBeVisible()
  await expect(card).toContainText('42%')
  await expect(page.getByTestId('update-card-bar')).toBeVisible()
  // The fill width tracks the percent.
  const w = await page.getByTestId('update-card-bar').locator('div')
    .evaluate((el) => (el as HTMLElement).getBoundingClientRect().width)
  expect(w).toBeGreaterThan(5)
  await page.screenshot({ path: 'update_card_shots/02-downloading.png' })
})

test('downloaded → Restart to update fires quit-and-install', async () => {
  await trackUpdateChannels()
  await pushStatus({ state: 'downloaded', version: '9.9.9' })
  const restart = page.getByTestId('update-card-restart')
  await expect(restart).toBeVisible()
  await restart.click()
  await expect.poll(firedChannels).toContain('spyde:quit-and-install')
})

test('error → friendly message + Retry (the banner dropped this)', async () => {
  await trackUpdateChannels()
  await pushStatus({ state: 'error', message: 'Update check timed out — check your connection and try again.' })
  const card = page.getByTestId('update-card')
  await expect(card).toBeVisible()
  await expect(card).toContainText('Update failed')
  await expect(page.getByTestId('update-card-error'))
    .toContainText('check your connection and try again')

  const retry = page.getByTestId('update-card-retry')
  await expect(retry).toBeVisible()
  await page.screenshot({ path: 'update_card_shots/03-error.png' })

  await retry.click()
  await expect.poll(firedChannels).toContain('spyde:check-for-updates')
})

test('dismiss hides the card; a NEW status re-shows it', async () => {
  await pushStatus({ state: 'available', version: '9.9.9' })
  await expect(page.getByTestId('update-card')).toBeVisible()

  await page.getByTestId('update-card-dismiss').click()
  await expect(page.getByTestId('update-card')).toHaveCount(0)

  // Same status again does NOT re-show (no new status delivered while dismissed
  // only re-shows on a fresh delivery) — but a DIFFERENT status must re-show it,
  // so a later error is never dismissed away from its Retry for good.
  await pushStatus({ state: 'error', message: 'You appear to be offline — check your connection and try again.' })
  await expect(page.getByTestId('update-card')).toBeVisible()
  await expect(page.getByTestId('update-card-retry')).toBeVisible()
})

test('checking is shown but unobtrusive; idle/not-available render nothing', async () => {
  await pushStatus({ state: 'checking' })
  await expect(page.getByTestId('update-card-checking')).toBeVisible()

  await pushStatus({ state: 'not-available' })
  await expect(page.getByTestId('update-card')).toHaveCount(0)

  await pushStatus({ state: 'idle' })
  await expect(page.getByTestId('update-card')).toHaveCount(0)
})
