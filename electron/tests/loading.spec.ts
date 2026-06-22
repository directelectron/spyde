/**
 * loading.spec.ts — the status bar shows a spinner + the backend's message while
 * a file is being read, so a slow cold-cache open of a large file looks like
 * work in progress, not a hang. Renderer-only.
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

test('busy=true shows the spinner + read message; busy=false clears it', async () => {
  await inject({ type: 'loading', busy: true, text: 'Reading big.mrc… (first open of a large file can take a while)' })
  await expect(page.getByTestId('loading-spinner')).toBeVisible()
  await expect(page.getByTestId('status-text')).toContainText('Reading big.mrc')

  await inject({ type: 'loading', busy: false, text: '' })
  await expect(page.getByTestId('loading-spinner')).toHaveCount(0)
})
