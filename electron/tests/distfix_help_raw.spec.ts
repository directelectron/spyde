/**
 * distfix_help_raw.spec.ts — the packaged-first-run diagnosability + GPU help
 * additions:
 *   1. Help → GPU & CUDA opens the static help dialog (screenshot).
 *   2. The Log panel's "Raw output" toggle switches to the raw stdout/stderr view.
 *   3. A backend-exited/env-setup failure surfaces the captured raw output in the
 *      blocking overlay (injected `backend_exited` with a `reason`) — screenshot.
 *
 * No Dask needed — pure UI wiring driven by the test inject hook. Self-contained
 * (Node 23 + Playwright break on cross-file .ts imports).
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

test('Help → GPU & CUDA opens the static help dialog', async () => {
  await page.click('[data-testid="menu-help"]')
  await page.click('[data-testid="menu-item-gpu-cuda"]')
  const dialog = page.locator('[data-testid="gpu-help-dialog"]')
  await expect(dialog).toBeVisible()
  await expect(dialog).toContainText('CUDA-12.4')
  await expect(dialog).toContainText('No separate CUDA Toolkit install is needed')
  await expect(dialog).toContainText('2.4 GiB')
  // The "Open GPU Status…" cross-link is present.
  await expect(page.locator('[data-testid="gpu-help-open-status"]')).toBeVisible()
  await page.screenshot({ path: join(__dirname, '..', 'distfix_shots', '01-gpu-help.png') })
  await page.click('[data-testid="gpu-help-close"]')
  await expect(dialog).toHaveCount(0)
})

test('triage section renders a machine verdict', async () => {
  await page.click('[data-testid="menu-help"]')
  await page.click('[data-testid="menu-item-gpu-cuda"]')
  await expect(page.locator('[data-testid="gpu-triage-section"]')).toBeVisible()
  const verdict = page.locator('[data-testid="gpu-triage-verdict"]')
  await expect(verdict).toBeVisible()
  // Wait for the REAL verdict: the backend's get_gpu_status may cold-import
  // torch (slow) and nvidia-smi runs on open — "Running checks…" is the
  // placeholder until both land.
  await expect(verdict).not.toContainText('Running checks', { timeout: 60000 })
  const text = (await verdict.textContent()) ?? ''
  console.log('[triage verdict]', text)
  expect(text.length).toBeGreaterThan(10)
  // This dev box: e2e runs the built bundle by path (app.isPackaged=false), so
  // whatever the hardware verdict is, a non-managed run must be report-only —
  // either no Fix button is offered (gpu-OK case) or it's disabled (dev note).
  const fixCount = await page.locator('[data-testid="gpu-triage-fix"]').count()
  if (fixCount > 0) {
    await expect(page.locator('[data-testid="gpu-triage-fix"]')).toBeDisabled()
  }
  await page.screenshot({ path: join(__dirname, '..', 'distfix_shots', '04-triage.png') })
  await page.click('[data-testid="gpu-help-close"]')
})

test('Log panel "Raw output" toggle switches the view', async () => {
  await page.click('[data-testid="toggle-log"]')
  await expect(page.locator('[data-testid="log-panel"]')).toBeVisible()
  // Default view is the structured application log.
  await expect(page.locator('[data-testid="log-panel"]')).toContainText('Application Log')
  // Flip to raw output.
  await page.click('[data-testid="log-raw-toggle"]')
  await expect(page.locator('[data-testid="log-panel"]')).toContainText('Raw Output')
  // The area/level controls are hidden in raw mode.
  await expect(page.locator('[data-testid="log-area-select"]')).toHaveCount(0)
  await expect(page.locator('[data-testid="log-level-select"]')).toHaveCount(0)
  await page.screenshot({ path: join(__dirname, '..', 'distfix_shots', '02-raw-output-view.png') })
  // Toggle back.
  await page.click('[data-testid="log-raw-toggle"]')
  await expect(page.locator('[data-testid="log-panel"]')).toContainText('Application Log')
})

test('env-setup failure surfaces raw output in the blocking overlay', async () => {
  // Simulate the packaged first-run env-setup failure the main process now
  // synthesises: a backend_exited message carrying a human-readable reason.
  await inject({
    type: 'backend_exited',
    code: null,
    reason:
      'Python environment setup failed. SpyDE could not build its analysis ' +
      'backend on first launch.\n\nuv sync exited with code 1',
  })
  const overlay = page.locator('[data-testid="backend-exited-overlay"]')
  await expect(overlay).toBeVisible()
  await expect(overlay).toContainText('Python environment setup failed')
  // The raw-output block is present (embeds whatever stdout/stderr was captured).
  await expect(page.locator('[data-testid="backend-exited-raw"]')).toBeVisible()
  await page.screenshot({ path: join(__dirname, '..', 'distfix_shots', '03-backend-exited-overlay.png') })
})
