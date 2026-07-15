/**
 * env_setup_overlay.spec.ts — the floating first-run "Setting up SpyDE" overlay.
 *
 * The overlay is normally driven by real `uv sync` output on a packaged first
 * launch (index.ts → env_setup messages). Here we inject those same messages
 * via the renderer's _spyde_test_inject hook (no packaged build, no download),
 * drive it through the phases, and SCREENSHOT each stage so the visual result
 * is actually inspected (per CLAUDE.md: pixels are the test for UI features).
 *
 * Self-contained (Node 23 + Playwright break on cross-file .ts imports).
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'

const SHOTS = 'env_setup_shots'
mkdirSync(SHOTS, { recursive: true })

let app: ElectronApplication
let page: Page

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_NO_DASK: '1' },
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  await page.waitForSelector('[data-testid="mdi-area"]')
})
test.afterAll(async () => { await app?.close() })

async function inject(msg: Record<string, unknown>) {
  await page.evaluate((m) => { (window as any)._spyde_test_inject?.(m) }, msg)
}

test('first-run setup overlay: appears, shows phases + step + %, live log, then clears', async () => {
  const overlay = page.locator('[data-testid="env-setup-overlay"]')

  // Not present before setup starts.
  await expect(overlay).toHaveCount(0)

  // ── start ──────────────────────────────────────────────────────────────
  await inject({ type: 'env_setup', event: 'start' })
  await expect(overlay).toBeVisible()
  await expect(page.locator('[data-testid="env-setup-step"]')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/01-start.png` })

  // ── resolving ──────────────────────────────────────────────────────────
  await inject({ type: 'env_setup', event: 'progress', phase: 'resolving',
    step: 'Resolving dependencies', percent: null, raw: 'Resolved 142 packages in 1.2s' })
  await expect(page.locator('[data-testid="env-setup-step"]'))
    .toContainText('Resolving dependencies')
  await expect(page.locator('[data-testid="env-setup-log"]'))
    .toContainText('Resolved 142 packages')

  // ── downloading torch WITH a percentage (the big, slow step) ─────────────
  for (const pct of [8, 34, 67, 92]) {
    await inject({ type: 'env_setup', event: 'progress', phase: 'torch',
      step: 'Downloading PyTorch', percent: pct,
      raw: `torch    ====>    ${pct}%  (${pct * 8} MiB/825 MiB)` })
  }
  await expect(page.locator('[data-testid="env-setup-step"]')).toContainText('Downloading PyTorch')
  await expect(page.locator('[data-testid="env-setup-step"]')).toContainText('92%')
  await page.screenshot({ path: `${SHOTS}/02-torch-download.png` })

  // ── installing ───────────────────────────────────────────────────────────
  await inject({ type: 'env_setup', event: 'progress', phase: 'installing',
    step: 'Installing 142 packages', percent: null, raw: 'Installed 142 packages in 800ms' })
  await expect(page.locator('[data-testid="env-setup-step"]')).toContainText('Installing 142 packages')
  await page.screenshot({ path: `${SHOTS}/03-installing.png` })

  // Log tail must be visibly growing (newest line present).
  await expect(page.locator('[data-testid="env-setup-log"]')).toContainText('Installed 142 packages')

  // ── done clears the overlay ──────────────────────────────────────────────
  await inject({ type: 'env_setup', event: 'done' })
  await expect(overlay).toHaveCount(0)
  await page.screenshot({ path: `${SHOTS}/04-cleared.png` })
})

test('a backend_exited (setup failure) supersedes the setup overlay', async () => {
  await page.reload()
  await page.waitForSelector('[data-testid="mdi-area"]')
  await inject({ type: 'env_setup', event: 'start' })
  await expect(page.locator('[data-testid="env-setup-overlay"]')).toBeVisible()

  // A packaged setup failure comes through as backend_exited with a reason.
  await inject({ type: 'backend_exited', code: null, reason: 'Python environment setup failed. …' })
  await expect(page.locator('[data-testid="env-setup-overlay"]')).toHaveCount(0)
  await expect(page.locator('[data-testid="backend-exited-overlay"]')).toBeVisible()
})
