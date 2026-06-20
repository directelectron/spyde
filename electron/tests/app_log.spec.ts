/**
 * app_log.spec.ts — the Application Log panel: toggle, level switcher, streaming
 * records, clear, and the status-bar problem badge. Drives the panel with
 * injected backend `log` / `log_backfill` / `log_level` messages (no Dask).
 *
 * Self-contained (Node 23 + Playwright break on cross-file .ts imports).
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

function logMsg(level: string, name: string, m: string, time = Date.now() / 1000) {
  return { type: 'log', level, name, msg: m, time }
}

test('toggle shows the log panel and streams records by level', async () => {
  await page.click('[data-testid="toggle-log"]')
  await expect(page.locator('[data-testid="log-panel"]')).toBeVisible()

  await inject(logMsg('INFO', 'spyde.backend.session', 'Dask cluster ready'))
  await inject(logMsg('DEBUG', 'spyde.actions.find_vectors', 'VRAM probe: 8 GB'))
  await inject(logMsg('WARNING', 'spyde.drawing.update_functions', 'shm write retried'))
  await inject(logMsg('ERROR', 'spyde.actions.orientation_action', 'CIF parse failed'))

  await expect(page.locator('[data-testid="log-row"]')).toHaveCount(4)
  await expect(page.locator('[data-testid="log-row"][data-level="WARNING"]')).toHaveCount(1)
  await expect(page.locator('[data-testid="log-row"][data-level="ERROR"]')).toContainText('CIF parse failed')
})

test('level switcher reflects the backend-confirmed level', async () => {
  await page.click('[data-testid="toggle-log"]')
  await page.selectOption('[data-testid="log-level-select"]', 'DEBUG')
  // Backend confirms the new level (the controlled <select> follows state).
  await inject({ type: 'log_level', level: 'DEBUG' })
  await expect(page.locator('[data-testid="log-level-select"]')).toHaveValue('DEBUG')
})

test('backfill replaces the visible history', async () => {
  await page.click('[data-testid="toggle-log"]')
  await inject(logMsg('INFO', 'spyde.x', 'one'))
  await inject({
    type: 'log_backfill',
    entries: [
      logMsg('INFO', 'spyde.a', 'history A'),
      logMsg('WARNING', 'spyde.b', 'history B'),
    ],
  })
  await expect(page.locator('[data-testid="log-row"]')).toHaveCount(2)
  await expect(page.locator('[data-testid="log-body"]')).toContainText('history B')
})

test('clear hides current rows but keeps streaming new ones', async () => {
  await page.click('[data-testid="toggle-log"]')
  await inject(logMsg('INFO', 'spyde.x', 'old line', 100))   // ancient → cleared
  await expect(page.locator('[data-testid="log-row"]')).toHaveCount(1)
  await page.click('[data-testid="log-clear"]')
  await expect(page.locator('[data-testid="log-empty"]')).toBeVisible()
  await inject(logMsg('INFO', 'spyde.x', 'fresh line'))      // time≈now → shown
  await expect(page.locator('[data-testid="log-row"]')).toHaveCount(1)
  await expect(page.locator('[data-testid="log-body"]')).toContainText('fresh line')
})

test('status-bar badge counts warnings/errors while the log is hidden', async () => {
  await inject(logMsg('INFO', 'spyde.x', 'quiet'))
  await inject(logMsg('WARNING', 'spyde.y', 'a warning'))
  await inject(logMsg('ERROR', 'spyde.z', 'an error'))
  await expect(page.locator('[data-testid="log-badge"]')).toHaveText('2')
})

test('SCREENSHOT: populated application log for visual approval', async () => {
  await page.click('[data-testid="toggle-log"]')
  // Let the panel-open backfill round-trip with the (persistent) backend settle
  // first, then inject an authoritative backfill — so the curated lines are the
  // last write and the shot is deterministic regardless of backend chatter.
  await page.waitForTimeout(350)
  await inject({ type: 'log_level', level: 'DEBUG' })

  const lines = [
    logMsg('INFO', 'spyde.backend.session', 'Dask cluster ready — 7 workers, 2 threads each'),
    logMsg('INFO', 'spyde.backend.session', 'Loaded mgo_nanocrystals (64×64 nav · 128×128 sig)'),
    logMsg('DEBUG', 'spyde.actions.find_vectors', 'VRAM probe: 8.0 GB → GPU pool cap 4.0 GB'),
    logMsg('DEBUG', 'spyde.drawing.update_functions', 'cross-chunk move → routing via shared memory'),
    logMsg('INFO', 'spyde.actions.find_vectors_action', 'Found 5128 diffraction vectors'),
    logMsg('WARNING', 'spyde.dask_manager', 'worker tcp://127.0.0.1:51823 restarted'),
    logMsg('DEBUG', 'spyde.actions.vector_orientation_gpu', 'CUDA autograd warmup skipped (no CUDA)'),
    logMsg('INFO', 'spyde.actions.orientation_action', 'Orientation map complete — 4096 patterns'),
    logMsg('ERROR', 'spyde.actions.composition', 'COD search failed: HTTP 503 (will retry)'),
  ]
  await inject({ type: 'log_backfill', entries: lines })

  await expect(page.locator('[data-testid="log-body"]')).toContainText('Found 5128 diffraction vectors')
  await expect(page.locator('[data-testid="log-body"]')).toContainText('COD search failed')
  await expect(page.locator('[data-testid="log-row"]').first()).toBeVisible()
  await page.waitForTimeout(150)
  await page.screenshot({ path: join(__dirname, '..', 'app_log.png') })
})
