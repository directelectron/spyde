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
  // Scope by marker text, not bare row counts: this launch runs the REAL
  // backend, and a late startup record can stream into the ring at any moment
  // — CI once saw 2 rows here with only one injected. The clear contract is
  // exactly "rows from before the click hide, rows from after it show", which
  // the markers assert without racing unrelated traffic. (Same reason there is
  // no log-empty check: a real record arriving after clear legitimately shows.)
  const oldRow = page.locator('[data-testid="log-row"]', { hasText: 'old line' })
  const freshRow = page.locator('[data-testid="log-row"]', { hasText: 'fresh line' })
  await inject(logMsg('INFO', 'spyde.x', 'old line', 100))   // ancient → cleared
  await expect(oldRow).toHaveCount(1)
  await page.click('[data-testid="log-clear"]')
  await expect(oldRow).toHaveCount(0)
  await inject(logMsg('INFO', 'spyde.x', 'fresh line'))      // time≈now → shown
  await expect(freshRow).toHaveCount(1)
  await expect(page.locator('[data-testid="log-body"]')).toContainText('fresh line')
})

test('status-bar badge counts warnings/errors while the log is hidden', async () => {
  await inject(logMsg('INFO', 'spyde.x', 'quiet'))
  await inject(logMsg('WARNING', 'spyde.y', 'a warning'))
  await inject(logMsg('ERROR', 'spyde.z', 'an error'))
  await expect(page.locator('[data-testid="log-badge"]')).toHaveText('2')
})

/**
 * PERF REGRESSION GUARD. The panel used to render EVERY buffered record, so each
 * incoming line re-ran ~1000 row components and reconciled ~5000 DOM nodes —
 * measured 5.4 ms of main-thread time per line once the buffer hit its 1000 cap
 * (vs 0.05 ms now), which is why the app got sluggish "around 1000 log lines".
 *
 * Two independent things are asserted, because either one regressing brings the
 * cost back:
 *   1. the MOUNTED row count stays bounded by the viewport, not the buffer
 *      (virtualisation — LogPanel's useWindowed)
 *   2. appends are COALESCED, so N records cost far fewer than N React commits
 *      (SpyDEContext's queueLog)
 * The wall-time bound is deliberately loose (a shared CI box is noisy); the two
 * structural assertions are the sharp ones and cannot flake.
 */
test('PERF: 2000 log lines stay bounded in DOM and cost bounded main-thread time', async () => {
  test.setTimeout(120_000)
  await page.click('[data-testid="toggle-log"]')
  await expect(page.locator('[data-testid="log-panel"]')).toBeVisible()
  // The open triggers a backend log_backfill that REPLACES the buffer; let it
  // land before injecting or it wipes the injected records mid-run.
  await page.waitForTimeout(1500)

  const result = await page.evaluate(async () => {
    const inject = (window as any)._spyde_test_inject
    const yieldTask = () => new Promise<void>((r) => {
      const ch = new MessageChannel()
      ch.port1.onmessage = () => r()
      ch.port2.postMessage(0)
    })
    const N = 2000
    const t0 = performance.now()
    for (let i = 0; i < N; i++) {
      inject({
        type: 'log', level: i % 7 === 0 ? 'WARNING' : 'INFO',
        name: 'spyde.drawing.update_functions', area: 'navigator',
        msg: '[NAV-PROFILE] read=1.8ms levels=0.4ms transport=6.2ms idx=' + i,
        time: Date.now() / 1000,
      })
      await yieldTask()
      await yieldTask()
    }
    const elapsed = performance.now() - t0
    // Let the final coalesced batch flush + commit.
    await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())))
    const body = document.querySelector('[data-testid="log-body"]')
    return {
      elapsed,
      msPerLine: elapsed / N,
      rows: document.querySelectorAll('[data-testid="log-row"]').length,
      bodyNodes: body ? body.querySelectorAll('*').length : 0,
    }
  })

  // Virtualised: the ~220 px body shows ~13 rows; overscan and a taller panel
  // give plenty of head-room, but 2000 buffered records must never all mount.
  expect(result.rows).toBeGreaterThan(0)
  expect(result.rows).toBeLessThan(120)
  expect(result.bodyNodes).toBeLessThan(700)
  // Wall time per injected line. Measured ~0.05 ms/line with the fix and
  // ~5.4 ms/line without it, so 1.5 ms leaves a 30× cushion for a loaded box
  // while still failing hard if the whole buffer starts rendering again.
  expect(result.msPerLine).toBeLessThan(1.5)
})

test('PERF: a burst of records is coalesced into far fewer React renders', async () => {
  test.setTimeout(120_000)
  await page.click('[data-testid="toggle-log"]')
  await expect(page.locator('[data-testid="log-panel"]')).toBeVisible()
  await page.waitForTimeout(1500)

  // Count renders by observing the log body's DOM instead of hooking React: one
  // commit that changes the visible rows produces one batch of mutation records.
  const renders = await page.evaluate(async () => {
    const inject = (window as any)._spyde_test_inject
    const body = document.querySelector('[data-testid="log-body"]')!
    let batches = 0
    const obs = new MutationObserver(() => { batches++ })
    obs.observe(body, { childList: true, subtree: true, characterData: true })
    // 400 records delivered back-to-back, each in its own task (how IPC arrives).
    for (let i = 0; i < 400; i++) {
      inject({
        type: 'log', level: 'INFO', name: 'spyde.x', area: 'navigator',
        msg: 'burst ' + i, time: Date.now() / 1000,
      })
      await new Promise<void>((r) => {
        const ch = new MessageChannel()
        ch.port1.onmessage = () => r()
        ch.port2.postMessage(0)
      })
    }
    await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())))
    obs.disconnect()
    return batches
  })

  // One dispatch per record would give ≥400 mutation batches. Coalescing per
  // animation frame caps it at roughly the number of frames the burst spanned.
  expect(renders).toBeLessThan(150)
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
