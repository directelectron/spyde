/**
 * nav_drag_distributed.spec.ts — the DISTRIBUTED (real Dask) crosshair-drag path.
 *
 * Reproduces the reported bug: dragging the navigator crosshair on a LAZY +
 * real-Dask 4D dataset with MULTIPLE nav chunks — the diffraction pattern does
 * not update (or updates only sometimes) even though the futures run on Dask.
 *
 * This is the path selector.spec.ts (SPYDE_NO_DASK eager) does NOT cover: lazy
 * data → _get_cache_dask_chunk → get_inds future → write_shared_array future →
 * PlotUpdateWorker poll → _on_plot_ready → plot.update → set_data → trait →
 * state_update → renderer.
 *
 * Captures the backend's [REDRAW]/NAV-DEBUG trace (stderr, since the backend
 * routes logs there) so the test output shows exactly where frames are lost.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'

let app: ElectronApplication
let page: Page
const backendLines: string[] = []
let navDragResult: any = null

test.beforeAll(async () => {
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    // NO SPYDE_NO_DASK → real LocalCluster + client. SPYDE_NAV_TIMING on so the
    // per-frame NAV-DEBUG timing fires. DEBUG so [REDRAW] traces emit.
    env: { ...process.env, SPYDE_NAV_TIMING: '1', SPYDE_LOG_LEVEL: 'DEBUG' },
  })
  let daskReady = false
  const grab = (d: Buffer) => {
    const s = String(d)
    for (const ln of s.split('\n')) {
      if (ln.includes('[REDRAW]') || ln.includes('[plot-paint]') || ln.includes('Failed to update')) {
        backendLines.push(ln.replace(/^.*spyde/, 'spyde'))
      }
      if (ln.includes('cluster READY') || ln.includes('Dask cluster ready')) daskReady = true
      const i = ln.indexOf('PLOTAPP:')
      if (i >= 0) {
        try {
          const obj = JSON.parse(ln.slice(i + 'PLOTAPP:'.length))
          if (obj.type === 'nav_drag_result') navDragResult = obj
        } catch { /* */ }
      }
    }
  }
  app.process().stdout?.on('data', grab)
  app.process().stderr?.on('data', grab)

  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  // Bump the renderer log level to DEBUG (so the backend lifts its level too).
  await page.evaluate(() => window.electron.action('set_log_level', { level: 'DEBUG' })).catch(() => {})
  for (let i = 0; i < 100 && !daskReady; i++) await page.waitForTimeout(500)  // ≤50s
  await page.evaluate(() => window.electron.action('load_test_data_lazy_chunked', {}))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 60_000 },
  )
  await page.waitForTimeout(3000)   // initial nav fill + first DP
})

test.afterAll(async () => {
  // Dump the captured backend [REDRAW] trace so it's visible in the report.
  console.log(`\n===== BACKEND [REDRAW]/NAV-DEBUG TRACE (${backendLines.length} lines) =====`)
  for (const ln of backendLines.slice(-200)) console.log(ln)
  await app?.close()
})

async function figIdForWindow(isNav: boolean): Promise<string> {
  const subs = page.getByTestId('subwindow')
  const n = await subs.count()
  for (let i = 0; i < n; i++) {
    const title = (await subs.nth(i).getByTestId('subwindow-title').textContent()) || ''
    if (/navigator/i.test(title) === isNav) {
      const tid = await subs.nth(i).locator('iframe').getAttribute('data-testid')
      return (tid || '').replace('figure-', '')
    }
  }
  throw new Error(`no ${isNav ? 'navigator' : 'signal'} iframe`)
}

test('crosshair drag across chunks updates the DP on the distributed path', async () => {
  // Drive the crosshair server-side through the navigator's real selector (the
  // _test_nav_drag action), across the 24×24 grid (3×3 chunk grid) so most moves
  // cross a chunk boundary → real worker round-trips. The backend reports, per
  // move, whether the SIGNAL plot's painted data actually changed.
  const targets = [
    [2, 2], [10, 2], [18, 2], [18, 10], [18, 18], [10, 18], [2, 18], [2, 10], [12, 12],
  ]

  navDragResult = null
  // small settle so the worker/backend are idle before we drive the drag
  await page.waitForTimeout(500)
  const sent = await page.evaluate((t) => {
    try { window.electron.action('test_nav_drag', { targets: t }); return 'sent' }
    catch (e) { return 'send-failed: ' + String(e) }
  }, targets)
  console.log('action send:', sent)
  // Poll the shared capture (filled by the beforeAll stdout grabber).
  for (let i = 0; i < 180 && navDragResult === null; i++) await page.waitForTimeout(500)
  const result = navDragResult ?? { error: 'timeout' }

  console.log('\n===== NAV_DRAG_RESULT =====')
  console.log(JSON.stringify(result, null, 2))

  expect(result.error, `nav_drag failed: ${result.error}`).toBeFalsy()
  expect(result.changed,
    `DP only updated ${result.changed}/${result.total} moves on the distributed ` +
    `path — frames dropped before paint`).toBeGreaterThanOrEqual(targets.length - 1)
})
