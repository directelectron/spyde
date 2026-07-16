/**
 * dask_monitor.spec.ts — the StatusBar compute HUD (renderer-only, injected
 * `dask_stats` messages; the backend sampler is covered by
 * spyde/tests/migrated/test_dask_stats.py, and the live end-to-end readout is
 * asserted during a real compute in fv_neural_calibration.spec.ts).
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

const STATS = {
  type: 'dask_stats',
  workers: [
    { name: '0', cpu: 3, mem: 1e9, mem_limit: 8e9, executing: 0, ready: 0 },  // idle
    { name: '1', cpu: 97, mem: 2e9, mem_limit: 8e9, executing: 3, ready: 5 },
  ],
  tasks: { executing: 3, queued: 5 },
  gpu: { util: 96, vram_used: 3000, vram_total: 8192 },
  host_cpu: 88,
  host_mem: 52,
}

test('dask_stats shows the HUD; click opens the per-worker popover', async () => {
  // Hidden until stats flow.
  await expect(page.getByTestId('dask-monitor')).toHaveCount(0)

  await inject(STATS)
  const seg = page.getByTestId('dask-monitor')
  await expect(seg).toBeVisible()
  await expect(seg).toContainText('CPU 88%')
  await expect(seg).toContainText('MEM 52%')
  await expect(seg).toContainText('GPU 96%')
  await expect(seg).toContainText('8 tasks')

  await seg.click()
  const pop = page.getByTestId('dask-monitor-popover')
  await expect(pop).toBeVisible()
  await expect(pop).toContainText('2 workers, 3 running / 5 queued')
  await expect(pop).toContainText('tasks')                      // column legend
  await expect(page.getByTestId('dask-worker-1')).toContainText('97%')
  await expect(page.getByTestId('dask-worker-1')).toContainText('3+5')
  await expect(page.getByTestId('dask-worker-0')).toContainText('–')  // idle worker
  await expect(page.getByTestId('dask-gpu-row')).toContainText('2.9/8.0')
  await page.screenshot({ path: 'dask_monitor_shots/01-popover.png' })
})

test('idle cluster reads as idle; the HUD hides when samples stop', async () => {
  await inject({ ...STATS, tasks: { executing: 0, queued: 0 },
    workers: STATS.workers.map(w => ({ ...w, executing: 0, ready: 0, cpu: 3 })),
    gpu: { util: 0, vram_used: 500, vram_total: 8192 }, host_cpu: 5 })
  const seg = page.getByTestId('dask-monitor')
  await expect(seg).toContainText('idle')
  // Staleness sweep (7 s without a sample) hides the readout.
  await expect(seg).toHaveCount(0, { timeout: 12_000 })
})
