/** zoom_bench.spec.ts — measure draw2d cost of a zoom-only redraw on a big image. */
import { test } from '@playwright/test'
import { existsSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

const MOVIES = [
  'C:/Users/CarterFrancis/Downloads/20251117_88075_run3 some growth_1236_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20251117_88074_run1_9104_movie.mrc',
]
const movie = () => MOVIES.find((p) => existsSync(p)) || null

test('zoom bench', async () => {
  test.setTimeout(420_000)
  const path = movie()
  test.skip(!path, 'no movie')
  const { app, page } = await launchApp({ dask: true })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'open_file', { path })
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 180_000 })
    await page.waitForFunction(
      () => !/Reading .*\.mrc/i.test(document.body.textContent || ''),
      { timeout: 300_000 })
    await backendAction(page, 'test_nav_drag', { targets: [[200, 0]] })
    await page.waitForTimeout(3000)

    // Find the figure iframe with a GPU 2-D panel and time repeated zoom redraws.
    for (const fr of page.frames()) {
      try {
        const res = await fr.evaluate(async () => {
          const g: any = (globalThis as any).__apl_gpu2d
          const setZoom = (globalThis as any).__apl_setZoom
          if (!g || typeof setZoom !== 'function') return null
          const id = Object.keys(g)[0]
          const times: number[] = []
          // Sweep zoom in/out like a scroll gesture.
          for (let i = 0; i < 20; i++) {
            const z = 1 + (i % 10) * 0.3
            const ms = setZoom(id, z, 0.5, 0.5)
            if (typeof ms === 'number') times.push(ms)
          }
          times.sort((a, b) => a - b)
          const mean = times.reduce((x, y) => x + y, 0) / times.length
          // The per-tick serialise cost the wheel handler pays AFTER draw2d.
          // dbg.stringify/len = OLD full-state write (geom-polluted);
          // dbg.viewStringify/viewLen = NEW _viewStateJson (geom stripped).
          // On the binary transport dbg.hasBytes is true and the OLD stringify
          // of the Uint8Array (a {"0":..} object) is the zoom-lag culprit.
          const dbg = (globalThis as any).__apl_zoom_dbg || null
          return { id, n: times.length, mean: +mean.toFixed(1),
                   median: +times[times.length >> 1].toFixed(1),
                   max: +times[times.length - 1].toFixed(1),
                   gpuActive: !!(g[id] && g[id].active),
                   serialize: dbg && {
                     oldMs: +(+dbg.stringify).toFixed(2), oldLen: dbg.len,
                     newMs: +(+dbg.viewStringify).toFixed(2), newLen: dbg.viewLen,
                     hasBytes: dbg.hasBytes } }
        })
        if (res) { console.log('ZOOM draw2d:', JSON.stringify(res)); break }
      } catch { /* frame gone */ }
    }
  } finally {
    await app.close()
  }
})
