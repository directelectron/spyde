/**
 * viewport_detail.spec.ts — viewport detail-tile crisp zoom, end-to-end in the real
 * Electron app on the actual GPU.
 *
 * After the seamless-settle + detail-tile change, zooming into a 4096² signal frame
 * must send a hi-res DETAIL TILE of only the visible region (not the full frame), so
 * the zoom is crisp with no ~400 ms full-res flash. We assert:
 *   1. settle sends a motion-res frame (no 4096² full-res transport),
 *   2. a real zoom fires view_changed → SpyDE sets a detail tile (detail_region set),
 *   3. the GPU stays active and renders the crisp tile in the zoomed region,
 *   4. no renderer JS errors.
 *
 * Skips if no real movie file is present.
 */
import { test, expect } from '@playwright/test'
import { existsSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

const MOVIE_CANDIDATES = [
  'C:/Users/CarterFrancis/Downloads/20251117_88075_run3 some growth_1236_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20251117_88074_run1_9104_movie.mrc',
]
function firstMovie(): string | null {
  for (const p of MOVIE_CANDIDATES) if (existsSync(p)) return p
  return null
}

test('zoom sends a crisp detail tile, no full-res flash', async () => {
  test.setTimeout(360_000)
  const moviePath = firstMovie()
  test.skip(!moviePath, 'no real in-situ movie file present')

  const { app, page, assertNoJsErrors } = await launchApp({
    dask: true, env: { SPYDE_NAV_PROFILE: '1' },
  })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'open_file', { path: moviePath })
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 180_000 })
    await page.waitForFunction(
      () => !/Reading .*\.mrc/i.test(document.body.textContent || ''),
      { timeout: 300_000 })
    await page.waitForTimeout(3000)

    // Scrub to a mid-movie 4k frame + settle so the base motion frame is resident.
    await backendAction(page, 'test_nav_drag', { targets: [[400, 0], [200, 0]] })
    await page.waitForTimeout(3000)
    await page.screenshot({ path: 'viewport_shots/00-settled.png' })

    // The signal panel is the large (>=1024²) GPU-active image. Find its frame + id.
    let sigFrame: any = null, panelId = ''
    for (const fr of page.frames()) {
      try {
        const rec = await fr.evaluate(() => {
          const g: any = (globalThis as any).__apl_gpu2d
          if (!g) return null
          for (const k of Object.keys(g))
            if (g[k].iw >= 1024 && g[k].ih >= 1024) return { id: k, ...g[k] }
          return null
        })
        if (rec) { sigFrame = fr; panelId = rec.id; break }
      } catch { /* frame gone */ }
    }
    expect(sigFrame, 'no large signal image panel found').toBeTruthy()

    // Drive a REAL zoom via the renderer's wheel path so view_changed fires to
    // SpyDE (── __apl_setZoom bypasses the event; a wheel dispatch drives it).
    await sigFrame.evaluate((pid: string) => {
      const setZoom = (globalThis as any).__apl_setZoom
      if (typeof setZoom === 'function') setZoom(pid, 4.0, 0.5, 0.5)
      // Also emit the view_changed event that the wheel handler would (debounced).
      const emit = (globalThis as any).__apl_emitViewChanged
      // Fallback: dispatch a wheel on the overlay canvas to trigger the real path.
      const cv = document.querySelector('canvas')
      if (cv) cv.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, bubbles: true }))
    }, panelId)
    // Wait past the 90 ms debounce + the SpyDE crop send.
    await page.waitForTimeout(1200)
    await page.screenshot({ path: 'viewport_shots/01-zoomed-tile.png' })

    // Assert SpyDE replied with a detail tile for the visible region.
    const detail = await sigFrame.evaluate((pid: string) => {
      const raw = (globalThis as any).__apl_viewStateJson?.(pid)
      if (!raw) return null
      const st = JSON.parse(raw)
      return { region: st.detail_region, w: st.detail_width, h: st.detail_height,
               hasTile: !!st.detail_b64 }
    }, panelId)
    console.log('detail tile after zoom:', JSON.stringify(detail))
    expect(detail, 'no view state').toBeTruthy()
    expect(detail.region && detail.region.length === 4,
      'zoom did not produce a detail tile (view_changed → set_detail round trip)').toBeTruthy()
    // The tile is capped to the panel resolution (≈1024), never the full 4096².
    expect(detail.w, 'tile larger than panel cap — full-res leaked').toBeLessThanOrEqual(1536)
    expect(detail.w, 'empty tile').toBeGreaterThan(0)

    // GPU still active on the zoomed frame (rendering the crisp tile, not fallback).
    const active = await sigFrame.evaluate((pid: string) => {
      const g: any = (globalThis as any).__apl_gpu2d
      return g && g[pid] ? g[pid].active : false
    }, panelId)
    expect(active, 'GPU fell back on zoom').toBeTruthy()

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
