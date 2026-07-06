/**
 * webgpu_image.spec.ts — the anyplotlib WebGPU 2-D large-image render path,
 * verified end-to-end in the real Electron app on the actual GPU.
 *
 * Playwright's bundled Chromium has NO WebGPU, but full Electron does (Pascal
 * GPU here). We load a real in-situ movie — whose signal window shows a 4096×4096
 * frame (>1 Mpx → above GPU_IMAGE_THRESHOLD) — and assert:
 *   1. navigator.gpu is present (sanity: the GPU path is even reachable),
 *   2. the signal panel's gpuCanvas activates (_gpu === 'active', canvas visible),
 *   3. it renders a NON-BLACK, colour-varying image (the shader-LUT actually ran),
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

test('WebGPU 2-D image path activates and renders a large movie frame', async () => {
  test.setTimeout(360_000)
  const moviePath = firstMovie()
  test.skip(!moviePath, 'no real in-situ movie file present')

  const { app, page, assertNoJsErrors } = await launchApp({ dask: true })
  try {
    // Sanity: the app's Chromium actually exposes WebGPU (else the whole test is
    // moot and should FAIL loudly, not silently pass on the canvas fallback).
    const gpu = await page.evaluate(async () => {
      if (!navigator.gpu) return { gpu: false }
      const a = await navigator.gpu.requestAdapter()
      return { gpu: true, adapter: !!a, arch: a?.info?.architecture || '?' }
    })
    console.log('WebGPU available:', JSON.stringify(gpu))
    expect(gpu.gpu, 'Electron must expose navigator.gpu').toBeTruthy()
    expect(gpu.adapter, 'a GPU adapter must resolve').toBeTruthy()

    await page.waitForTimeout(1500)
    await backendAction(page, 'open_file', { path: moviePath })
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 180_000 },
    )
    await page.waitForTimeout(3000)
    // Scrub to a mid-movie frame so a REAL 4k frame paints (the t=0 frame can be
    // dark/placeholder). draw2d then runs → the async GPU device resolves → the
    // panel flips to GPU on the next paint. Scrub twice with a settle so the
    // GPU-active redraw definitely happens after activation.
    await backendAction(page, 'test_nav_drag', { targets: [[400, 0], [200, 0]] })
    await page.waitForTimeout(4000)

    await page.screenshot({ path: 'webgpu_shots/01-gpu-image.png' })

    // figure_esm.js sets globalThis.__apl_gpu2d[panelId] = {active, iw, ih}
    // inside the figure iframe when a 2-D image GPU draw actually runs. Read it
    // from every same-origin figure iframe — the definitive activation signal.
    // The figure iframes are cross-origin (spyde-fig://), so we can't read into
    // them from the top page. Playwright CAN execute inside each frame directly.
    const gpuReport: any[] = []
    for (const fr of page.frames()) {
      try {
        const rec = await fr.evaluate(() => {
          const g: any = (globalThis as any).__apl_gpu2d
          const build = (globalThis as any).__apl_build || null
          if (!g) return { build, panels: [] }
          return { build, panels: Object.keys(g).map((k) => ({ id: k, ...g[k] })) }
        })
        if (rec.build || rec.panels.length) {
          console.log(`frame ${fr.url().slice(0, 45)}: build=${rec.build}`,
            JSON.stringify(rec.panels))
          gpuReport.push(...rec.panels)
        }
      } catch { /* frame gone */ }
    }
    console.log('GPU 2-D report:', JSON.stringify(gpuReport, null, 2))

    // Activation is the reliable automated signal: __apl_gpu2d proves the WGSL
    // image draw actually ran for a large frame. (Direct pixel readback of a live
    // WebGPU swapchain canvas is unreliable under automation — it reads black
    // via drawImage; see FIGURE_ESM.md. The rendered image is instead verified by
    // the screenshot 01-gpu-image.png, which shows the correct colormapped frame.)
    const activePanel = gpuReport.find((r) => r.active && r.iw >= 1024 && r.ih >= 1024)
    expect(
      activePanel,
      'the anyplotlib 2-D image WebGPU path did not activate for the large frame',
    ).toBeTruthy()

    // Correctness via OFFSCREEN-TEXTURE readback (the reliable method — the live
    // swapchain canvas reads black under automation). __apl_gpuReadback re-renders
    // the active image panel into an offscreen RGBA texture and copies it to CPU;
    // we assert the shader-LUT produced a real, varied colormapped image.
    let rb: any = null
    for (const fr of page.frames()) {
      try {
        const has = await fr.evaluate(() => typeof (globalThis as any).__apl_gpuReadback === 'function')
        if (!has) continue
        rb = await fr.evaluate(async (pid) => {
          const px = await (globalThis as any).__apl_gpuReadback(pid, 24)
          if (!px) return null
          let mn = 255, mx = 0, nz = 0
          const lums = px.px.map((p: number[]) => 0.3*p[0] + 0.59*p[1] + 0.11*p[2])
          for (const L of lums) { if (L > 4) nz++; if (L < mn) mn = L; if (L > mx) mx = L }
          return { min: +mn.toFixed(1), max: +mx.toFixed(1),
                   nonzeroFrac: +(nz / lums.length).toFixed(2),
                   sample: px.px.slice(0, 3) }
        }, activePanel.id)
        if (rb) break
      } catch { /* frame gone */ }
    }
    console.log('GPU offscreen readback:', JSON.stringify(rb))
    expect(rb, 'offscreen GPU readback returned nothing').toBeTruthy()
    expect(rb.nonzeroFrac, 'GPU-rendered image is mostly black').toBeGreaterThan(0.5)
    expect(rb.max - rb.min,
      'GPU image has no contrast — the shader LUT / clim did not run correctly',
    ).toBeGreaterThan(25)

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
