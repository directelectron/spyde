/**
 * tiered_nav_read.spec.ts — verify the TIERED navigator read on a REAL large
 * in-situ movie (the user's case: "if the integration area is large everything
 * freezes … same with rebinning").
 *
 * Cheap reads (single-frame scrub) stay synchronous; EXPENSIVE reads (a large
 * integrating region) go through ComputeBackend.submit_graph OFF the dispatcher,
 * cancellable on supersede — so the navigator never freezes. The integrating
 * region is also capped to <=16 nav positions per dimension on the widget
 * geometry (it physically stops growing).
 *
 * Uses the shared harness (real Dask) + backend test actions (test_nav_drag,
 * test_region_scrub) + stderr log capture — robust, unlike the iframe-image-sig
 * path. Skips (not fails) if no real movie is present.
 */
import { test, expect } from '@playwright/test'
import { existsSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

const MOVIE_CANDIDATES = [
  'C:/Users/CarterFrancis/Downloads/20251117_88075_run3 some growth_1236_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20251117_88074_run1_9104_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20241002_07954_movie.mrc',
]
function firstMovie(): string | null {
  for (const p of MOVIE_CANDIDATES) if (existsSync(p)) return p
  return null
}

test('tiered read: cheap scrub + async region (capped) never freezes the navigator', async () => {
  test.setTimeout(600_000)
  const moviePath = firstMovie()
  test.skip(!moviePath, 'no real in-situ movie file present on this machine')

  const { app, page, backend, assertNoJsErrors } = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'DEBUG' },
  })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'open_file', { path: moviePath })
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 240_000 },
    )
    await page.waitForTimeout(3000)
    await page.screenshot({ path: 'tiered_nav_shots/00-loaded.png' })

    // ── (1) CHEAP TIER: single-frame scrub still paints a fresh frame per move ──
    const targets: number[][] = [[1, 0], [30, 0], [120, 0], [60, 0]]
    const cheapVerdictP = backend.waitForLog('test_nav_drag:', 180_000)
    await backendAction(page, 'test_nav_drag', { targets })
    const cheapLine = await cheapVerdictP
    await page.screenshot({ path: 'tiered_nav_shots/01-cheap-scrub.png' })
    console.log('\n===== CHEAP SCRUB VERDICT =====\n' + cheapLine)
    const cm = cheapLine.match(/test_nav_drag:\s*(\d+)\/(\d+)\s+moves changed/)
    expect(cm, `no cheap verdict in: ${cheapLine}`).toBeTruthy()
    expect(Number(cm![1]), 'cheap single-frame scrub did not repaint per move')
      .toBeGreaterThanOrEqual(Number(cm![2]) - 1)

    // ── (2) EXPENSIVE TIER: a large integrating region (async submit_graph) ────
    // Sets an OVERSIZED span (must clamp to <=16), scrubs it, reports whether the
    // DP repainted (an ndarray landed via the async callback → no freeze).
    const regionVerdictP = backend.waitForLog('test_region_scrub:', 180_000)
    await backendAction(page, 'test_region_scrub', {})
    const regionLine = await regionVerdictP
    await page.screenshot({ path: 'tiered_nav_shots/02-region-scrub.png' })
    console.log('\n===== REGION SCRUB VERDICT =====\n' + regionLine)

    const rm = regionLine.match(/test_region_scrub:\s*(\d+)\/(\d+)\s+region moves painted; clamp=(\{[^}]*\}) cap=(\d+)/)
    expect(rm, `no region verdict in: ${regionLine}`).toBeTruthy()
    const painted = Number(rm![1])
    const totalR = Number(rm![2])
    const cap = Number(rm![4])
    // Every region move must paint (async read landed) — the navigator did NOT
    // freeze on a multi-second region compute.
    expect(painted, `region scrub painted only ${painted}/${totalR} — froze/stalled`)
      .toBeGreaterThanOrEqual(totalR - 1)

    // The extent cap clamped the (oversized) span/rectangle to <= cap indices.
    const clamp = JSON.parse(rm![3].replace(/'/g, '"'))
    const extent = clamp.span ?? Math.max(clamp.w ?? 0, clamp.h ?? 0)
    console.log(`region clamp extent=${extent} cap=${cap}`)
    expect(extent, `region extent ${extent} exceeds the cap ${cap}`)
      .toBeLessThanOrEqual(cap + 1e-6)

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
