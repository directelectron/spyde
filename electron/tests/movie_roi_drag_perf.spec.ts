/**
 * movie_roi_drag_perf.spec.ts — sliding an INTEGRATING SPAN across a real 4k
 * in-situ .mrc movie, the case a user actually complained about.
 *
 * Distinct from region_drag_perf.spec.ts, which drags a 2-D ROI over a 4D-STEM
 * scan of 32² frames. There the frames are 2 KB and the reader owns decoded
 * blocks (sum_points); here each frame is 32 MB and the reader is BinaryReader,
 * which has no block to sum from — so the read goes through RegionIntegrator's
 * per-frame accumulate. That is the path that measured 660 ms/step: ~166 ms of
 * I/O and ~500 ms of single-threaded numpy (16 × acc(float32 64 MB) += frame,
 * plus the /n, rint, astype tail). See CLAUDE.md Live-Display §3.
 *
 * `mrc: true` matters. A da.from_array(numpy) movie resolves to
 * SourceArrayReader and never touches the code under test — it would measure a
 * different path and pass while the real one regressed.
 *
 * The span is driven through `test_region_scrub` (which sets the widget in DATA
 * coords and forces an update) rather than with the mouse. A pointer-driven
 * version of this spec spent its whole run measuring nothing: the navigator panel
 * is small, sub-frame pointer steps produced "+0-0" no-op reads, and an overshoot
 * past the last frame clamped all 16 points onto frame 39. Frame indices are
 * exact here, so a step is always exactly one frame.
 *
 * Run: npx playwright test tests/movie_roi_drag_perf.spec.ts --project=electron \
 *        --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, navWindow, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'movie_roi_shots'
// 4096² uint16 = 32 MB/frame; 40 frames = 1.25 GB written once to the temp dir
// and reused across runs. Big enough that the ROI's working set (16 × 32 MB)
// overflows the ArrayCache default, which is what used to thrash it to 0 hits.
const SIZE = 4096
const FRAMES = 40
// Slide one frame at a time, staying clear of the last frame so the span is never
// clamped (that degenerates to N copies of one index — handled, but a different
// path from the sliding window this measures).
const START = 2
const SLIDES = 16

test('integrating span slides smoothly across a 4k .mrc movie', async () => {
  test.setTimeout(600_000)

  const ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO', SPYDE_NAV_PROFILE: '1' },
  })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'load_test_data_movie',
      { size: SIZE, frames: FRAMES, mrc: true })
    await waitForSubwindowCount(page, 2, 300_000)
    await page.waitForTimeout(4_000)          // first frame + tile enable
    await page.screenshot({ path: `${SHOTS}/01-loaded.png` })

    await expect(navWindow(page)).toBeVisible()
    await expect(sigWindow(page)).toBeVisible()

    // Prove the reader is the one under test before measuring anything.
    const readerLines = backend.logBuffer.filter((l: string) =>
      l.includes('[NAV-PROFILE]'))
    console.log(`[movie-roi] pre-scrub NAV-PROFILE lines=${readerLines.length}`)

    const marker = backend.logBuffer.length
    const positions = Array.from({ length: SLIDES }, (_, i) => [START + i])
    const done = backend.waitForLog('test_region_scrub:', 300_000)
    await backendAction(page, 'test_region_scrub', { positions })
    const summary = await done
    console.log(`[movie-roi] ${summary.trim().slice(-160)}`)
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/02-after-scrub.png` })

    const sigShot = await sigWindow(page).screenshot(
      { path: `${SHOTS}/03-signal-panel.png` })
    expect(sigShot.byteLength).toBeGreaterThan(2_000)

    // ---- the numbers: only lines emitted DURING the scrub -----------------
    const logs: string[] = backend.logBuffer.slice(marker)
    const navLines = logs.filter((l: string) => l.includes('[NAV-PROFILE]'))
    const durations: number[] = []
    let regionServed = 0
    let maxPoints = 0
    let realSlides = 0
    let threaded = 0
    let serialFallback = 0
    let asyncSubmits = 0
    for (const l of navLines) {
      if (l.includes('async-submit')) { asyncSubmits++; continue }
      const m = l.match(/total=([\d.]+)ms/)
      if (m) durations.push(parseFloat(m[1]))
      const r = l.match(/array-cache region x(\d+)/)
      if (r) { regionServed++; maxPoints = Math.max(maxPoints, parseInt(r[1], 10)) }
      // "+0-0" is the window not having moved; those cost nothing and prove nothing.
      const inc = l.match(/incremental \+(\d+)-(\d+)/)
      if (inc && parseInt(inc[1], 10) + parseInt(inc[2], 10) > 0) realSlides++
      if (/ t\d+/.test(l)) threaded++
      if (/region x\d+ serial/.test(l)) serialFallback++
    }
    durations.sort((a, b) => a - b)
    const at = (q: number) => durations.length
      ? durations[Math.min(durations.length - 1, Math.floor(durations.length * q))]
      : NaN
    const med = at(0.5)

    console.log(`[movie-roi] NAV-PROFILE lines=${navLines.length} ` +
      `region-served=${regionServed} widest-span=${maxPoints} ` +
      `real-slides=${realSlides} threaded=${threaded} ` +
      `serial-fallback=${serialFallback} async=${asyncSubmits}`)
    console.log(`[movie-roi] backend nav reads: n=${durations.length} ` +
      `median=${med.toFixed(1)}ms p95=${at(0.95).toFixed(1)}ms`)
    const sample = navLines.filter((l) => /array-cache region/.test(l)).pop()
    if (sample) console.log(`[movie-roi] sample: ${sample.slice(-200)}`)

    // First, that the spec measured the thing at all — an earlier pointer-driven
    // version went green while every read was a no-op.
    expect(regionServed).toBeGreaterThan(0)       // not a single-point read
    expect(maxPoints).toBeGreaterThanOrEqual(8)   // a real integrating window
    expect(realSlides).toBeGreaterThan(3)         // the window actually moved
    expect(serialFallback).toBe(0)                // the integrator served them
    // Then the cost. The floor is the ~500 ms of single-threaded accumulate the
    // integrator removes, so 250 ms leaves room for a slower machine while still
    // failing if the threading or the running-sum reuse is lost.
    if (durations.length) expect(med).toBeLessThan(250)
    assertNoJsErrors()
  } finally {
    await ctx.app.close().catch(() => {})
  }
})
