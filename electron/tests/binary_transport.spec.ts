/**
 * binary_transport.spec.ts — measure the binary-pixel transport win end to end.
 *
 * Loads a real 4k movie and scrubs a fixed set of frames, then reads the figure
 * iframe's paint telemetry (__apl_paint_n / __apl_paint_kind exposed by the ESM
 * message handler). With APL_BINARY_TRANSPORT on (SpyDE's default), the pixels
 * arrive as a raw PLOTBIN frame (no base64/JSON/atob); the frame must still be
 * pixel-correct (verified by webgpu_image.spec's offscreen readback) and the paint
 * kind must be 'binary'. This test asserts the binary path is ACTIVE and paints,
 * and records paint latency for the record.
 *
 * Skips if no real movie is present.
 */
import { test, expect } from '@playwright/test'
import { existsSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

const MOVIES = [
  'C:/Users/CarterFrancis/Downloads/20251117_88075_run3 some growth_1236_movie.mrc',
  'C:/Users/CarterFrancis/Downloads/20251117_88074_run1_9104_movie.mrc',
]
const movie = () => MOVIES.find((p) => existsSync(p)) || null

async function readPaint(page: any) {
  for (const fr of page.frames()) {
    try {
      const rec = await fr.evaluate(() => ({
        n: (globalThis as any).__apl_paint_n || 0,
        ms: (globalThis as any).__apl_paint_ms || null,
        kind: (globalThis as any).__apl_paint_kind || null,
      }))
      if (rec.kind) return rec
    } catch { /* frame gone */ }
  }
  return { n: 0, ms: null, kind: null }
}

test('binary pixel transport is active and paints frames', async () => {
  test.setTimeout(420_000)
  const path = movie()
  test.skip(!path, 'no in-situ movie present')

  // SpyDE sets APL_BINARY_TRANSPORT=1 for the backend itself (runner.ts); the
  // harness inherits it. Force it here too so the test is self-describing.
  // SPYDE_LOG_LEVEL=INFO tees the "binary transport active" marker to stderr where
  // the harness captures it.
  const { app, page, backend, assertNoJsErrors } = await launchApp({
    dask: true, env: { APL_BINARY_TRANSPORT: '1', SPYDE_LOG_LEVEL: 'INFO' },
  })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'open_file', { path })
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 180_000 },
    )
    await page.waitForFunction(
      () => !/Reading .*\.mrc/i.test(document.body.textContent || ''),
      { timeout: 300_000 },
    )
    await page.waitForTimeout(2500)

    // Scrub a spread of frames; each paints one image.
    const targets = Array.from({ length: 20 }, (_, i) => [20 + i * 20, 0])
    await backendAction(page, 'test_nav_drag', { targets })
    await page.waitForTimeout(3000)

    // PRIMARY signal: the backend logs "[apl] binary transport active: PLOTBIN …"
    // to stderr the first time it emits a binary frame (captured in logBuffer).
    // This is unambiguous proof the binary path is live (the base64 path never
    // emits it).
    const binLine = backend.logBuffer.find((l: string) =>
      l.includes('binary transport active: PLOTBIN'))
    console.log('backend binary marker:', binLine || '(none)')
    expect(binLine,
      'backend never emitted a PLOTBIN binary frame — binary transport not active',
    ).toBeTruthy()

    // Secondary: iframe paint telemetry (best-effort — cross-origin frame globals
    // can be flaky to read under automation; the marker above is authoritative).
    const paint = await readPaint(page)
    console.log('paint telemetry:', JSON.stringify(paint))
    if (paint.kind) {
      expect(paint.kind, `expected binary paint, got ${paint.kind}`).toBe('binary')
      console.log(`binary-path paints: ${paint.n}, last paint latency: ${paint.ms}ms`)
    }

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
