/**
 * binary_pixel_transport.spec.ts — verify the RAW-uint8 side-channel renders
 * correctly in the real app (fast, synthetic data — no 16 GB movie).
 *
 * With APL_BINARY_TRANSPORT=1, Plot2D.set_data stashes raw bytes + a token and
 * ships pixels over PLOTBIN. This is a CORRECTNESS guard: after a navigator
 * scrub the signal panel must show a NON-BLANK, CHANGING image (proving the
 * token→side-table→PLOTBIN→renderer path actually delivered pixels), with no JS
 * errors. The 2.2x perf win itself is measured in anyplotlib's isolated bench;
 * here we only prove the wiring works end-to-end in Electron.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

test('binary pixel transport renders a live signal frame', async () => {
  test.setTimeout(120_000)
  mkdirSync('firstpaint_shots', { recursive: true })
  const { app, page, backend } = await launchApp({
    dask: false,
    env: { APL_BINARY_TRANSPORT: '1', SPYDE_LOG_LEVEL: 'WARNING' },
  })
  try {
    await page.waitForTimeout(1500)
    await backendAction(page, 'load_test_data')
    // Two panels: navigator + signal.
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 60_000 })
    await page.waitForTimeout(1500)

    // Sample the SIGNAL panel specifically (NOT the navigator, which is
    // constant across a scrub) by screenshotting the "Signal" subwindow and
    // hashing its pixels + counting the non-black area. Screenshotting the
    // parent-page element (rather than reading a canvas inside the figure
    // iframe) isolates the right panel and sidesteps cross-frame canvas access.
    // A working binary transport paints a NON-BLANK frame on first paint, and a
    // navigator scrub changes WHICH diffraction pattern the signal shows → the
    // hash changes. NB the FIRST-paint (before-scrub) sample is the guard for
    // the first-paint race this spec exists to catch: without the fix the signal
    // panel was permanently blank (nonBlack ≈ 0) until an organic second paint.
    const signalBox = async () => {
      const el = page.locator(
        '[data-testid="subwindow"]:has([data-testid="subwindow-title"]:text-is("Signal"))',
      )
      await el.first().waitFor({ state: 'visible', timeout: 30_000 })
      return el.first()
    }
    const sampleSignal = async (): Promise<{ hash: string; nonBlack: number }> => {
      const png = await (await signalBox()).screenshot()
      return await page.evaluate(async (b64: string) => {
        const img = await new Promise<HTMLImageElement>((res, rej) => {
          const i = new Image(); i.onload = () => res(i); i.onerror = rej
          i.src = 'data:image/png;base64,' + b64
        })
        const cv = document.createElement('canvas')
        cv.width = img.width; cv.height = img.height
        const ctx = cv.getContext('2d')!
        ctx.drawImage(img, 0, 0)
        const d = ctx.getImageData(0, 0, cv.width, cv.height).data
        let s = 0, nonBlack = 0
        for (let i = 0; i < d.length; i += 4) {
          s = (s * 31 + d[i] + d[i + 1] + d[i + 2]) >>> 0
          // Count clearly-lit pixels; the dark subwindow chrome/background
          // stays below this floor so only real image content contributes.
          if (d[i] + d[i + 1] + d[i + 2] > 90) nonBlack++
        }
        return { hash: String(s), nonBlack }
      }, png.toString('base64'))
    }

    const before = await sampleSignal()
    // Evidence: the signal panel painted on FIRST paint with NO interaction
    // (the first-paint race is fixed — this was permanently blank before).
    await page.screenshot({ path: 'firstpaint_shots/binary-transport-first-paint.png' })
    // Move the navigator to several distinct positions.
    await backendAction(page, 'test_nav_drag', { targets: [[7, 7], [0, 7], [3, 3]] })
    await page.waitForTimeout(1500)
    const after = await sampleSignal()

    console.log('signal signature before/after:', JSON.stringify(before), JSON.stringify(after))
    // FIRST-PAINT guard: the signal panel must show real content BEFORE any
    // interaction (this is the first-paint race the spec exists to catch).
    expect(before.nonBlack, 'signal panel blank on FIRST paint (no interaction)').toBeGreaterThan(20)
    expect(after.nonBlack, 'signal panel blank after scrub').toBeGreaterThan(20)
    expect(after.hash, 'signal frame did not change across a navigator scrub').not.toBe(before.hash)

    // No BACKEND errors surfaced to the log. `logBuffer` is an ARRAY of raw
    // lines (harness API), so filter it directly — no .split. Exclude the benign
    // Electron dev-mode "Security Warning (Insecure Content-Security-Policy)"
    // renderer console notice, which carries the RENDERER-ERROR tag but is not a
    // real error (present on every dev-server run); match Python backend errors
    // (a real ERROR log line or a Traceback) instead.
    const errs = (backend.logBuffer as string[]).filter(
      (l) => /Traceback|\bERROR\b/.test(l) && !/Security Warning|Content-Security-Policy/.test(l),
    )
    expect(errs, `backend errors:\n${errs.join('\n')}`).toHaveLength(0)
  } finally {
    await app.close()
  }
})
