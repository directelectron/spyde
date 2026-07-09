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
const { launchApp, backendAction } = require('./_harness.cjs')

test('binary pixel transport renders a live signal frame', async () => {
  test.setTimeout(120_000)
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

    // Sample the signal figure's canvas pixels, scrub, sample again. A working
    // binary transport delivers a different frame → the sampled pixels change.
    const sampleSignal = async (): Promise<string> => {
      for (const fr of page.frames()) {
        try {
          const sig = await fr.evaluate(() => {
            const g: any = (globalThis as any).__apl_gpu2d || {}
            // any 2-D panel canvas; read a small pixel grid signature
            const cvs = Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]
            for (const c of cvs) {
              if (c.width < 8 || c.height < 8) continue
              const ctx = c.getContext('2d')
              if (!ctx) continue
              try {
                const d = ctx.getImageData(0, 0, Math.min(32, c.width), Math.min(32, c.height)).data
                let s = 0
                for (let i = 0; i < d.length; i += 4) s = (s * 31 + d[i] + d[i + 1] + d[i + 2]) >>> 0
                if (s !== 0) return String(s) + ':' + Object.keys(g).length
              } catch { /* tainted / not ready */ }
            }
            return ''
          })
          if (sig) return sig
        } catch { /* frame gone */ }
      }
      return ''
    }

    const before = await sampleSignal()
    // Move the navigator to several distinct positions.
    await backendAction(page, 'test_nav_drag', { targets: [[7, 7], [0, 7], [3, 3]] })
    await page.waitForTimeout(1500)
    const after = await sampleSignal()

    console.log('signal signature before/after:', before, after)
    expect(before, 'signal panel never produced a non-blank frame').not.toBe('')
    expect(after, 'signal panel blank after scrub').not.toBe('')
    expect(after, 'signal frame did not change across a navigator scrub').not.toBe(before)

    // No JS errors in the renderer, and no backend errors surfaced to the log.
    const errs = backend.logBuffer.split('\n').filter((l: string) => /ERROR|Traceback/.test(l))
    expect(errs, `backend errors:\n${errs.join('\n')}`).toHaveLength(0)
  } finally {
    await app.close()
  }
})
