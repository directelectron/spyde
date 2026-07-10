/**
 * first_paint_probe.spec.ts — DIAGNOSTIC (not a CI guard): when does the
 * SIGNAL panel actually show pixels during a fresh open of the real 16 GB
 * movie? Samples the figure IFRAME only (no subwindow chrome, so title-bar
 * text can't fake "content"), polls for 60 s with timed screenshots, no
 * interaction. Deletes the sidecar first so the fill really runs.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync, existsSync, unlinkSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

const REPRO = process.env.SPYDE_REPRO_FILE ||
  'D:\\20251117_88075_run3 some growth_1236_movie.mrc'
const SIDECAR = REPRO + '.spyde-nav.npz'

test('probe: signal iframe pixels over time on fresh 16GB movie open', async () => {
  test.setTimeout(300_000)
  // Opt-in: hammers the real 16 GB file on D: and deletes/rebuilds its
  // navigator sidecar — not for a default `npm test` run.
  test.skip(process.env.SPYDE_REAL_DATA_E2E !== '1', 'set SPYDE_REAL_DATA_E2E=1 to run')
  test.skip(!existsSync(REPRO), `repro file missing: ${REPRO}`)
  mkdirSync('firstpaint_shots', { recursive: true })
  try { unlinkSync(SIDECAR) } catch {}

  const { app, page, backend } = await launchApp({
    dask: true, env: { SPYDE_LOG_LEVEL: 'DEBUG', SPYDE_NAV_PROFILE: '1' },
  })
  const t0 = Date.now()
  const el = () => `${((Date.now() - t0) / 1000).toFixed(1)}s`
  try {
    await backendAction(page, 'open_file', { path: REPRO })
    await page.waitForFunction(
      () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
      { timeout: 180_000 })
    console.log(`[t] 2 subwindows at ${el()}`)

    // Pixel-sample the nth subwindow's IFRAME (the figure content only).
    const sampleIframe = async (idx: number) => {
      const fr = page.locator('[data-testid="subwindow"]').nth(idx).locator('iframe').first()
      if (!(await fr.isVisible().catch(() => false))) return { nonBlack: -1 }
      const png = await fr.screenshot({ timeout: 5000 }).catch(() => null)
      if (!png) return { nonBlack: -2 }
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
        let nonBlack = 0
        for (let i = 0; i < d.length; i += 4) {
          if (d[i] + d[i + 1] + d[i + 2] > 90) nonBlack++
        }
        return { nonBlack }
      }, png.toString('base64'))
    }

    const shotAt = new Set([3, 6, 10, 15, 25, 40, 55])
    let sigFirst = -1
    for (let i = 0; i < 60; i++) {
      const nav = await sampleIframe(0)
      const sig = await sampleIframe(1)
      console.log(`[poll ${el()} @${new Date().toISOString().slice(11, 23)}] navIframe=${nav.nonBlack} sigIframe=${sig.nonBlack}`)
      // Chrome/axes inside the iframe idle at ~9k lit px; a real painted 4k
      // frame is ~100k+. 30k cleanly separates them.
      if (sigFirst < 0 && sig.nonBlack > 30000) {
        sigFirst = Date.now()
        console.log(`[t] SIGNAL IFRAME real content at ${el()}`)
        await page.screenshot({ path: 'firstpaint_shots/probe-sig-first.png' })
      }
      if (shotAt.has(i)) {
        await page.screenshot({ path: `firstpaint_shots/probe-${String(i).padStart(2, '0')}s.png` })
      }
      if (sigFirst > 0 && i > 10) break
      await page.waitForTimeout(1000)
    }
    console.log(`RESULT sigIframeFirst=${sigFirst > 0 ? ((sigFirst - t0) / 1000).toFixed(1) : 'NEVER'}`)
    const lines = (backend.logBuffer as string[]).filter((l) =>
      /TILEDBG|sidecar|navigator|Reading|re-loaded/i.test(l))
    for (const l of lines.slice(-40)) console.log(l)
    expect(sigFirst, 'signal iframe never painted').toBeGreaterThan(0)
  } finally {
    await app.close()
  }
})
