/**
 * first_paint_real_load.spec.ts — end-to-end verification on the user's REAL
 * 16 GB DE movie MRC (977 x 4096 x 4096 uint8):
 *
 *  Phase 1 (fresh open, no sidecar): the SIGNAL panel must paint its first
 *  frame WITHOUT any navigator interaction (the nav fill is deferred until
 *  that frame lands, so the fill's 977 chunk reads can't starve it), and the
 *  completed fill must write the navigator sidecar beside the file.
 *
 *  Phase 2 (relaunch + reopen): the navigator must load from the sidecar
 *  (no whole-file re-read) and both panels must show content immediately.
 *
 * Skips (passes trivially) if the movie file is absent.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync, existsSync, unlinkSync } from 'fs'
const { launchApp, backendAction } = require('./_harness.cjs')

const REPRO = process.env.SPYDE_REPRO_FILE ||
  'D:\\20251117_88075_run3 some growth_1236_movie.mrc'
const SIDECAR = REPRO + '.spyde-nav.npz'

// Sample the nth subwindow's figure IFRAME pixels: hash + count of clearly-lit
// pixels. The iframe (not the subwindow) so title-bar chrome can't fake
// "content" — chrome/axes idle at ~9k lit px, a real painted 4k frame is 100k+.
async function sample(page, idx: number) {
  const el = page.locator('[data-testid="subwindow"]').nth(idx).locator('iframe').first()
  if (!(await el.isVisible().catch(() => false))) return { hash: 'none', nonBlack: -1 }
  const png = await el.screenshot({ timeout: 5000 }).catch(() => null)
  if (!png) return { hash: 'shot-fail', nonBlack: -2 }
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
      if (d[i] + d[i + 1] + d[i + 2] > 90) nonBlack++
    }
    return { hash: String(s), nonBlack }
  }, png.toString('base64'))
}

test('16GB movie: first open paints signal w/o interaction + writes sidecar; reopen is instant', async () => {
  test.setTimeout(600_000)
  // Opt-in: reads the whole real 16 GB file on D: and rewrites its navigator
  // sidecar — not for a default `npm test` run.
  test.skip(process.env.SPYDE_REAL_DATA_E2E !== '1', 'set SPYDE_REAL_DATA_E2E=1 to run')
  test.skip(!existsSync(REPRO), `repro file missing: ${REPRO}`)
  mkdirSync('firstpaint_shots', { recursive: true })

  // ── Phase 1: fresh open (no sidecar) ─────────────────────────────────────
  try { unlinkSync(SIDECAR) } catch {}
  {
    const { app, page, backend } = await launchApp({
      dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' },
    })
    const t0 = Date.now()
    const el = (ms: number) => `${((ms - t0) / 1000).toFixed(1)}s`
    try {
      await backendAction(page, 'open_file', { path: REPRO })
      await page.waitForFunction(
        () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
        { timeout: 180_000 })
      const tWindows = Date.now()
      console.log(`[t] 2 subwindows at ${el(tWindows)}`)

      // The SIGNAL panel (index 1) must paint WITHOUT interaction. Poll ≤30 s.
      let sigFirst = -1
      for (let i = 0; i < 30 && sigFirst < 0; i++) {
        const s = await sample(page, 1)
        if (s.nonBlack > 30000) { sigFirst = Date.now(); break }
        await page.waitForTimeout(1000)
      }
      console.log(`[t] signal first content at ${sigFirst > 0 ? el(sigFirst) : 'NEVER'}`)
      await page.screenshot({ path: 'firstpaint_shots/v2-10-first-open.png' })
      expect(sigFirst, 'signal panel never painted without interaction').toBeGreaterThan(0)
      expect(sigFirst - tWindows, 'signal paint took too long after windows appeared')
        .toBeLessThan(20_000)

      // The fill must complete and write the sidecar (INFO log line + file).
      await backend.waitForLog('saved navigator sidecar', 420_000)
      expect(existsSync(SIDECAR), 'sidecar file missing after fill').toBe(true)
      console.log(`[t] sidecar written at ${el(Date.now())}`)
      await page.screenshot({ path: 'firstpaint_shots/v2-11-fill-done.png' })
    } finally {
      await app.close()
    }
  }

  // ── Phase 2: relaunch + reopen — navigator from sidecar, instant ─────────
  {
    const { app, page, backend } = await launchApp({
      dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' },
    })
    const t0 = Date.now()
    const el = (ms: number) => `${((ms - t0) / 1000).toFixed(1)}s`
    try {
      await backendAction(page, 'open_file', { path: REPRO })
      await page.waitForFunction(
        () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
        { timeout: 180_000 })
      const tWindows = Date.now()
      console.log(`[t] 2 subwindows at ${el(tWindows)}`)

      // BOTH panels must show content promptly — the navigator has real data
      // from the sidecar (no NaN placeholder, no whole-file read).
      let navFirst = -1, sigFirst = -1
      for (let i = 0; i < 20 && (navFirst < 0 || sigFirst < 0); i++) {
        const nav = await sample(page, 0)
        const sig = await sample(page, 1)
        if (navFirst < 0 && nav.nonBlack > 500) navFirst = Date.now()
        if (sigFirst < 0 && sig.nonBlack > 30000) sigFirst = Date.now()
        if (navFirst < 0 || sigFirst < 0) await page.waitForTimeout(1000)
      }
      console.log(`[t] reopen: nav at ${navFirst > 0 ? el(navFirst) : 'NEVER'}, ` +
        `signal at ${sigFirst > 0 ? el(sigFirst) : 'NEVER'}`)
      await page.screenshot({ path: 'firstpaint_shots/v2-20-reopen.png' })

      expect(navFirst, 'navigator blank on reopen').toBeGreaterThan(0)
      expect(sigFirst, 'signal blank on reopen').toBeGreaterThan(0)
      expect(navFirst - tWindows, 'navigator not instant on reopen').toBeLessThan(10_000)

      const hit = (backend.logBuffer as string[]).some(
        (l) => l.includes('navigator loaded from sidecar'))
      expect(hit, 'backend did not log a sidecar hit on reopen').toBe(true)
    } finally {
      await app.close()
    }
  }
})
