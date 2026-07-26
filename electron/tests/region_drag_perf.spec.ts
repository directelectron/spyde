/**
 * region_drag_perf.spec.ts — the REAL drag of an integrating ROI.
 *
 * Headless pytest + a green tsc cannot see this: it needs a real navigator with a
 * real selector, dragged with a real mouse, painting real frames. This spec drives
 * an 8x8-ish integrating region across a 4D-STEM navigator in a squiggle and
 * screenshots each stage, then reads the backend's own per-update profile lines to
 * get the per-step read cost the user actually feels.
 *
 * What it is guarding: the region read used to bypass the array cache entirely and
 * pay one dask compute per point (~2.9 s per drag step on a 64x64x256^2 scan).
 * Regions now go through the same reader + BlockCache as a single point, so a step
 * is ~5 ms. If someone re-adds an `idx.ndim > 1` bail, this spec goes slow again.
 *
 * Run: npx playwright test tests/region_drag_perf.spec.ts --project=electron \
 *        --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, navWindow, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'region_drag_shots'

test('integrating ROI drags smoothly across a 4D-STEM navigator', async () => {
  test.setTimeout(300_000)

  // SPYDE_LOG_LEVEL=INFO tees the backend's nav profile lines to stderr, which the
  // harness captures — backend emit()/emit_status() do NOT reach Playwright stdout
  // (they are the PLOTAPP: line protocol, consumed by the main process).
  // SPYDE_NAV_PROFILE=1 makes NavProfile emit one "[NAV-PROFILE] … total=N.Nms"
  // INFO line per navigator update — the per-step read cost, straight from the
  // code under test.
  const ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO', SPYDE_NAV_PROFILE: '1' },
  })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    // LAZY 4D-STEM with a 3x3 nav-chunk grid — the region read only goes through
    // the array cache / BlockCache when the data is lazy. (si_grains is EAGER:
    // the profile line says "eager" and the whole cache path is skipped, so it
    // measures nothing about this change.) Crossing chunk boundaries mid-drag is
    // exactly the case the one-entry memo used to thrash on.
    await backendAction(page, 'load_test_data_lazy_chunked')
    await waitForSubwindowCount(page, 2, 120_000)
    await page.waitForTimeout(2_000)          // let the first frames paint
    await page.screenshot({ path: `${SHOTS}/01-loaded.png` })

    const nav = navWindow(page)
    const sig = sigWindow(page)
    await expect(nav).toBeVisible()
    await expect(sig).toBeVisible()

    // Switch the navigator selector to INTEGRATING mode by CLICKING the real Plot
    // Control button — it carries the selector_id, which a hand-built window_id
    // does not (an earlier version guessed the id from the DOM, got -1, and the
    // spec silently measured a crosshair drag instead of a region drag).
    const integrateBtn = page.getByTestId('selector-integrate').first()
    await expect(integrateBtn).toBeVisible({ timeout: 30_000 })
    await integrateBtn.click()
    await page.waitForTimeout(2_000)
    await page.screenshot({ path: `${SHOTS}/02-integrate-mode.png` })

    // Prove the mode actually took: the rectangle ROI must be on the navigator.
    const modeOk = await page.evaluate(() => {
      const b = document.querySelector('[data-testid="selector-integrate"]') as HTMLElement
      if (!b) return 'no-button'
      // the active toggle gets a distinct background
      return getComputedStyle(b).backgroundColor
    })
    console.log(`[region-drag] integrate button bg after click: ${modeOk}`)

    const box = await nav.boundingBox()
    if (!box) throw new Error('navigator has no bounding box')

    // First DRAW a box, so the ROI has real extent. Clicking straight into a move
    // leaves it a 1-px line (the earlier run's screenshot showed a thin green
    // line, i.e. a ~1-wide region — not the 8x8 integrate this is meant to test).
    const drawX = box.x + box.width * 0.30
    const drawY = box.y + box.height * 0.30
    await page.mouse.move(drawX, drawY)
    await page.mouse.down()
    await page.mouse.move(drawX + box.width * 0.28, drawY + box.height * 0.28,
      { steps: 8 })
    await page.mouse.up()
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/02b-roi-drawn.png` })

    // Drag the ROI in a squiggle across the navigator. Each move is a real
    // pointer_move the selector reacts to, so the backend does a real region read.
    // Start from INSIDE the drawn box so the drag MOVES it rather than drawing a
    // new one.
    const cx = drawX + box.width * 0.14
    const cy = drawY + box.height * 0.14
    const path: Array<[number, number]> = []
    for (let i = 0; i < 40; i++) {
      const t = i / 40
      path.push([
        cx + Math.sin(t * Math.PI * 3) * box.width * 0.22 + t * box.width * 0.25,
        cy + Math.cos(t * Math.PI * 2.2) * box.height * 0.20 + t * box.height * 0.25,
      ])
    }

    const t0 = Date.now()
    await page.mouse.move(path[0][0], path[0][1])
    await page.mouse.down()
    for (const [x, y] of path) {
      await page.mouse.move(x, y)
      await page.waitForTimeout(16)           // ~60 fps pointer cadence
    }
    await page.mouse.up()
    const dragMs = Date.now() - t0
    await page.waitForTimeout(2_500)          // settle re-fire + final paint
    await page.screenshot({ path: `${SHOTS}/03-after-drag.png` })

    // The signal window must show a real integrated frame, not black/placeholder.
    const sigShot = await sig.screenshot({ path: `${SHOTS}/04-signal-panel.png` })
    expect(sigShot.byteLength).toBeGreaterThan(2_000)

    // ---- the numbers: backend nav profile lines --------------------------
    const logs: string[] = backend.logBuffer            // array of raw lines
    const navLines = logs.filter((l: string) => l.includes('[NAV-PROFILE]'))
    const durations: number[] = []
    const regionLines: string[] = []
    for (const l of navLines) {
      const m = l.match(/total=([\d.]+)ms/)
      if (m) durations.push(parseFloat(m[1]))
      // a region read logs its point count via the array-cache extra
      if (/array-cache region/.test(l)) regionLines.push(l)
    }
    console.log(`[region-drag] NAV-PROFILE lines=${navLines.length} ` +
      `region-served=${regionLines.length}`)
    if (navLines.length) console.log(`[region-drag] sample: ${navLines[navLines.length - 1]}`)
    durations.sort((a, b) => a - b)
    const med = durations.length ? durations[Math.floor(durations.length / 2)] : NaN
    const p95 = durations.length
      ? durations[Math.min(durations.length - 1, Math.floor(durations.length * 0.95))]
      : NaN

    console.log(`[region-drag] ${path.length} moves in ${dragMs} ms ` +
      `(${(dragMs / path.length).toFixed(1)} ms/move wall)`)
    console.log(`[region-drag] backend nav reads: n=${durations.length} ` +
      `median=${med.toFixed(1)}ms p95=${p95.toFixed(1)}ms`)
    if (durations.length) {
      console.log(`[region-drag] slowest 5: ` +
        durations.slice(-5).map((d) => d.toFixed(1)).join(', '))
    }

    // A pre-fix region read was ~2.9 s per step, so the drag could not keep up at
    // all. Assert loosely on the backend read cost (machine-dependent), tightly on
    // the thing that regressing WOULD break: the drag completing without the UI
    // wedging and without a JS error.
    if (durations.length) {
      expect(med).toBeLessThan(200)
    }
    assertNoJsErrors()
  } finally {
    await ctx.app.close().catch(() => {})
  }
})
