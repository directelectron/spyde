/**
 * fit_drag_latency.spec.ts — the model must KEEP UP with the navigator drag.
 *
 * Reported as "a pause and then a snap": drag the navigator and the fit sits
 * still, then jumps to the final position when you let go.
 *
 * Every pointer frame of a drag posts a `pointer_move`, and the Fit caret
 * sends a `fit_navigated` for each one — each of which recalls, redraws the
 * preview and re-sends the whole model. If those arrive faster than they are
 * served they queue, and the queue drains after the drag: a pause, then a
 * snap.
 *
 * This drives a real stream of moves and measures how long AFTER the last one
 * the drawn model settles. A caret that keeps up settles almost immediately;
 * a backlog shows up as a long tail.
 *
 * Run: npx playwright test tests/fit_drag_latency.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, raiseWindowOwning, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

test('the fit keeps up with a navigator drag', async () => {
  test.setTimeout(600_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'tutorial_load', { name: 'spectroscopy' })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_500)

    const figIds = await page.evaluate(() => {
      const out: Record<string, string> = {}
      for (const s of Array.from(document.querySelectorAll('[data-testid="subwindow"]'))) {
        const tid = s.querySelector('iframe')?.getAttribute('data-testid') ?? ''
        const crumb = s.querySelector('[data-testid="window-breadcrumb"]')?.textContent ?? ''
        if (!tid.startsWith('figure-')) continue
        out[crumb.startsWith('N-') ? 'nav' : 'sig'] = tid.slice('figure-'.length)
      }
      return out
    })

    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Fit').click()
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })
    for (let i = 0; i < 2; i++) {
      await page.locator('[data-testid="fit-add-toggle"]').click()
      await expect(page.locator('[data-testid="fit-add-Gaussian"]')).toBeVisible({ timeout: 30_000 })
      await page.locator('[data-testid="fit-add-Gaussian"]').click()
      await page.waitForTimeout(800)
    }
    // The live-preview window opened by adding components can sit over the caret
    // by now; raise its own window first (the click a user makes without noticing).
    await raiseWindowOwning(page, "fit-wizard")
    await page.getByTestId('fit-tab-Run').click()
    await page.locator('[data-testid="fit-run"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/converged/i, { timeout: 300_000 })
    await page.getByTestId('fit-tab-Model').click()
    await page.waitForTimeout(1_500)

    const cross = await page.evaluate((f) => {
      const ws = (window as any)._spyde_test_widgets(f)
      return ws.find((w: any) => w.type === 'crosshair')
    }, figIds.nav)

    /** The model's A for the first component — a cheap proxy for "which
     *  position's fit is on screen". */
    const shownA = async () =>
      Number(await page.locator('[data-testid="fit-p-Gaussian-A"]').inputValue())

    // Park at one end, note the value, then stream moves to the other end at
    // pointer rate and time how long after the LAST move it settles.
    const drag = async (path: Array<[number, number]>, gapMs: number) => {
      await page.evaluate(({ f, panel, id, pts, gap }) => {
        const post = (cx: number, cy: number, t: string) =>
          window.postMessage({
            type: 'awi_event', figId: f,
            data: JSON.stringify({
              source: 'js', panel_id: panel, widget_id: id,
              event_type: t, cx, cy,
            }),
          }, '*')
        return new Promise<void>((done) => {
          let i = 0
          const step = () => {
            if (i >= pts.length) { post(pts[pts.length - 1][0], pts[pts.length - 1][1], 'pointer_up'); done(); return }
            post(pts[i][0], pts[i][1], 'pointer_move')
            i += 1
            setTimeout(step, gap)
          }
          step()
        })
      }, { f: figIds.nav, panel: cross.panel_id, id: cross.id, pts: path, gap: gapMs })
    }

    await drag([[2, 2]], 16)
    await page.waitForTimeout(2_500)
    const start = await shownA()

    const path: Array<[number, number]> = []
    for (let i = 0; i <= 28; i++) path.push([2 + i, 2 + i])
    const t0 = Date.now()
    await drag(path, 16)                       // ~60 fps, 29 moves
    const posted = Date.now() - t0

    const target = await (async () => {
      // Whatever the final position settles to — sampled until it STOPS
      // moving, not after a flat 6s. If the first drag has not finished
      // settling, `target` is a moving goalpost and the 2%-of-swing window
      // below can never be hit: on CI this reported settled = -1, i.e. never
      // converged in 12s, rather than merely exceeding the 700 ms budget it
      // is actually there to measure. Making the reference real keeps that
      // budget honest instead of loosening it.
      const deadline = Date.now() + 30_000
      let prev = await shownA()
      let stable = 0
      while (Date.now() < deadline) {
        await page.waitForTimeout(200)
        const now = await shownA()
        stable = Math.abs(now - prev) <= Math.abs(now) * 1e-9 ? stable + 1 : 0
        prev = now
        if (stable >= 3) break
      }
      return prev
    })()
    expect(target).not.toBeCloseTo(start, 3)

    // Re-run the same drag and time how long after the last post the value
    // reaches its settled target.
    await drag([[2, 2]], 16)
    await page.waitForTimeout(2_500)
    const t1 = Date.now()
    await drag(path, 16)
    const lastPost = Date.now()
    let settled = -1
    for (let i = 0; i < 120; i++) {
      if (Math.abs((await shownA()) - target) < Math.abs(target - start) * 0.02) {
        settled = Date.now() - lastPost
        break
      }
      await page.waitForTimeout(100)
    }
    console.log(`drag posted ${path.length} moves over ${posted} ms ` +
      `(total ${lastPost - t1} ms); model settled ${settled} ms after the last one`)

    expect(settled, 'the model never caught up with the drag').toBeGreaterThan(-1)
    expect(
      settled,
      `the model lagged ${settled} ms behind the end of the drag — that is ` +
      `the pause, and the value arriving is the snap`,
    ).toBeLessThan(700)

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-15).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
