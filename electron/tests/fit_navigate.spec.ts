/**
 * fit_navigate.spec.ts — after fitting the scan, does moving the navigator
 * show THAT position's fit?
 *
 * "Maybe it's just using the fit from the previous position." This settles it:
 * fit the whole scan, then move the navigator between positions whose true
 * parameters differ, and require the caret AND the overlaid model to change.
 *
 * two_gaussians has a narrow component whose centre varies across the scan
 * (`50 + 10*sin(3*pi*x/32)*cos(4*pi*y/32)`), so two well-chosen positions have
 * genuinely different answers — if the caret shows the same numbers at both,
 * it is showing a stale model, not this position's fit.
 *
 * Run: npx playwright test tests/fit_navigate.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, raiseWindowOwning, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fit_navigate_shots'

test('the overlaid model follows the navigator after a scan fit', async () => {
  test.setTimeout(600_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'tutorial_load', { name: 'spectroscopy' })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_500)

    // Both figure ids: the navigator's crosshair is what we drive, the
    // signal's panel state is where the overlaid model lives.
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
    expect(figIds.nav, 'no navigator figure').toBeTruthy()
    expect(figIds.sig, 'no signal figure').toBeTruthy()

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

    // ── fit the whole scan ───────────────────────────────────────────────
    // The live-preview window opened by adding components can sit over the caret
    // by now; raise its own window first (the click a user makes without noticing).
    await raiseWindowOwning(page, "fit-wizard")
    await page.getByTestId('fit-tab-Run').click()
    await page.locator('[data-testid="fit-run"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/converged/i, { timeout: 300_000 })
    await page.getByTestId('fit-tab-Model').click()
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/01-scan-fitted.png`, fullPage: true })

    // ── move the navigator and read the model at each stop ───────────────
    const navCrosshair = async () => {
      const ws = await page.evaluate((f) => (window as any)._spyde_test_widgets(f), figIds.nav)
      const w = ws.find((v: any) => v.type === 'crosshair')
      expect(w, 'the navigator has no crosshair').toBeTruthy()
      return w
    }
    const cross = await navCrosshair()

    /** Signature of the SPECTRUM on the signal plot — changes when the
     *  navigator actually lands on a different position. */
    const dataSig = async () => page.evaluate((f) => {
      const hook = (window as any)._spyde_test_panel_json
      for (const raw of hook ? hook(f) : []) {
        const d = JSON.parse(raw)
        if (!d.data_b64) continue
        const bin = atob(d.data_b64)
        const bytes = new Uint8Array(bin.length)
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
        const v = Array.from(new Float64Array(bytes.buffer))
        return `${v.length}:${v[0]}:${v[v.length >> 1]}:${v[v.length - 1]}`
      }
      return ''
    }, figIds.sig)

    const goTo = async (cx: number, cy: number) => {
      const before = await dataSig()
      const seqBefore = await page.evaluate(() =>
        (window as unknown as { _spyde_fit_state_seq?: number })
          ._spyde_fit_state_seq ?? 0)
      await page.evaluate(({ f, panel, id, x, y }) => {
        window.postMessage({
          type: 'awi_event', figId: f,
          data: JSON.stringify({
            source: 'js', panel_id: panel, widget_id: id,
            event_type: 'pointer_up', cx: x, cy: y,
          }),
        }, '*')
      }, { f: figIds.nav, panel: cross.panel_id, id: cross.id, x: cx, y: cy })
      // Confirm the navigator LANDED before reading anything, instead of
      // assuming 2.5s was enough. When it was not, the caret never showed the
      // new position and the reads below compared a position against itself —
      // "the model stayed IDENTICAL at two different navigator positions",
      // which is a moved-too-early failure, not a stale-fit one.
      const deadline = Date.now() + 60_000
      while (Date.now() < deadline) {
        await page.waitForTimeout(100)
        if ((await dataSig()) !== before) break
      }
      // …and then for the CARET to have processed this move. The spectrum and
      // the fit arrive separately: the data landing above says the navigator
      // moved, not that `fit_navigated` has run for the new position, so the
      // model read below could still be the previous position's.
      //
      // `fit_navigated` pushes the overlay and THEN emits its state down the
      // same ordered protocol, so a new fit_state is proof the caret is now
      // showing THIS position — the same completion signal its own navigator
      // coalescer waits on (navDone), and the one fit_quality's sweep uses.
      //
      // Bounded, not fatal: the caret coalesces moves by keeping one request
      // in flight, so a move posted while another is pending can legitimately
      // be folded into it and produce no additional state. The assertion below
      // is still the real check.
      await page.waitForFunction(
        (s: number) =>
          ((window as unknown as { _spyde_fit_state_seq?: number })
            ._spyde_fit_state_seq ?? 0) > s,
        seqBefore, { timeout: 30_000 },
      ).catch(() => { /* coalesced; the assertion below still decides */ })
    }

    const readModel = async () => {
      const names = await page.locator('[data-testid^="fit-comp-"]')
        .evaluateAll((els) => els.map((e) =>
          e.getAttribute('data-testid')!.replace('fit-comp-', '')))
      const out: Record<string, number> = {}
      for (const n of names) {
        for (const p of ['A', 'centre', 'sigma']) {
          const loc = page.locator(`[data-testid="fit-p-${n}-${p}"]`)
          if (await loc.count()) out[`${n}.${p}`] = Number(await loc.inputValue())
        }
      }
      return out
    }

    // The narrow gaussian's true centre swings +/-10 across the scan, so
    // these two corners have genuinely different answers.
    await goTo(6, 6)
    const at66 = await readModel()
    await page.screenshot({ path: `${SHOTS}/02-at-6-6.png`, fullPage: true })

    await goTo(26, 26)
    // Poll rather than read once, for the same reason as the curve read at the
    // bottom: goTo's flat 2.5s can be shorter than the caret's refresh on a
    // loaded runner, and a stale read here fails as "the model is IDENTICAL".
    // The assertion keeps its teeth — if the fields never diverge, the poll
    // times out and reports it.
    let at2626: Record<string, number> = {}
    await expect
      .poll(async () => {
        at2626 = await readModel()
        return Object.keys(at66)
          .filter((k) => Math.abs(at66[k] - at2626[k]) > 1e-9).length
      }, {
        timeout: 60_000,
        message: 'the model stayed IDENTICAL at two different navigator '
          + 'positions — the caret is showing a stale fit, not this position\'s',
      })
      .toBeGreaterThan(0)
    await page.screenshot({ path: `${SHOTS}/03-at-26-26.png`, fullPage: true })

    console.log('model at (6,6)  :', JSON.stringify(at66))
    console.log('model at (26,26):', JSON.stringify(at2626))

    // ── and each named component must be the SAME PEAK at both stops ─────
    // Two gaussians are exchangeable, so an unconstrained fit puts the broad
    // one in whichever slot each position happens to pick. Measured before
    // this was fixed: component 1 was the broad peak at 43% of positions —
    // so scrubbing made one component's amplitude drop by an order of
    // magnitude (it read as "the fit suppresses a component to zero") and a
    // committed map would be a checkerboard of two different peaks.
    const names = Object.keys(at66)
      .filter((k) => k.endsWith('.sigma')).sort()
    expect(names.length).toBe(2)
    const widthOrder = (m: Record<string, number>) =>
      m[names[0]] < m[names[1]] ? 'narrow-first' : 'wide-first'
    expect(
      widthOrder(at2626),
      `the components SWAPPED identity between positions: ` +
      `${names[0]}=${at66[names[0]].toFixed(2)}/${at2626[names[0]].toFixed(2)}, ` +
      `${names[1]}=${at66[names[1]].toFixed(2)}/${at2626[names[1]].toFixed(2)}`,
    ).toBe(widthOrder(at66))

    // And the overlaid curve must have moved too, not just the numbers.
    const modelCurve = async () => page.evaluate((f) => {
      const hook = (window as any)._spyde_test_panel_json
      for (const raw of hook ? hook(f) : []) {
        const d = JSON.parse(raw)
        for (const ln of d.extra_lines ?? []) {
          if (ln.label !== 'model') continue
          const bin = atob(ln.data_b64)
          const bytes = new Uint8Array(bin.length)
          for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
          const v = Array.from(new Float64Array(bytes.buffer))
          return { peak: Math.max(...v), argmax: v.indexOf(Math.max(...v)) }
        }
      }
      return null
    }, figIds.sig)

    const curveHere = await modelCurve()
    expect(curveHere, 'no overlaid model curve on the signal plot').toBeTruthy()
    await goTo(6, 6)
    // POLL, don't read once. goTo ends in a flat waitForTimeout(2_500), and
    // repainting the overlay is a backend round trip — on a loaded CI runner
    // it can outlast that, so a single read returns the PREVIOUS position's
    // curve and the assertion below fires on two identical values. That is
    // exactly how this went flaky (peak 404.84/argmax 486 at both stops).
    // Waiting for the change is also faster than the fixed sleep in the
    // common case, and it is the property the test is named for.
    await expect
      .poll(async () => JSON.stringify(await modelCurve()), {
        timeout: 60_000,
        message: 'the overlaid model curve did not change between positions',
      })
      .not.toEqual(JSON.stringify(curveHere))

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-20).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
