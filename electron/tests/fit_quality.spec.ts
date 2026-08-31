/**
 * fit_quality.spec.ts — after fitting, does the drawn model match the drawn data?
 *
 * Every other fit spec asserts on parameters, coverage or window counts. None
 * of them would notice the thing actually reported: a model curve sitting well
 * below the spectrum it is supposed to describe. This reads BOTH curves out of
 * the same panel state and compares them, which is exactly what the eye does.
 *
 * It also checks the follow-up: at a position that has already been fitted,
 * pressing "Fit spectrum" should have nothing left to do. If it visibly
 * improves the curve then what was on screen was not that position's fit.
 *
 * Run: npx playwright test tests/fit_quality.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, raiseWindowOwning, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'fit_quality_shots'

test('the fitted model matches the spectrum on screen', async () => {
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

    /** The panel's PRIMARY line (the spectrum) and its `model` overlay. */
    const curves = async () => page.evaluate((f) => {
      const dec = (b64?: string) => {
        if (!b64) return null
        const bin = atob(b64)
        const bytes = new Uint8Array(bin.length)
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
        return Array.from(new Float64Array(bytes.buffer))
      }
      const hook = (window as any)._spyde_test_panel_json
      for (const raw of hook ? hook(f) : []) {
        const d = JSON.parse(raw)
        const data = dec(d.data_b64)
        if (!data) continue
        const model = (d.extra_lines ?? [])
          .filter((l: any) => l.label === 'model')
          .map((l: any) => dec(l.data_b64))[0] ?? null
        return { data, model }
      }
      return null
    }, figIds.sig)

    /** Residual as a fraction of the data's own range — what the eye judges. */
    const misfit = (c: { data: number[]; model: number[] | null }) => {
      expect(c.model, 'no model curve is drawn').toBeTruthy()
      const n = Math.min(c.data.length, c.model!.length)
      let se = 0
      for (let i = 0; i < n; i++) se += (c.data[i] - c.model![i]) ** 2
      const rms = Math.sqrt(se / n)
      const span = Math.max(...c.data) - Math.min(...c.data)
      return rms / (span || 1)
    }

    /** misfit(), but only once the overlay has stopped changing.
     *
     * Every goTo below ends in a flat waitForTimeout, and repainting the model
     * overlay is a backend round trip that can outlast it on a loaded runner.
     * A single read then scores the PREVIOUS position's curve against THIS
     * position's data, which is not a small error: the sweep logged worst
     * misfits of 47%, 76%, 83% and 191% across CI runs, while re-measuring
     * that very position a moment later gave 2-12%. So "worst" was really
     * "where the read was most stale", and the assertion that compares it to a
     * dedicated refit went flaky on the noise.
     *
     * Sampling until two consecutive reads agree fixes the measurement rather
     * than loosening the tolerance around it — and unlike waiting for the
     * curve to CHANGE, it makes no assumption that neighbouring positions have
     * visibly different fits.
     */
    const settledMisfit = async (timeout = 30_000) => {
      const deadline = Date.now() + timeout
      let prev = misfit((await curves())!)
      // THREE consecutive agreements, not two: one repeat can land in a lull
      // while the model overlay is still catching up with the data.
      let stable = 0
      while (Date.now() < deadline) {
        await page.waitForTimeout(150)
        const now = misfit((await curves())!)
        stable = Math.abs(now - prev) < 1e-9 ? stable + 1 : 0
        prev = now
        if (stable >= 2) return now
      }
      return prev
    }

    const sig = sigWindow(page)
    await sig.getByTestId('subwindow-titlebar').hover()
    await sig.getByTestId('action-btn-Fit').click()
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })

    // Park the navigator FIRST. A new component is seeded from the spectrum
    // ON SCREEN, so which one that is decides the whole scan's starting point
    // — and under load the first paint may not have landed when the component
    // is added. Pinning it keeps this a test of fit QUALITY rather than of
    // whichever spectrum happened to be up.
    const crossEarly = await page.evaluate((f) => {
      const ws = (window as any)._spyde_test_widgets(f)
      return ws.find((w: any) => w.type === 'crosshair')
    }, figIds.nav)
    await page.evaluate(({ f, panel, id }) => {
      window.postMessage({
        type: 'awi_event', figId: f,
        data: JSON.stringify({
          source: 'js', panel_id: panel, widget_id: id,
          event_type: 'pointer_up', cx: 16, cy: 16,
        }),
      }, '*')
    }, { f: figIds.nav, panel: crossEarly.panel_id, id: crossEarly.id })
    await page.waitForTimeout(2_000)

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
    await page.waitForTimeout(2_000)
    await page.screenshot({ path: `${SHOTS}/01-after-fit-all.png`, fullPage: true })

    const afterAll = await settledMisfit()
    console.log(`misfit right after Fit all Spectra: ${(afterAll * 100).toFixed(1)}% of range`)

    // ── sweep the navigator: ONE position proves nothing ─────────────────
    // The report is "if I go through all of the positions" — so walk a grid
    // of them and find the worst, rather than trusting whichever the
    // navigator happens to start on.
    const cross = await page.evaluate((f) => {
      const ws = (window as any)._spyde_test_widgets(f)
      return ws.find((w: any) => w.type === 'crosshair')
    }, figIds.nav)
    expect(cross, 'the navigator has no crosshair').toBeTruthy()

    const sigOf = (a: number[] | null) =>
      a ? `${a.length}:${a[0]}:${a[a.length >> 1]}:${a[a.length - 1]}` : ''

    /** Signatures of BOTH curves. They update independently — the spectrum can
     *  be this position's while the model overlay is still the previous one's. */
    const sigs = async () => {
      const c = await curves()
      return { data: sigOf(c?.data ?? null), model: sigOf(c?.model ?? null) }
    }

    const goTo = async (cx: number, cy: number) => {
      const before = await sigs()
      await page.evaluate(({ f, panel, id, x, y }) => {
        window.postMessage({
          type: 'awi_event', figId: f,
          data: JSON.stringify({
            source: 'js', panel_id: panel, widget_id: id,
            event_type: 'pointer_up', cx: x, cy: y,
          }),
        }, '*')
      }, { f: figIds.nav, panel: cross.panel_id, id: cross.id, x: cx, y: cy })

      // A best-effort wait for both curves to become this position's — and
      // best-effort is now GOOD ENOUGH, which is the point of the rework
      // below: the sweep only nominates candidates, and every asserted number
      // is re-measured at a parked navigator afterwards. So this wait costs
      // sensitivity when it is wrong, never correctness.
      //
      // Do not make this exact, and above all do not make it FATAL. Waiting on
      // the caret's fit_state was tried here and is a trap twice over: it does
      // not fix the tearing (the state and the panel push are applied by
      // different consumers — the React wizard vs the plot iframe — so wire
      // order is not screen order), and the caret coalesces navigator moves by
      // keeping one request in flight, so a folded move produces no extra
      // state and a hard wait on it times out. Under 64 busy processes that
      // turned a passing test into a 30 s timeout.
      const deadline = Date.now() + 30_000
      while (Date.now() < deadline) {
        await page.waitForTimeout(100)
        const now = await sigs()
        if (now.data !== before.data && now.model !== before.model) break
      }
    }

    // The sweep SEARCHES for suspicious positions. It does not judge them —
    // see the note above the assertions for why its readings cannot be trusted
    // to that job.
    const seen: { m: number; at: number[] }[] = []
    const swept: number[] = []
    for (const cy of [2, 10, 16, 24, 30]) {
      for (const cx of [2, 10, 16, 24, 30]) {
        await goTo(cx, cy)
        const m = await settledMisfit()
        swept.push(m)
        seen.push({ m, at: [cx, cy] })
        if (swept.length <= 3) {
          console.log(`  at [${cx},${cy}] misfit ${(m * 100).toFixed(1)}% ` +
            `status="${await page.locator('[data-testid="fit-status"]').textContent()}"` +
            ` coverage="${await page.locator('[data-testid="fit-coverage"]').textContent()}"`)
        }
      }
    }
    swept.sort((a, b) => a - b)
    const candidates = [...seen].sort((a, b) => b.m - a.m).slice(0, 3)
    console.log(`swept ${swept.length} positions — median ` +
      `${(swept[swept.length >> 1] * 100).toFixed(1)}%, top candidates ` +
      candidates.map((c) => `${(c.m * 100).toFixed(1)}%@[${c.at}]`).join(' '))

    // RE-MEASURE the candidates, and judge on those numbers alone.
    //
    // A sweep reading is taken while the navigator is being driven position to
    // position, and the spectrum and the model overlay arrive as SEPARATE
    // pushes into the panel — so a single read can pair this position's data
    // with the previous position's model. That is not noise around a true
    // value, it is a different quantity: CI has logged 47%, 73.7%, 82.6%,
    // 191.6% for positions that measure 2-12% the moment they are re-read.
    //
    // No wait fixes this from the outside. Quiescence cannot tell "finished"
    // from "not started"; waiting on the caret's fit_state does not help
    // either, because that state and the panel push are applied by DIFFERENT
    // consumers (the React wizard vs the plot iframe), so ordering on the wire
    // does not give ordering on screen.
    //
    // So: sweep to FIND suspicious positions, then park on each one and
    // measure it properly. A torn read can now only cost sensitivity (a false
    // candidate crowding out a real one), never a false failure — and taking
    // three candidates rather than one buys that sensitivity back.
    let worst = { m: 0, at: candidates[0].at }
    for (const c of candidates) {
      await goTo(c.at[0], c.at[1])
      const m = await settledMisfit()
      console.log(`  re-measured [${c.at}]: sweep ${(c.m * 100).toFixed(1)}% ` +
        `-> ${(m * 100).toFixed(1)}%`)
      if (m > worst.m) worst = { m, at: c.at }
    }
    console.log(`worst after re-measure: ${(worst.m * 100).toFixed(1)}% at [${worst.at}]`)

    await goTo(worst.at[0], worst.at[1])
    await page.screenshot({ path: `${SHOTS}/03-worst-position.png`, fullPage: true })
    const worstBefore = await settledMisfit()
    await page.locator('[data-testid="fit-spectrum"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/chi2/i, { timeout: 60_000 })
    await page.waitForTimeout(1_500)
    const worstAfter = await settledMisfit()
    await page.screenshot({ path: `${SHOTS}/04-worst-refitted.png`, fullPage: true })
    console.log(`worst position: ${(worstBefore * 100).toFixed(1)}% -> ` +
      `${(worstAfter * 100).toFixed(1)}% after Fit spectrum`)

    // The claim under test: the whole-scan fit leaves each position at its own
    // answer, so re-fitting the worst one should have little left to do. Both
    // sides of this comparison are RE-MEASURED at a parked navigator, which is
    // the only reason it can be trusted (see the note above the re-measure).
    //
    // Deliberately NOT asserted on: any statistic of `swept`. An earlier
    // version scored the 80th percentile of the sweep, on the theory that the
    // maximum of 25 samples was merely the noisiest one. That made things
    // worse — with torn reads in the set, p80 itself reached 25.1% against a
    // 10% bound, so the test failed 2.5x over on data that was never a
    // measurement of fit quality at all. `swept` stays for the median in the
    // log, which is a useful thing to eyeball, and for nothing else.
    //
    // Bounds are generous because each side is one sample: this should fail on
    // a real regression (a drawn curve that is not the fit made at that
    // position, which is the defect that prompted the spec), not on a wobble.
    expect(
      worstBefore,
      `the WORST swept position misses by ${(worstBefore * 100).toFixed(1)}% ` +
      `while its own dedicated fit reaches ${(worstAfter * 100).toFixed(1)}%`,
    ).toBeLessThan(Math.max(worstAfter * 2.5, 0.20))

    // ── the reported symptom: press Fit spectrum at an ALREADY fitted
    //    position. If it improves the curve, what was drawn was not the fit
    //    that was made here.
    await goTo(16, 16)
    await page.waitForTimeout(1_000)
    await page.locator('[data-testid="fit-spectrum"]').click()
    await expect(page.locator('[data-testid="fit-status"]'))
      .toContainText(/chi2/i, { timeout: 60_000 })
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/02-after-fit-spectrum.png`, fullPage: true })
    const afterOne = await settledMisfit()
    console.log(`misfit after Fit spectrum here    : ${(afterOne * 100).toFixed(1)}% of range`)

    expect(
      afterAll,
      `the model drawn after Fit all Spectra misses the spectrum by ` +
      `${(afterAll * 100).toFixed(1)}% of its range — fitting one spectrum ` +
      `here gets to ${(afterOne * 100).toFixed(1)}%`,
    ).toBeLessThan(0.10)

    expect(
      afterAll,
      `"Fit spectrum" improved an already-fitted position from ` +
      `${(afterAll * 100).toFixed(1)}% to ${(afterOne * 100).toFixed(1)}% — ` +
      `the model on screen was not this position's fit`,
    ).toBeLessThan(afterOne * 1.5 + 0.01)

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-20).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
