/**
 * action_scoping.spec.ts — an action's artefacts must not outlive its caret.
 *
 * Two leaks of the same shape:
 *
 *  - **Remove Background** left its green span on the plot forever. It had the
 *    whole backend half — `bg_open`/`bg_close`, a controller whose `remove()`
 *    deletes the widget — but no renderer caret, so it was reached through the
 *    plain toolbar path: the click fired `bg_open`, and nothing ever unmounted
 *    to fire `bg_close`.
 *  - **Fit** left its components-maps window behind, still showing a model it
 *    had stopped tracking.
 *
 * The staged contract is `<key>_open` on mount / `<key>_close` on unmount, and
 * FloatingToolbar renders ONE caret at a time — so selecting another action is
 * what has to clean up. This drives exactly that.
 *
 * Run: npx playwright test tests/action_scoping.spec.ts \
 *        --project=electron --reporter=line --retries=0
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow,
} = require('./_harness.cjs')

const SHOTS = 'action_scoping_shots'

test('closing an action removes what it put on the plot', async () => {
  test.setTimeout(600_000)

  const ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page, backend, assertNoJsErrors } = ctx

  try {
    await backendAction(page, 'tutorial_load', { name: 'spectroscopy' })
    await waitForSubwindowCount(page, 2, 180_000)
    await page.waitForTimeout(2_500)

    const sigFig = await page.evaluate(() => {
      for (const s of Array.from(document.querySelectorAll('[data-testid="subwindow"]'))) {
        const tid = s.querySelector('iframe')?.getAttribute('data-testid') ?? ''
        const crumb = s.querySelector('[data-testid="window-breadcrumb"]')?.textContent ?? ''
        if (tid.startsWith('figure-') && !crumb.startsWith('N-')) return tid.slice('figure-'.length)
      }
      return ''
    })
    const widgets = async () =>
      page.evaluate((f) => (window as any)._spyde_test_widgets(f), sigFig)

    const openAction = async (name: string) => {
      const sig = sigWindow(page)
      await sig.getByTestId('subwindow-titlebar').hover()
      await sig.getByTestId(`action-btn-${name}`).click()
      await page.waitForTimeout(1_200)
    }

    const baseline = (await widgets()).length

    // ── Remove Background: the span must go when the caret does ──────────
    await openAction('Remove Background')
    await expect(page.locator('[data-testid="bg-wizard"]')).toBeVisible({ timeout: 20_000 })
    await page.waitForTimeout(1_000)
    const withSpan = await widgets()
    await page.screenshot({ path: `${SHOTS}/01-background-open.png`, fullPage: true })
    expect(
      withSpan.filter((w: any) => w.type === 'range').length,
      'Remove Background put no span on the plot',
    ).toBeGreaterThan(0)

    // The caret follows the band, so its fields must show where the band IS —
    // `bg_state` has to be re-broadcast to the renderer for that to happen.
    const [x0, x1] = await Promise.all([
      page.locator('[data-testid="bg-x0"]').inputValue().then(Number),
      page.locator('[data-testid="bg-x1"]').inputValue().then(Number),
    ])
    console.log(`background band reported as ${x0} .. ${x1}`)
    expect(x1, 'the caret never heard where the band is (bg_state not relayed)')
      .toBeGreaterThan(x0)

    // ── the preview must FOLLOW the band, not wait for the release ───────
    const bgCurve = async () => page.evaluate((f) => {
      const hook = (window as any)._spyde_test_panel_json
      for (const raw of hook ? hook(f) : []) {
        const d = JSON.parse(raw)
        for (const ln of d.extra_lines ?? []) {
          if (ln.label !== 'background' || !ln.data_b64) continue
          const bin = atob(ln.data_b64)
          const b = new Uint8Array(bin.length)
          for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i)
          const v = Array.from(new Float64Array(b.buffer))
          return { n: v.length, sum: v.reduce((a, c) => a + c, 0) }
        }
      }
      return null
    }, sigFig)

    const span = (await widgets()).find((w: any) => w.type === 'range')
    const postSpan = (x0: number, x1: number, type: string) =>
      page.evaluate(({ f, panel, id, a, b, t }) => {
        window.postMessage({
          type: 'awi_event', figId: f,
          data: JSON.stringify({
            source: 'js', panel_id: panel, widget_id: id,
            event_type: t, x0: a, x1: b,
          }),
        }, '*')
      }, { f: sigFig, panel: span.panel_id, id: span.id, a: x0, b: x1, t: type })

    const before = await bgCurve()
    expect(before, 'no background preview curve is drawn').toBeTruthy()
    // A pointer_MOVE only — the release is what used to be the only thing
    // that redrew, and reading the geometry off the event rather than off
    // `event.source` meant even that never happened.
    await postSpan(55, 95, 'pointer_move')
    await page.waitForTimeout(1_500)
    const during = await bgCurve()
    expect(
      Math.abs((during!.sum - before!.sum) / (before!.sum || 1)),
      'the background preview did not follow the band mid-drag',
    ).toBeGreaterThan(1e-6)
    await page.screenshot({ path: `${SHOTS}/01b-band-dragged.png`, fullPage: true })

    await page.locator('[data-testid="bg-close"]').click()
    await page.waitForTimeout(1_500)
    await page.screenshot({ path: `${SHOTS}/02-background-closed.png`, fullPage: true })
    expect(
      (await widgets()).length,
      'the background span stayed on the plot after the caret closed',
    ).toBe(baseline)

    // ── and again when another action takes over, not just on the X ──────
    await openAction('Remove Background')
    await expect(page.locator('[data-testid="bg-wizard"]')).toBeVisible({ timeout: 20_000 })
    await page.waitForTimeout(1_000)
    await openAction('Fit')
    await expect(page.locator('[data-testid="fit-wizard"]')).toBeVisible({ timeout: 20_000 })
    await page.waitForTimeout(1_500)
    const nowRanges = (await widgets()).filter((w: any) => w.type === 'range').length
    expect(
      nowRanges,
      'switching from Remove Background to Fit left the span behind',
    ).toBe(0)

    // ── Fit: its maps window must go when the caret does ─────────────────
    await page.locator('[data-testid="fit-add-toggle"]').click()
    await expect(page.locator('[data-testid="fit-add-Gaussian"]')).toBeVisible({ timeout: 30_000 })
    await page.locator('[data-testid="fit-add-Gaussian"]').click()
    await page.waitForTimeout(1_200)
    const withMaps = await page.getByTestId('subwindow').count()
    await page.screenshot({ path: `${SHOTS}/03-fit-open.png`, fullPage: true })

    await page.locator('[data-testid="fit-close"]').click()
    await page.waitForTimeout(2_000)
    await page.screenshot({ path: `${SHOTS}/04-fit-closed.png`, fullPage: true })
    expect(
      await page.getByTestId('subwindow').count(),
      'the Fit components window outlived the caret that owns it',
    ).toBeLessThan(withMaps)
    expect(
      (await widgets()).length,
      'the Fit drag handles stayed on the plot after the caret closed',
    ).toBe(baseline)

    await assertNoJsErrors()
  } finally {
    console.log(`\n──── backend log (tail) ────\n${backend.logBuffer.slice(-15).join('\n')}\n`)
    await ctx.app.close().catch(() => {})
  }
})
