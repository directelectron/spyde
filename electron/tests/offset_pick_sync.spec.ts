/**
 * offset_pick_sync.spec.ts — the Axes table "+" origin pick must always agree
 * with the crosshair that is actually on the plot: **button ON ⟺ crosshair
 * alive**.
 *
 * Regression for "the offset + button is turned off after dragging the
 * crosshair and then when the pointer leaves the plot, but the crosshair
 * remains and you have to toggle it on/off": the dock kept its own boolean and
 * a `useEffect` cleared it on every active-window change, WITHOUT telling the
 * backend to remove the widget. So focusing another window (clicking the
 * navigator, the usual next move after setting an origin) darkened the button
 * while the crosshair stayed live, and the next click then targeted the newly
 * active plot — the stale crosshair could never be dismissed.
 *
 * The state is backend-owned now (`offset_pick` messages, per window), so this
 * spec drives the real windows and asserts the button against the real widget
 * list, not against what the renderer last guessed.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount, sigWindow, navWindow } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'offset_pick_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 60_000)
  await page.waitForTimeout(2500)
})
test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(180_000)

/** The anyplotlib fig id of the signal (or navigator) window's figure. */
async function figIdOf(page: any, navigator: boolean): Promise<string> {
  return page.evaluate((wantNav: boolean) => {
    for (const s of Array.from(document.querySelectorAll('[data-testid="subwindow"]'))) {
      const tid = s.querySelector('iframe')?.getAttribute('data-testid') ?? ''
      const crumb = s.querySelector('[data-testid="window-breadcrumb"]')?.textContent ?? ''
      if (tid.startsWith('figure-') && crumb.startsWith('N-') === wantNav) {
        return tid.slice('figure-'.length)
      }
    }
    return ''
  }, navigator)
}

test('the + toggle tracks the crosshair that is really on the plot', async () => {
  const { page } = ctx
  const toggle = page.getByTestId('offset-pick-toggle')
  const pickOn = async () => (await toggle.getAttribute('data-on')) === 'true'
  /** The ORANGE origin crosshair widgets currently on a figure. */
  const originCrosshairs = async (figId: string) => page.evaluate((f: string) =>
    ((window as any)._spyde_test_widgets(f) ?? []).filter(
      (w: any) => w.type === 'crosshair' && w.data?.color === '#ffae57'),
  figId)

  const sig = sigWindow(page)
  const nav = navWindow(page)
  await sig.getByTestId('subwindow-titlebar').click()
  await page.waitForTimeout(500)
  const sigFig = await figIdOf(page, false)
  expect(sigFig, 'no signal figure found').toBeTruthy()

  // ── 1. toggle ON: the button lights AND a crosshair lands on the plot ────
  await toggle.click()
  await expect(page.getByTestId('offset-pick-hint')).toBeVisible({ timeout: 10_000 })
  expect(await pickOn(), 'the + did not light up').toBe(true)
  const cross = (await originCrosshairs(sigFig))[0]
  expect(cross, 'toggling + put no crosshair on the signal plot').toBeTruthy()
  await page.screenshot({ path: join(SHOTS, '01-pick-on.png') })

  // ── 2. DRAG the crosshair with a real pointer: the offsets follow and the
  //       button stays on ───────────────────────────────────────────────────
  const fb = (await sig.locator('iframe').first().boundingBox())!
  const scale = 231 / 128                      // image px → screen px (128² frame)
  const sx = fb.x + 84.6 + (cross.data.cx + 0.5) * scale
  const sy = fb.y + 20 + (cross.data.cy + 0.5) * scale
  const offsetBefore = await page.getByTestId('axis-2-offset').textContent()
  await page.mouse.move(sx, sy)
  await page.mouse.down()
  await page.mouse.move(sx + 12, sy + 16, { steps: 8 })
  await page.mouse.move(sx + 26, sy + 32, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(1000)
  expect(await page.getByTestId('axis-2-offset').textContent(),
    'the drag never reached the axes (the crosshair was not grabbed)')
    .not.toBe(offsetBefore)
  expect(await pickOn(), 'the + went dark during the drag').toBe(true)

  // ── 3. pointer LEAVES the plot — the button must not follow it out ───────
  await page.mouse.move(fb.x + fb.width + 250, fb.y + fb.height + 200, { steps: 14 })
  await page.waitForTimeout(1000)
  expect(await pickOn(), 'the + went dark when the pointer left the plot').toBe(true)
  expect((await originCrosshairs(sigFig)).length,
    'the crosshair vanished when the pointer left').toBe(1)
  await page.screenshot({ path: join(SHOTS, '02-after-drag-and-leave.png') })

  // ── 4. focus ANOTHER window: the "+" now reports THAT window (no crosshair
  //       there), while the signal plot keeps its own ─────────────────────
  await nav.getByTestId('subwindow-titlebar').click()
  await page.waitForTimeout(900)
  expect(await pickOn(), 'the navigator has no crosshair, so its + must be off')
    .toBe(false)
  expect((await originCrosshairs(sigFig)).length,
    'focusing the navigator removed the signal plot crosshair').toBe(1)
  await page.screenshot({ path: join(SHOTS, '03-navigator-focused.png') })

  // ── 5. …and back: the window that OWNS the crosshair shows + lit again.
  //       (This is the regression: it used to come back dark, so the crosshair
  //       could only be removed by toggling twice.) ────────────────────────
  await sig.getByTestId('subwindow-titlebar').click()
  await page.waitForTimeout(900)
  expect(await pickOn(),
    'back on the window that owns the crosshair, the + must be lit').toBe(true)
  await page.screenshot({ path: join(SHOTS, '04-back-on-signal.png') })

  // ── 6. ONE click turns it off and the crosshair really goes ─────────────
  await toggle.click()
  await page.waitForTimeout(1000)
  expect(await pickOn(), 'the + stayed lit after toggling off').toBe(false)
  expect((await originCrosshairs(sigFig)).length,
    'the crosshair survived a single toggle-off').toBe(0)
  await expect(page.getByTestId('offset-pick-hint')).toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '05-toggled-off.png') })

  ctx.assertNoJsErrors()
})
