/**
 * caret_above_windows.spec.ts — an open caret must stay clickable when another
 * window is focused over it.
 *
 * THE BUG: `FloatingToolbar` (and the wizard it opens) renders INSIDE
 * `SubWindow`, whose root is positioned with a z-index and therefore creates a
 * stacking context. Nothing inside can paint above a sibling window with a
 * higher z-index, whatever z-index it gives itself — so raising the caret's own
 * z-index does nothing. A window focused after the caret opened covers it, and
 * its figure iframe swallows every click aimed at the caret.
 *
 * That is how the Electron 34 -> 44 upgrade reddened ~25 e2e tests: the fit,
 * DPC, strain and orientation workflows all open windows AFTER their caret, so
 * on CI every one of them timed out with "iframe ... intercepts pointer
 * events". It reproduces on any platform once the overlap is forced, which is
 * what this spec does rather than relying on where windows happen to land.
 *
 * The assertion is `elementFromPoint` at the control's centre, because that is
 * exactly the question Playwright's click asks: what does a click here hit?
 */
import { test, expect } from '@playwright/test'
const {
  launchApp, backendAction, waitForSubwindowCount, titlebarGrabPoint,
} = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
})
test.afterAll(async () => { await ctx?.app?.close() })

test('a caret stays hittable when another window is focused over it', async () => {
  const { page } = ctx

  // Open a caret on the signal window.
  const sig = page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }),
  }).first()
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Center Zero Beam').click()
  const wizard = page.getByTestId('center-zero-beam-wizard')
  await expect(wizard).toBeVisible()

  const target = wizard.getByTestId('czb-close')
  const box = (await target.boundingBox())!
  const cx = Math.round(box.x + box.width / 2)
  const cy = Math.round(box.y + box.height / 2)

  // Force the overlap instead of hoping for it: move the OTHER window so it
  // covers that point, then focus it so it takes the higher z-index. This is
  // the situation a fit/DPC/strain run creates by opening result windows.
  const nav = page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }),
  }).first()
  const navBox = (await nav.boundingBox())!
  // Grab RIGHT of the breadcrumb pill: the pill is an HTML5 drag source that
  // stops pointerdown, so grabbing it starts a DnD payload instead of a window
  // move (mdi_layout.spec.ts makes the same point). Keep the drop math in
  // window coords via the grab offset.
  const grab = await titlebarGrabPoint(nav)
  const offX = grab.x - navBox.x
  const offY = grab.y - navBox.y
  // Put the navigator's TOP-LEFT just above/left of the caret control so its
  // body lands squarely over it.
  await page.mouse.move(grab.x, grab.y)
  await page.mouse.down()
  await page.mouse.move(cx - 60 + offX, cy - 60 + offY, { steps: 10 })
  await page.mouse.up()
  // The drag itself focuses the navigator, which is what lifts it above the
  // signal window in the unfixed build — no extra click needed (and once the
  // fix is in, the pinned window's toolbar sits over the navigator's title, so
  // clicking it would fail for a reason that has nothing to do with this test).
  await page.waitForTimeout(400)

  // Diagnostics first: an overlap that never happened would make this spec
  // pass for the wrong reason, which is worse than failing.
  const diag = await page.evaluate(([x, y]) => {
    const rect = (sel: string) => {
      const el = document.querySelector(sel) as HTMLElement | null
      if (!el) return null
      const r = el.getBoundingClientRect()
      return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }
    }
    const wins = [...document.querySelectorAll('[data-testid="subwindow"]')].map((el) => {
      const r = el.getBoundingClientRect()
      return {
        z: getComputedStyle(el as HTMLElement).zIndex,
        x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height),
        covers: (x as number) >= r.x && (x as number) <= r.right
              && (y as number) >= r.y && (y as number) <= r.bottom,
      }
    })
    const caretEl = document.querySelector('[data-testid="center-zero-beam-wizard"]') as HTMLElement | null
    return {
      point: { x, y },
      caret: rect('[data-testid="center-zero-beam-wizard"]'),
      caretZ: caretEl ? getComputedStyle(caretEl).zIndex : null,
      windows: wins,
      anyWindowCovers: wins.some((w) => w.covers),
    }
  }, [cx, cy])
  console.log('[caret-z] ' + JSON.stringify(diag, null, 2))
  expect(diag.anyWindowCovers,
    'setup failed: no window was moved over the caret control, so this spec '
    + 'would pass without testing anything').toBe(true)

  // The caret must still be what a click at that point would reach.
  const hit = await page.evaluate(([x, y]) => {
    const el = document.elementFromPoint(x as number, y as number)
    if (!el) return 'nothing'
    return `${el.tagName}:${el.getAttribute('data-testid') ?? ''}` +
      (el.closest('[data-testid="center-zero-beam-wizard"]') ? '|in-caret' : '')
  }, [cx, cy])
  console.log('[caret-z] hit =', hit)

  expect(hit, `a focused window covers the open caret — a click at its control `
    + `lands on ${hit} instead. The caret cannot escape SubWindow's stacking `
    + `context, so raising its own z-index cannot fix this.`).toContain('in-caret')
})
