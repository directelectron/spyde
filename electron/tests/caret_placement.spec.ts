/**
 * caret_placement.spec.ts — a new window must not be PLACED on top of an open
 * caret.
 *
 * Z-order is deliberately NOT the subject here. A window that overlaps a caret
 * still comes to the front over it, and the caret stays open underneath — that
 * is the intended behaviour and this spec must not be read as forbidding it.
 * What it forbids is the app *choosing* to drop a brand-new window onto the
 * panel the user is working in.
 *
 * Why it matters: `findFreeSlot` packs a new window into the first spot that
 * collides with no existing WINDOW, and it used to know nothing about carets.
 * So a fit / DPC / strain / orientation run would place its own result window
 * squarely over the caret that launched it, and because a caret lives inside
 * its window's stacking context (SubWindow's root is positioned WITH a
 * z-index), it cannot paint above that new window whatever z-index it takes —
 * measured: a caret at z-index 1002 lost to a window at 11. The result window's
 * figure iframe then swallowed every click meant for the caret, which is what
 * reddened ~18 e2e tests on the Electron 34 -> 44 upgrade.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await waitForSubwindowCount(ctx.page, 2, 120_000)
})
test.afterAll(async () => { await ctx?.app?.close() })

/** Every subwindow's rect, in viewport coords. */
async function windowRects(page: any) {
  return page.evaluate(() =>
    [...document.querySelectorAll('[data-testid="subwindow"]')].map((el) => {
      const r = el.getBoundingClientRect()
      return { x: r.x, y: r.y, w: r.width, h: r.height }
    }))
}

const overlapArea = (a: any, b: any) =>
  Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
  * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))

test('windows opened while a caret is up are not placed on top of it', async () => {
  const { page } = ctx

  const sig = page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }),
  }).first()
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Center Zero Beam').click()
  const wizard = page.getByTestId('center-zero-beam-wizard')
  await expect(wizard).toBeVisible()

  const cb = (await wizard.boundingBox())!
  const caret = { x: cb.x, y: cb.y, w: cb.width, h: cb.height }
  const before = (await windowRects(page)).length

  // Open more windows while the caret is up — the situation every staged action
  // creates when it publishes its result.
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, before + 2, 120_000)
  await page.waitForTimeout(500)

  // The caret must still be open and still be what a click at its centre hits.
  await expect(wizard).toBeVisible()
  const after = await windowRects(page)
  const worst = after
    .map((w: any) => overlapArea(w, caret))
    .sort((a: number, b: number) => b - a)[0] ?? 0
  const caretArea = caret.w * caret.h

  console.log(`[caret-placement] caret ${Math.round(caretArea)}px², `
    + `worst window overlap ${Math.round(worst)}px² `
    + `(${Math.round((worst / caretArea) * 100)}%)`)

  // A sliver of overlap is tolerable (the search steps in 26px increments and
  // the caret rect handed to it is approximate for side placements); burying
  // the panel is not.
  expect(worst / caretArea,
    'a newly opened window was placed over the open caret — findFreeSlot is not '
    + 'treating the caret as occupied space').toBeLessThan(0.25)

  const hit = await page.evaluate(([x, y]) => {
    const el = document.elementFromPoint(x as number, y as number)
    return el?.closest('[data-testid="center-zero-beam-wizard"]') ? 'in-caret'
      : `${el?.tagName}:${el?.getAttribute('data-testid') ?? ''}`
  }, [Math.round(caret.x + caret.w / 2), Math.round(caret.y + 12)])
  expect(hit, 'the caret is covered, so its controls cannot be clicked')
    .toBe('in-caret')
})

test('a full area still keeps new windows off the open caret', async () => {
  // Once no free slot is left, a new window has to overlap SOMETHING. It used
  // to cascade from the top-left corner with no regard for carets, which is how
  // the vector-orientation IPF window landed on the caret that opened it on a
  // 1280-wide CI screen and swallowed the click on its Library tab.
  const { page } = ctx
  const wizard = page.getByTestId('center-zero-beam-wizard')
  await expect(wizard).toBeVisible()
  const cb = (await wizard.boundingBox())!
  const caret = { x: cb.x, y: cb.y, w: cb.width, h: cb.height }

  // Fill the area, then keep opening: every window from here on has nowhere
  // free to go.
  for (let round = 0; round < 5; round++) {
    const count = await page.getByTestId('subwindow').count()
    await backendAction(page, 'load_test_data_si_grains')
    await waitForSubwindowCount(page, count + 2, 120_000)
  }
  await page.waitForTimeout(500)

  await expect(wizard).toBeVisible()
  const worst = (await windowRects(page))
    .map((w: any) => overlapArea(w, caret))
    .sort((a: number, b: number) => b - a)[0] ?? 0
  console.log(`[caret-placement] full area: caret at ${JSON.stringify(caret)}, `
    + `worst window overlap ${Math.round((worst / (caret.w * caret.h)) * 100)}%`)
  expect(worst / (caret.w * caret.h),
    'with the area full, a new window was stacked over the open caret')
    .toBeLessThan(0.25)

  // Every corner of the panel must still take its own clicks, not just its
  // middle: a window over one corner is a control that silently does nothing.
  const inset = 12
  const points = [
    [caret.x + caret.w / 2, caret.y + caret.h / 2],
    [caret.x + inset, caret.y + inset], [caret.x + caret.w - inset, caret.y + inset],
    [caret.x + inset, caret.y + caret.h - inset],
    [caret.x + caret.w - inset, caret.y + caret.h - inset],
  ].map(([x, y]) => [Math.round(x), Math.round(y)])
  const covered = await page.evaluate((pts) => pts.filter(([x, y]) =>
    !document.elementFromPoint(x, y)?.closest('[data-testid="center-zero-beam-wizard"]')),
  points)
  expect(covered, 'parts of the caret are covered by a window placed on it')
    .toEqual([])
})
