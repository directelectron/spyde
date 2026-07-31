/**
 * dock_compact.spec.ts — the Plot Control dock on a SMALL screen.
 *
 * The dock is a fixed-height flex column of sections. On a laptop-height window
 * that column overflows, and the failure mode is invisible to headless tests: the
 * bottom sections (Layers, Navigator Selector) are simply cut off with no way to
 * reach them. So this shrinks the real window and checks two things — the primary
 * controls still fit above the fold, and everything below them is reachable by
 * scrolling — with a screenshot at each size.
 */
import { test, expect } from '@playwright/test'
const { launchApp, backendAction } = require('./_harness.cjs')

const SHOTS = 'dock_shots'
// A 13" laptop with the app maximised, minus chrome — the size that broke.
const SMALL = { w: 1280, h: 680 }

let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  await expect(ctx.page.getByTestId('subwindow').first()).toBeVisible({ timeout: 60_000 })
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(120_000)

test('the dock stays usable at laptop height', async () => {
  const { page, app } = ctx
  await expect(page.getByTestId('histogram')).toBeVisible({ timeout: 30_000 })
  await page.getByTestId('plot-control-dock').screenshot({ path: `${SHOTS}/01-default.png` })

  await app.evaluate(({ BrowserWindow }, s) => {
    BrowserWindow.getAllWindows()[0].setSize(s.w, s.h)
  }, SMALL)
  await expect.poll(async () => page.viewportSize()?.height ?? 0).toBeLessThan(SMALL.h + 1)

  // Above the fold: histogram + its buttons + colormap + signal type. These are
  // the controls reached constantly, so they must not need a scroll.
  const dockBox = (await page.getByTestId('plot-control-dock').boundingBox())!
  for (const id of ['histogram', 'clim-auto', 'clim-reset', 'colormap-select',
                    'signal-type-select']) {
    const box = (await page.getByTestId(id).boundingBox())!
    expect(box.y + box.height, `${id} is below the fold at ${SMALL.h}px`)
      .toBeLessThan(dockBox.y + dockBox.height)
  }
  await page.getByTestId('plot-control-dock').screenshot({ path: `${SHOTS}/02-small.png` })

  // Budget for the top section (label + widget + clim labels + button row +
  // padding). It was ~195px with the taller widget and a hint line under it;
  // this fails if either creeps back.
  const histSection = (await page.getByTestId('histogram-section').boundingBox())!
  expect(histSection.height, 'the histogram section grew back').toBeLessThan(140)

  // The CONTROLS stay put: the metadata panel is the elastic child that absorbs
  // the leftover height, so the sections below it (Layers, Navigator Selector)
  // are on screen without scrolling anything.
  await expect(page.getByTestId('selector-integrate')).toBeInViewport()
  const body = page.getByTestId('dock-body')
  expect(await body.evaluate((el) => el.scrollHeight - el.clientHeight),
    'the dock body scrolls — the pinned sections no longer fit')
    .toBeLessThanOrEqual(1)

  ctx.assertNoJsErrors()
})
