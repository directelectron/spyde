/**
 * presentation_editing.spec.ts — presentation authoring fixes, end-to-end.
 *
 * Drives the real app to verify:
 *   1) The "+ Add slide" menu offers FOUR options — Add text / split / title /
 *      figure slide — and "Add figure slide" creates a placeholder figure cell as
 *      its own slide.
 *   2) A split slide is fully EDITABLE in a presentation: text edits persist, and
 *      the layout dropdown offers the four arrangements (left/right/top/bottom),
 *      persisting the choice.
 *   3) Present mode → presenter view has an explicit EXIT (back to audience + out
 *      of Present) — the presenter dashboard no longer traps the user.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const { launchApp, backendErrorLines } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'presentation_editing_shots')

async function reportDoc(page: any): Promise<any> {
  return await page.evaluate(() => (window as any)._spyde_test_report?.())
}

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: false })
  const { page } = ctx
  await page.waitForTimeout(1200)
  const tour = page.getByTestId('tour-close')
  if (await tour.count()) await tour.click().catch(() => {})
  await page.getByTestId('toggle-report').click()
  await page.getByTestId('report-new-presentation-card').click()
  await expect(page.getByTestId('report-type-badge')).toHaveText('Presentation')
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

test('1) Add-slide menu has 4 options incl. figure; Add figure slide → placeholder', async () => {
  const { page } = ctx
  await page.getByTestId('report-add-slide').click()
  await expect(page.getByTestId('report-add-slide-menu')).toBeVisible()
  for (const t of ['add-slide-text', 'add-slide-split', 'add-slide-title', 'add-slide-figure']) {
    await expect(page.getByTestId(t), `${t} missing`).toBeVisible()
  }
  await page.screenshot({ path: join(SHOTS, '01-add-slide-menu.png') })
  await page.getByTestId('add-slide-figure').click()
  await page.waitForTimeout(400)
  const doc = await reportDoc(page)
  const fig = doc?.cells?.find((c: any) => c.cell_type === 'figure' && c.placeholder)
  expect(fig, 'no placeholder figure slide created').toBeTruthy()
  expect(fig.slide_break).toBe(true)
  ctx.assertNoJsErrors()
})

test('2) split slide is editable — text persists, layout dropdown has 4 arrangements', async () => {
  const { page } = ctx
  await page.getByTestId('report-add-slide').click()
  await page.getByTestId('add-slide-split').click()
  await page.waitForTimeout(400)
  const doc = await reportDoc(page)
  const split = doc.cells.find((c: any) => c.cell_type === 'split')
  expect(split).toBeTruthy()
  const id = split.id

  // Text edit persists.
  await page.getByTestId(`report-split-rendered-${id}`).dblclick()
  const ta = page.locator(`[data-testid="report-splitcell-${id}"] textarea`)
  await expect(ta).toBeVisible()
  await ta.fill('Slide body text')
  await ta.blur()
  await page.waitForTimeout(400)
  expect((await reportDoc(page)).cells.find((c: any) => c.id === id).source).toBe('Slide body text')

  // Layout dropdown → 4 arrangements; pick "text-top" (stacked) and confirm persist.
  await page.getByTestId(`report-splitcell-${id}`).hover()
  await page.getByTestId(`report-split-layout-${id}`).click()
  const menu = page.getByTestId(`report-split-layout-menu-${id}`)
  await expect(menu).toBeVisible()
  for (const l of ['text-left', 'text-right', 'text-top', 'text-bottom']) {
    await expect(page.getByTestId(`report-split-layout-${id}-${l}`), `layout ${l} missing`).toBeVisible()
  }
  await page.screenshot({ path: join(SHOTS, '02-split-layout-menu.png') })
  await page.getByTestId(`report-split-layout-${id}-text-top`).click()
  await page.waitForTimeout(300)
  expect((await reportDoc(page)).cells.find((c: any) => c.id === id).split_layout).toBe('text-top')
  ctx.assertNoJsErrors()
})

test('3) presenter view has an explicit exit', async () => {
  const { page } = ctx
  // Enter Present mode.
  await page.getByTestId('report-present').click()
  await expect(page.getByTestId('present-mode')).toBeVisible({ timeout: 10_000 })
  // Toggle presenter view (S).
  await page.keyboard.press('s')
  await expect(page.getByTestId('presenter-view')).toBeVisible({ timeout: 5_000 })
  // The exit controls are present + clickable (not covered).
  await expect(page.getByTestId('presenter-exit-view')).toBeVisible()
  const exitPresent = page.getByTestId('presenter-exit-present')
  await expect(exitPresent).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '03-presenter-exit.png') })
  // "Audience view" leaves presenter view but stays in Present.
  await page.getByTestId('presenter-exit-view').click()
  await expect(page.getByTestId('presenter-view')).toBeHidden({ timeout: 5_000 })
  await expect(page.getByTestId('present-mode')).toBeVisible()
  // Re-enter presenter and exit Present entirely.
  await page.keyboard.press('s')
  await expect(page.getByTestId('presenter-view')).toBeVisible({ timeout: 5_000 })
  await page.getByTestId('presenter-exit-present').click()
  await expect(page.getByTestId('present-mode')).toBeHidden({ timeout: 5_000 })
  ctx.assertNoJsErrors()
})

test('4) no backend tracebacks', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[pres] backend errors:\n' + errs.join('\n'))
  expect(errs).toEqual([])
})
