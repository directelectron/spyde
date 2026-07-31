/**
 * sidebar_unit_latex.spec.ts — the Plot Control dock's Axes table renders a
 * LaTeX units string instead of showing it literally.
 *
 * HyperSpy stores units as raw LaTeX ("$\AA^{-1}$"). The contract this pins:
 *   1. NOT editing → the cell shows RENDERED maths (a KaTeX <math> element,
 *      Å⁻¹), never the raw "$...$" text.
 *   2. Plain units ("px") stay plain text — no KaTeX, no <math>.
 *   3. Clicking to edit shows the RAW string, so it round-trips.
 *   4. Committing stores the raw string unchanged (edit to "$\mu$m", and the
 *      value that comes BACK from the backend is that same raw string,
 *      re-rendered).
 *
 * si_grains carries "$\AA^{-1}$" on its two signal axes and "px" on the two
 * navigation axes, so one dataset exercises both branches.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'unit_latex_shots')
const RAW = '$\\AA^{-1}$'

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  await backendAction(ctx.page, 'load_test_data_si_grains')
  // navigator + signal window
  await waitForSubwindowCount(ctx.page, 2, 60_000)
  await ctx.page.waitForSelector('[data-testid="axes-table"]', { timeout: 60_000 })
})
test.afterAll(async () => { await ctx?.app?.close() })

/** The units cell of the first SIGNAL axis (si_grains: index 2). */
const unitsCell = (page, i: number) => page.getByTestId(`axis-${i}-units`)

test('LaTeX units render as maths, plain units stay plain', async () => {
  const { page } = ctx
  const cell = unitsCell(page, 2)
  await expect(cell).toBeVisible()

  // 1. rendered, not literal. (MathML text extraction puts each atom on its own
  // line — "A\n˚\n−\n1" — so assert on the ATOMS, not a composed "Å⁻¹".)
  const text = (await cell.innerText()).trim()
  console.log('[units] signal-axis cell text =', JSON.stringify(text))
  expect(text).not.toContain('$')
  expect(text).not.toContain('\\')
  expect(text).toContain('A')
  expect(text).toContain('1')
  // KaTeX MathML actually landed in the DOM, and the raw string is one hover away.
  expect(await cell.locator('math').count()).toBeGreaterThan(0)
  await expect(cell.getByTestId('unit-latex')).toHaveAttribute('title', RAW)

  // 2. a plain unit is NOT pushed through KaTeX
  const nav = unitsCell(page, 0)
  expect((await nav.innerText()).trim()).toBe('px')
  expect(await nav.locator('math').count()).toBe(0)

  await page.getByTestId('plot-control-dock').screenshot({
    path: join(SHOTS, '01-rendered-dock.png') })
  await page.screenshot({ path: join(SHOTS, '02-rendered-full.png') })
  ctx.assertNoJsErrors()
})

test('clicking the cell exposes the RAW string, Escape restores the render', async () => {
  const { page } = ctx
  await unitsCell(page, 2).click()
  const input = page.getByTestId('axis-2-units-input')
  await expect(input).toBeVisible()
  await expect(input).toHaveValue(RAW)

  await page.getByTestId('plot-control-dock').screenshot({
    path: join(SHOTS, '03-editing-raw-dock.png') })
  await page.screenshot({ path: join(SHOTS, '04-editing-raw-full.png') })

  await input.press('Escape')
  await expect(input).toBeHidden()
  expect(await unitsCell(page, 2).locator('math').count()).toBeGreaterThan(0)
  ctx.assertNoJsErrors()
})

test('committing stores the raw string unchanged (round trip)', async () => {
  const { page } = ctx
  await unitsCell(page, 2).click()
  const input = page.getByTestId('axis-2-units-input')
  await input.fill('$\\mu$m')
  await input.press('Enter')

  // The backend writes it to axes_manager and re-emits the table; what comes
  // back must be the RAW string we typed (rendered again, not the glyphs).
  const cell = unitsCell(page, 2)
  await expect(cell.getByTestId('unit-latex')).toHaveAttribute('title', '$\\mu$m',
    { timeout: 20_000 })
  // …a real mu glyph, whichever variant KaTeX picked (Greek small, micro sign,
  // or the mathematical-italic codepoint) — never the literal "\mu".
  const shown = (await cell.innerText()).trim()
  console.log('[units] after commit, cell text =', JSON.stringify(shown))
  expect(shown).not.toContain('\\')
  expect(shown).toMatch(/[μµ\u{1D707}]/u)
  await page.getByTestId('plot-control-dock').screenshot({
    path: join(SHOTS, '05-committed-mu-m.png') })

  // …and put it back, proving the same round trip in the other direction.
  await cell.click()
  const again = page.getByTestId('axis-2-units-input')
  await expect(again).toHaveValue('$\\mu$m')
  await again.fill(RAW)
  await again.press('Enter')
  await expect(unitsCell(page, 2).getByTestId('unit-latex'))
    .toHaveAttribute('title', RAW, { timeout: 20_000 })
  ctx.assertNoJsErrors()
})
