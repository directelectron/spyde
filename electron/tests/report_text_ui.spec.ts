/**
 * report_text_ui.spec.ts — the Report text-cell UI upgrades, end-to-end.
 *
 * No dataset needed (text cells only) → SPYDE_NO_DASK fast launch. Drives:
 *   • KaTeX math: inline `$…$` and display `$$…$$` render as MathML in the
 *     rendered cell; a `$` inside a code span stays literal.
 *   • The formatting toolbar: Bold wraps the selection in `**`, commits to a
 *     real <strong>; Ctrl+B does the same.
 *   • The header "Aa" text-size cycle actually changes the rendered font size.
 *
 * Screenshots to report_text_ui_shots/ — each Read by the author (a blank
 * panel is a failure even when selectors pass).
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdtempSync, existsSync, rmSync, readFileSync } from 'fs'
import { tmpdir } from 'os'
const { launchApp } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_text_ui_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(120_000)

test.beforeAll(async () => {
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1000)
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await page.getByTestId('report-new').click()
  await expect(page.getByTestId('report-body')).toBeVisible()
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

// The one markdown cell this spec keeps re-editing.
const rendered = () => ctx.page.locator('[data-testid^="report-cell-rendered-"]').first()
const textarea = () => ctx.page.locator('[data-testid^="report-cell-textarea-"]').first()

async function setCellSource(src: string) {
  const { page } = ctx
  await rendered().dblclick()
  await expect(textarea()).toBeVisible()
  await textarea().fill(src)
  await textarea().press('Control+Enter')
  await expect(page.locator('[data-testid^="report-cell-textarea-"]')).toHaveCount(0)
}

test('1) inline $…$ and display $$…$$ render as KaTeX MathML', async () => {
  const { page } = ctx
  await page.getByTestId('report-add-text').click()
  await setCellSource(
    'Einstein wrote $E = mc^2$ inline.\n\n' +
    '$$\n\\int_0^1 x^2\\,dx = \\frac{1}{3}\n$$\n\n' +
    'And code keeps `$dollars$` literal.',
  )
  const cell = rendered()
  // Inline math → a .katex span holding real MathML.
  await expect(cell.locator('.katex math').first()).toBeAttached()
  // Display math → the centered .katex-display block, math display="block".
  await expect(cell.locator('.katex-display math[display="block"]')).toBeAttached()
  // The code span keeps its dollars as text — exactly one code el, no math in it.
  await expect(cell.locator('code')).toHaveText('$dollars$')
  // Two math regions total (inline + display) — the code-span $ was NOT parsed.
  expect(await cell.locator('.katex').count()).toBe(2)
  await page.screenshot({ path: join(SHOTS, '01-math-rendered.png') })
  ctx.assertNoJsErrors()
})

test('2) toolbar Bold wraps the selection; commit renders <strong>', async () => {
  const { page } = ctx
  await rendered().dblclick()
  await expect(textarea()).toBeVisible()
  await textarea().fill('make this bold please')
  // Select "this bold" (chars 5..14) and hit the toolbar Bold button.
  await textarea().evaluate((el: HTMLTextAreaElement) => {
    el.focus(); el.setSelectionRange(5, 14)
  })
  await page.locator('[data-testid^="report-fmt-bold-"]').click()
  await expect(textarea()).toHaveValue('make **this bold** please')
  // The textarea kept focus (mousedown preventDefault) — no blur-commit.
  await expect(page.locator('[data-testid^="report-cell-textarea-"]')).toHaveCount(1)
  // Ctrl+B toggles the same wrap back OFF via the keyboard path.
  await textarea().evaluate((el: HTMLTextAreaElement) => {
    el.focus(); el.setSelectionRange(7, 16)   // "this bold" inside the markers
  })
  await textarea().press('Control+b')
  await expect(textarea()).toHaveValue('make this bold please')
  // Re-bold + commit → a real <strong> in the rendered view.
  await textarea().evaluate((el: HTMLTextAreaElement) => {
    el.focus(); el.setSelectionRange(5, 14)
  })
  await page.locator('[data-testid^="report-fmt-bold-"]').click()
  // Shot WITH the editor + toolbar open (the committed views are shot elsewhere).
  await page.screenshot({ path: join(SHOTS, '02a-toolbar-open.png') })
  await textarea().press('Control+Enter')
  await expect(rendered().locator('strong')).toHaveText('this bold')
  await page.screenshot({ path: join(SHOTS, '02-toolbar-bold.png') })
  ctx.assertNoJsErrors()
})

test('3) heading + math-block toolbar buttons produce H2 and $$ block', async () => {
  const { page } = ctx
  await rendered().dblclick()
  await textarea().fill('Section title')
  await textarea().evaluate((el: HTMLTextAreaElement) => {
    el.focus(); el.setSelectionRange(0, 0)
  })
  await page.locator('[data-testid^="report-fmt-h2-"]').click()
  await expect(textarea()).toHaveValue('## Section title')
  // Append a math block at the end via the √x button.
  await textarea().evaluate((el: HTMLTextAreaElement) => {
    const n = el.value.length
    el.focus(); el.setSelectionRange(n, n)
  })
  await page.locator('[data-testid^="report-fmt-math-"]').click()
  await expect(textarea()).toHaveValue(/## Section title\n\$\$\nE = mc\^2\n\$\$\n/)
  await textarea().press('Control+Enter')
  await expect(rendered().locator('h2')).toHaveText('Section title')
  await expect(rendered().locator('.katex-display math[display="block"]')).toBeAttached()
  await page.screenshot({ path: join(SHOTS, '03-toolbar-heading-math.png') })
  ctx.assertNoJsErrors()
})

test('4) static HTML export carries the MathML through (self-contained math)', async () => {
  const { page } = ctx
  const workDir = mkdtempSync(join(tmpdir(), 'spyde-mdmath-'))
  const htmlPath = join(workDir, 'math.html')
  // Stub the MAIN-process export dialog (the report_export.spec.ts pattern) so
  // the UI export flow never blocks on an OS picker.
  await ctx.app.evaluate(({ ipcMain }: any, p: string) => {
    ipcMain.removeHandler('report:export-dialog')
    ipcMain.handle('report:export-dialog', async () => p)
  }, htmlPath)
  await page.getByTestId('report-export-toggle').click()
  await page.getByTestId('export-html-static').click()
  await expect
    .poll(() => existsSync(htmlPath), { timeout: 15_000, message: 'export never written' })
    .toBe(true)
  const html = readFileSync(htmlPath, 'utf-8')
  // The cell committed in test 3 (H2 + $$ block) exports REAL MathML — no
  // KaTeX CSS/fonts, no <pre> fallback.
  expect(html).toContain('Section title')
  expect(html).toContain('<math')
  expect(html).toContain('display="block"')
  // No raw-source fallback <pre> — the html cache carried real rendered HTML.
  // (The substring "md-src" alone also lives in the article CSS rule.)
  expect(html).not.toContain('<pre class="md-src"')
  rmSync(workDir, { recursive: true, force: true })
  ctx.assertNoJsErrors()
})

test('5) the Aa header button cycles the rendered text size', async () => {
  const { page } = ctx
  const sizeOf = async () =>
    await rendered().evaluate((el: HTMLElement) => getComputedStyle(el).fontSize)
  const before = await sizeOf()
  await page.getByTestId('report-md-size').click()
  const after = await sizeOf()
  expect(after).not.toBe(before)
  // Cycle the remaining steps → wraps back to the starting size (4 sizes).
  await page.getByTestId('report-md-size').click()
  await page.getByTestId('report-md-size').click()
  await page.getByTestId('report-md-size').click()
  expect(await sizeOf()).toBe(before)
  await page.screenshot({ path: join(SHOTS, '04-md-size-cycle.png') })
  ctx.assertNoJsErrors()
})
