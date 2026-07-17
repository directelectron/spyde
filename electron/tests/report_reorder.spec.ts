/**
 * report_reorder.spec.ts — figure-cell reordering in the Report sidebar.
 *
 * Real Dask + si_grains: build [md "Alpha", figure, md "Beta"], then drag the
 * FIGURE cell by its ⠿ handle onto the first markdown cell and assert the
 * figure lands first (renderer dragProps → report_move_cell). Also pins that
 * the figure cell's live iframe survives the move (no re-render blank).
 *
 * Screenshots to report_reorder_shots/ — each Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_reorder_shots')
const FIG_MIME = 'application/x-spyde-figure'

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(180_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2500)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

/** In-page native HTML5 drag with a shared DataTransfer (the proven pattern
 *  from report_sidebar.spec.ts). */
async function dragAndDrop(page: any, srcSelector: string, dstSelector: string) {
  return await page.evaluate(({ srcSelector, dstSelector }: any) => {
    const src = document.querySelector(srcSelector) as HTMLElement
    const dst = document.querySelector(dstSelector) as HTMLElement
    if (!src || !dst) throw new Error(`drag src/dst not found: ${!!src}/${!!dst}`)
    const dt = new DataTransfer()
    const fire = (target: HTMLElement, type: string) => {
      const r = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    fire(src, 'dragstart')
    const types = Array.from(dt.types)
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
    return { types }
  }, { srcSelector, dstSelector })
}

// The rendered cell order as `md:<text>` / `fig` tokens.
async function cellOrder(page: any): Promise<string[]> {
  return await page.evaluate(() => {
    const body = document.querySelector('[data-testid="report-body"]')!
    const out: string[] = []
    for (const el of Array.from(body.querySelectorAll('[data-report-cell="1"]'))) {
      if (el.querySelector('[data-testid^="report-figcell-"]')) out.push('fig')
      else {
        const r = el.querySelector('[data-testid^="report-cell-rendered-"]')
        out.push(`md:${(r?.textContent || '').trim().slice(0, 10)}`)
      }
    }
    return out
  })
}

test('1) build [md, figure, md]', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await page.getByTestId('report-new').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  // md "Alpha"
  await page.getByTestId('report-add-text').click()
  const ta = () => page.locator('[data-testid^="report-cell-textarea-"]').first()
  await page.locator('[data-testid^="report-cell-rendered-"]').first().dblclick()
  await ta().fill('Alpha')
  await ta().press('Control+Enter')

  // figure from the signal window pill
  const sigWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  await sigWin.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-fig-src', '1'))
  const res = await dragAndDrop(page, '[data-fig-src="1"]', '[data-testid="report-body"]')
  expect(res.types).toContain(FIG_MIME)
  await expect(page.locator('[data-testid^="report-figcell-"]').first())
    .toBeVisible({ timeout: 15_000 })

  // md "Beta"
  await page.getByTestId('report-add-text').click()
  const beta = page.locator('[data-testid^="report-cell-rendered-"]').last()
  await beta.dblclick()
  await page.locator('[data-testid^="report-cell-textarea-"]').last().fill('Beta')
  await page.locator('[data-testid^="report-cell-textarea-"]').last().press('Control+Enter')

  expect(await cellOrder(page)).toEqual(['md:Alpha', 'fig', 'md:Beta'])
  await page.screenshot({ path: join(SHOTS, '01-before-reorder.png') })
  ctx.assertNoJsErrors()
})

test('2) drag the figure cell by its handle above the first md cell', async () => {
  const { page } = ctx
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  // The ⠿ handle mounts on hover — synthetic mouseover (the report_edit2
  // pattern; a real hover lands on the iframe and never reaches the parent).
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const handle = page.locator('[data-testid^="report-figcell-drag-"]').first()
  await expect(handle).toBeVisible()
  await handle.evaluate((el: HTMLElement) => el.setAttribute('data-reorder-src', '1'))
  // Drop target = the FIRST markdown cell's root (its dragProps onDrop).
  await page.locator('[data-testid^="report-cell-rendered-"]').first()
    .evaluate((el: HTMLElement) => {
      // Walk up to the ReportCell root (the element with the drop handlers).
      const root = el.closest('[data-testid^="report-cell-"]') as HTMLElement
      root.setAttribute('data-reorder-dst', '1')
    })

  await dragAndDrop(page, '[data-reorder-src="1"]', '[data-reorder-dst="1"]')

  await expect.poll(() => cellOrder(page), { timeout: 10_000 })
    .toEqual(['fig', 'md:Alpha', 'md:Beta'])
  // The figure iframe is still live after the move (not blanked).
  await expect(
    page.locator('[data-testid^="report-figcell-"]').first()
      .locator('iframe[data-testid^="figure-"]'),
  ).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '02-after-reorder.png') })
  ctx.assertNoJsErrors()
})

test('3) drag a markdown cell below the figure (mixed-type reorder both ways)', async () => {
  const { page } = ctx
  // [fig, Alpha, Beta] → drag Alpha to the end (drop on report body past cells
  // is covered by unit tests; here drop onto Beta then re-check order).
  const alpha = page.locator('[data-testid^="report-cell-rendered-"]')
    .filter({ hasText: 'Alpha' }).first()
  await alpha.dispatchEvent('mouseover', { bubbles: true })
  const alphaRoot = page.locator('[data-testid^="report-cell-"]')
    .filter({ has: alpha }).first()
  const alphaHandle = alphaRoot.locator('[data-testid^="report-cell-drag-"]')
  await expect(alphaHandle).toBeVisible()
  await alphaHandle.evaluate((el: HTMLElement) => el.setAttribute('data-reorder-src2', '1'))
  // Target: the figure cell root — drop Alpha before the figure.
  await page.locator('[data-testid^="report-figcell-"]').first()
    .evaluate((el: HTMLElement) => el.setAttribute('data-reorder-dst2', '1'))

  await dragAndDrop(page, '[data-reorder-src2="1"]', '[data-reorder-dst2="1"]')
  await expect.poll(() => cellOrder(page), { timeout: 10_000 })
    .toEqual(['md:Alpha', 'fig', 'md:Beta'])
  await page.screenshot({ path: join(SHOTS, '03-md-before-fig.png') })
  ctx.assertNoJsErrors()
})
