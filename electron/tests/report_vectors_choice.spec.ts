/**
 * report_vectors_choice.spec.ts — dropping a vectors-carrying window into the
 * Report Builder defers the cell behind an embed-choice prompt (interactive
 * viewer vs static image); each pick creates the cell. The export-side honor
 * of the recorded choice is covered by test_report_vectors_embed.py.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_vectors_choice_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

// Full native HTML5 drag src→dst, entirely in-page so the constructed
// DataTransfer is shared across dragstart/dragover/drop (the proven pattern
// from report_sidebar.spec.ts / breadcrumb_header.spec.ts).
async function dragAndDrop(page: any, srcSelector: string, dstSelector: string) {
  await page.evaluate(({ srcSelector, dstSelector }: { srcSelector: string; dstSelector: string }) => {
    const src = document.querySelector(srcSelector) as HTMLElement | null
    const dst = document.querySelector(dstSelector) as HTMLElement | null
    if (!src || !dst) throw new Error('drag src/dst not found')
    const dt = new DataTransfer()
    const fire = (el: HTMLElement, type: string) => {
      const r = el.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true, dataTransfer: dt,
        clientX: r.x + r.width / 2, clientY: r.y + r.height / 2,
      })
      el.dispatchEvent(ev)
    }
    fire(src, 'dragstart')
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
  }, { srcSelector, dstSelector })
}

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_vectors')
  await waitForSubwindowCount(page, 4, 60_000)
  // The vectors attach at batch FINALIZE (not when the result window opens) —
  // the requires_vectors-gated toolbar button appearing is the attach signal.
  await expect(page.getByTestId('action-btn-Vector Virtual Imaging').first())
    .toBeAttached({ timeout: 60_000 })
  await page.waitForTimeout(500)
})

test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(180_000)

test('vectors window drop prompts; viewer and image picks each create a cell', async () => {
  const { page } = ctx

  // Open the report sidebar + a fresh document.
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await page.getByTestId('report-new').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  // The vectors SIGNAL window (its tree carries diffraction_vectors) is the
  // one with the Vector Virtual Imaging action.
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Vector Virtual Imaging') }).first()
  await vsig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-vx-src', '1'))

  // 1) Drop → the embed-choice prompt appears and NO cell is created yet.
  await dragAndDrop(page, '[data-vx-src="1"]', '[data-testid="report-body"]')
  await expect(page.getByTestId('report-vectors-choice')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId(/^report-figcell-c[0-9a-f]{8}$/)).toHaveCount(0)
  await page.screenshot({ path: join(SHOTS, '01-choice-prompt.png') })

  // 2) Pick the interactive viewer → the figure cell lands, prompt closes.
  await page.getByTestId('report-vectors-viewer').click()
  await expect(page.getByTestId('report-vectors-choice')).toHaveCount(0)
  await expect(page.getByTestId(/^report-figcell-c[0-9a-f]{8}$/)).toHaveCount(1, { timeout: 15_000 })

  // 3) Drop again, pick "Just the image" → a second cell.
  await dragAndDrop(page, '[data-vx-src="1"]', '[data-testid="report-body"]')
  await expect(page.getByTestId('report-vectors-choice')).toBeVisible({ timeout: 15_000 })
  await page.getByTestId('report-vectors-image').click()
  await expect(page.getByTestId(/^report-figcell-c[0-9a-f]{8}$/)).toHaveCount(2, { timeout: 15_000 })

  // 4) Drop once more and cancel → prompt closes, still two cells.
  await dragAndDrop(page, '[data-vx-src="1"]', '[data-testid="report-body"]')
  await expect(page.getByTestId('report-vectors-choice')).toBeVisible({ timeout: 15_000 })
  await page.getByTestId('report-vectors-cancel').click()
  await expect(page.getByTestId('report-vectors-choice')).toHaveCount(0)
  await expect(page.getByTestId(/^report-figcell-c[0-9a-f]{8}$/)).toHaveCount(2)

  await page.screenshot({ path: join(SHOTS, '02-both-cells.png') })
  ctx.assertNoJsErrors()
})
