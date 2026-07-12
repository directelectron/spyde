/**
 * breadcrumb_header.spec.ts — the window-header breadcrumb pill:
 * compact `S-name` / `N-name` pill, double-click-to-rename, draggable empty
 * titlebar space, and the minimized-window pill. Screenshots for the eyes check.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

// Full native HTML5 drag from `srcSelector` → `dstSelector`, entirely in-page so
// the constructed DataTransfer is shared across dragstart/dragover/drop (the way
// a real user drag is). Returns the MIME types the source stamped.
async function dragAndDrop(page: any, srcSelector: string, dstSelector: string) {
  return await page.evaluate(({ srcSelector, dstSelector }: any) => {
    const src = document.querySelector(srcSelector) as HTMLElement
    const dst = document.querySelector(dstSelector) as HTMLElement
    if (!src || !dst) throw new Error('drag src/dst not found')
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

const SHOTS = join(__dirname, '..', 'breadcrumb_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 60_000)  // navigator + signal
  await page.waitForTimeout(2500)
})
test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(120_000)

test('signal + navigator windows show compact S-/N- breadcrumb pills', async () => {
  const { page } = ctx
  await page.screenshot({ path: join(SHOTS, '01-breadcrumbs.png') })
  const pills = page.getByTestId('window-breadcrumb')
  const n = await pills.count()
  console.log('[breadcrumb] pill count =', n)
  expect(n).toBeGreaterThanOrEqual(2)
  // Each pill is a compact `S-name` / `N-name` (a kind prefix + editable name),
  // no trailing Root/nav segment.
  const names = page.getByTestId('breadcrumb-name')
  expect(await names.count()).toBeGreaterThanOrEqual(2)
  const texts = await page.getByTestId('window-breadcrumb').allInnerTexts()
  console.log('[breadcrumb] texts =', JSON.stringify(texts))
  // One window is a navigator (N-…) and one a signal (S-…); no "Navigator"/
  // "Signal" full words, and no trailing " | root"/" | base".
  expect(texts.some(t => /^N-/.test(t.replace(/\s/g, '')))).toBe(true)
  expect(texts.some(t => /^S-/.test(t.replace(/\s/g, '')))).toBe(true)
  expect(texts.every(t => !/\broot\b|\bbase\b/.test(t))).toBe(true)
  ctx.assertNoJsErrors()
})

test('double-click the Name renames the dataset on every window of the tree', async () => {
  const { page } = ctx
  // Double-click the first Name segment → inline input appears.
  const name = page.getByTestId('breadcrumb-name').first()
  await name.dblclick()
  const input = page.getByTestId('rename-input')
  await expect(input).toBeVisible()
  await input.fill('renamed_scan')
  await input.press('Enter')
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '02-renamed.png') })
  // Both the signal AND navigator windows share the root title → both show it.
  const texts = await page.getByTestId('breadcrumb-name').allInnerTexts()
  console.log('[breadcrumb] names after rename =', JSON.stringify(texts))
  expect(texts.filter(t => t === 'renamed_scan').length).toBeGreaterThanOrEqual(2)
  ctx.assertNoJsErrors()
})

test('minimized window renders as a breadcrumb pill and restores', async () => {
  const { page } = ctx
  const win = page.getByTestId('subwindow').first()
  await win.getByTestId('minimize-btn').click()
  await page.waitForTimeout(500)
  const bar = page.getByTestId('minimized-bar')
  await expect(bar).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '03-minimized-pill.png') })
  // The min-chip is a breadcrumb pill (has the name segment inside it).
  const minPill = bar.getByTestId('breadcrumb-name').first()
  await expect(minPill).toBeVisible()
  // Click restores.
  await bar.locator('[data-testid^="min-chip-"]').first().click()
  await page.waitForTimeout(400)
  ctx.assertNoJsErrors()
})

test('empty titlebar space (right of the pill) still drags the window', async () => {
  const { page } = ctx
  const win = page.getByTestId('subwindow').first()
  const bar = win.getByTestId('subwindow-titlebar')
  const pill = win.getByTestId('window-breadcrumb')
  const barBox = (await bar.boundingBox())!
  const pillBox = (await pill.boundingBox())!
  const before = (await win.boundingBox())!
  // Grab a point in the bare titlebar space: right of the pill, left of the
  // window buttons (~64px of controls on the right).
  const grabX = Math.min(pillBox.x + pillBox.width + 30, barBox.x + barBox.width - 80)
  const grabY = barBox.y + barBox.height / 2
  await page.mouse.move(grabX, grabY)
  await page.mouse.down()
  await page.mouse.move(grabX + 90, grabY + 60, { steps: 8 })
  await page.mouse.up()
  await page.waitForTimeout(300)
  const after = (await win.boundingBox())!
  console.log('[drag-space] before=', before.x, before.y, 'after=', after.x, after.y)
  // The window moved (the empty space is a drag handle, not swallowed).
  expect(Math.abs(after.x - before.x) + Math.abs(after.y - before.y)).toBeGreaterThan(30)
  ctx.assertNoJsErrors()
})

test('dragging the signal breadcrumb into the console binds + shows a pill', async () => {
  const { page } = ctx
  // The signal window's breadcrumb pill carries the signal-ref MIME. Signal
  // windows' pills start "S-"; navigators' start "N-".
  const sigWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  const pill = sigWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-drag-src', '1'))
  const input = page.getByTestId('console-input')
  await input.fill('')
  const res = await dragAndDrop(page, '[data-drag-src="1"]', '[data-testid="console-input"]')
  console.log('[drag] console drop types =', JSON.stringify(res.types))
  expect(res.types).toContain('application/x-spyde-signal-ref')
  // The variable name is inserted at the caret and PILL-IFIES in the overlay
  // (it matches a bound console var) — the old cosmetic dropped-pill is gone.
  await expect.poll(async () => (await input.inputValue()).trim(), { timeout: 5000 }).not.toBe('')
  await expect(page.locator('[data-testid^="console-pill-"]').first()).toBeVisible({ timeout: 5000 })
  await page.screenshot({ path: join(SHOTS, '04-console-drop-pill.png') })
  ctx.assertNoJsErrors()
})

test('dragging a Workflow node into the console binds it as a chip', async () => {
  const { page } = ctx
  // Focus the signal window (pill starts "S-") so the Plot Control dock shows
  // its Workflow tree.
  await page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first().click()
  await page.waitForTimeout(400)
  const node = page.getByTestId('tree-node-root').first()
  await expect(node).toBeVisible({ timeout: 5000 })
  await node.evaluate((el: HTMLElement) => el.setAttribute('data-wf-src', '1'))
  const res = await dragAndDrop(page, '[data-wf-src="1"]', '[data-testid="console-input"]')
  console.log('[drag] workflow drop types =', JSON.stringify(res.types))
  expect(res.types).toContain('application/x-spyde-workflow-node')
  // The workflow branch used to insert NO text (the bug). Now the assigned var
  // name lands (via console_node_bound) and is typed into the input, then pills
  // in the overlay. Also a console chip appears for the bound node.
  const wfInput = page.getByTestId('console-input')
  await expect.poll(async () => (await wfInput.inputValue()).trim(), { timeout: 5000 }).not.toBe('')
  await expect(page.locator('[data-testid^="console-pill-"]').first()).toBeVisible({ timeout: 5000 })
  await page.waitForTimeout(800)
  await page.screenshot({ path: join(SHOTS, '05-workflow-drop.png') })
  const chip = page.locator('[data-testid^="console-chip-"]')
  expect(await chip.count()).toBeGreaterThanOrEqual(1)
  ctx.assertNoJsErrors()
})
