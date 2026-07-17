/**
 * report_phase2_probe.spec.ts — Phase-2 RENDERER smoke/verify for the Report
 * Builder composition UI (edit toolbar, compose drop zones, MDI overlay drop).
 *
 * NOT the full Phase-2 test suite (report_compose.spec.ts / mdi_overlay.spec.ts
 * are the caller's to author) — this is the implementer's "look at the pixels"
 * check per CLAUDE.md: launch the real app, drive the new UI, screenshot each
 * stage, and confirm nothing crashes / no blank frames.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'report_phase2_shots')

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

// Native HTML5 drag src→dst sharing one DataTransfer (breadcrumb_header pattern).
async function dragAndDrop(page: any, srcSelector: string, dstSelector: string,
                           dstPoint?: { fx: number; fy: number }) {
  return await page.evaluate(({ srcSelector, dstSelector, dstPoint }: any) => {
    const src = document.querySelector(srcSelector) as HTMLElement
    const dst = document.querySelector(dstSelector) as HTMLElement
    if (!src || !dst) throw new Error('drag src/dst not found')
    const dt = new DataTransfer()
    const fire = (target: HTMLElement, type: string, atSrc = false) => {
      const r = target.getBoundingClientRect()
      const fx = atSrc ? 0.5 : (dstPoint?.fx ?? 0.5)
      const fy = atSrc ? 0.5 : (dstPoint?.fy ?? 0.5)
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width * fx, clientY: r.top + r.height * fy,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    fire(src, 'dragstart', true)
    const types = Array.from(dt.types)
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend', true)
    return { types }
  }, { srcSelector, dstSelector, dstPoint })
}

test('setup: open report + drop the signal → live figure cell', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await page.getByTestId('report-new').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  const sigWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
    .first()
  const pill = sigWin.getByTestId('window-breadcrumb')
  await pill.evaluate((el: HTMLElement) => el.setAttribute('data-p2-sig', '1'))
  await dragAndDrop(page, '[data-p2-sig="1"]', '[data-testid="report-body"]')

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '01-figure-cell.png') })
  ctx.assertNoJsErrors()
})

test('edit toolbar: opens, lists a layer + annotation palette', async () => {
  const { page } = ctx
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  const cellId = await figCell.evaluate((el) =>
    (el.getAttribute('data-testid') || '').replace('report-figcell-', ''))
  // Reveal the hover chrome. .hover() lands on the iframe (which swallows the
  // event), and React synthesizes onMouseEnter from a delegated `mouseover`, so
  // dispatch a bubbling mouseover on the cell wrapper (not a raw mouseenter,
  // which React ignores), then click the revealed Edit toggle.
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  await expect(page.getByTestId(`report-figcell-edit-toggle-${cellId}`)).toBeVisible()
  await page.getByTestId(`report-figcell-edit-toggle-${cellId}`).click()
  const editPanel = page.getByTestId(`figcell-edit-${cellId}`)
  await expect(editPanel).toBeVisible()
  // Slim-bar redesign: a SINGLE-panel figure renders no chips and auto-targets
  // its only panel — the layer row + add palette are present directly. The
  // panel id comes from the authoritative report doc (the old figcell-panel-
  // dock block is gone).
  await expect(editPanel.locator('[data-testid^="figcell-layer-cmap-"]').first()).toBeVisible()
  const panelId = await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    const cell = d?.cells?.find((c: any) => c.id === cid)
    return cell?.figure?.panels?.[0]?.id ?? null
  }, cellId)
  expect(panelId, 'no panel id in the report doc').toBeTruthy()
  await expect(page.getByTestId(`figcell-add-text-${panelId}`)).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '02-edit-toolbar.png') })
  ctx.assertNoJsErrors()

  // Add a Text annotation → the figure rebuilds; the annotation lands in the
  // spec (annotation rows are popover-only now — poll the report doc).
  await page.getByTestId(`figcell-add-text-${panelId}`).click()
  await expect.poll(async () => await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    const cell = d?.cells?.find((c: any) => c.id === cid)
    return (cell?.figure?.panels?.[0]?.annotations ?? []).length
  }, cellId), { timeout: 10_000, message: '+ Text did not append a panel annotation' })
    .toBeGreaterThan(0)
  await page.waitForTimeout(1500)   // let the rebuilt figure repaint
  await page.screenshot({ path: join(SHOTS, '03-annotation-added.png') })
  ctx.assertNoJsErrors()
})

test('compose: edge drop tiles a second panel onto the cell', async () => {
  const { page } = ctx
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  const oldFigId = await figCell.locator('iframe[data-testid^="figure-"]').first()
    .getAttribute('data-testid')

  const navWin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
    .first()
  const navPill = navWin.getByTestId('window-breadcrumb')
  await navPill.evaluate((el: HTMLElement) => el.setAttribute('data-p2-nav', '1'))

  // The compose shield mounts only after dragKind='window' triggers a re-render,
  // which happens AFTER the dragstart event's React tick. So split the drag:
  // fire dragstart + a dragover over the report body (promotes dragKind='window'),
  // yield to React so the shield mounts over the iframe, THEN dragover+drop onto
  // the shield's right edge. The shared DataTransfer is stashed on window across
  // the evaluates.
  await page.evaluate(() => {
    const src = document.querySelector('[data-p2-nav="1"]') as HTMLElement
    const body = document.querySelector('[data-testid="report-body"]') as HTMLElement
    const dt = new DataTransfer()
    ;(window as any).__p2dt = dt
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
    fire(body, 'dragover')   // promote dragKind='window' → shield mounts
  })
  await page.waitForTimeout(300)   // let the shield mount

  await page.evaluate(() => {
    const dt = (window as any).__p2dt as DataTransfer
    const shield = document.querySelector('[data-testid^="figcell-compose-shield-"]') as HTMLElement
    const src = document.querySelector('[data-p2-nav="1"]') as HTMLElement
    if (!shield) throw new Error('compose shield not mounted')
    const r = shield.getBoundingClientRect()
    const fire = (target: HTMLElement, type: string, fx = 0.9, fy = 0.5) => {
      const rr = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: rr.left + rr.width * fx, clientY: rr.top + rr.height * fy,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    void r
    fire(shield, 'dragenter'); fire(shield, 'dragover'); fire(shield, 'drop')
    fire(src, 'dragend', 0.5, 0.5)
  })

  await expect.poll(async () => {
    return await figCell.locator('iframe[data-testid^="figure-"]').first()
      .getAttribute('data-testid').catch(() => null)
  }, { timeout: 15_000, message: 'tile compose did not rebuild the cell figure' })
    .not.toBe(oldFigId)

  await page.waitForTimeout(2000)
  await page.screenshot({ path: join(SHOTS, '04-tiled.png') })
  ctx.assertNoJsErrors()
})
