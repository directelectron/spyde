/**
 * compose_real_drag.spec.ts — combine two figures into a SUBPLOT GRID by a REAL
 * mouse drag.
 *
 * Why this spec exists: `report_compose.spec.ts` and the phase-2 probe drive
 * compose by dispatching synthetic `DragEvent`s with a hand-built DataTransfer.
 * That exercises the HANDLERS but bypasses the browser's real drag machinery —
 * so a break in what actually mounts the drop target (the `dragKind === 'window'`
 * shield over the figure iframe) passes every existing test while the feature is
 * dead in the user's hands. This spec uses `page.mouse` so Chromium runs its own
 * DnD.
 *
 * The gesture under test is the EDGE drop: drag a window pill over an existing
 * figure cell, hover its LEFT edge → "Tile ←" zone → drop → the cell's figure
 * becomes a 1×2 anyplotlib grid (two panels).
 *
 * Screenshots to compose_drag_shots/ — each Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'compose_drag_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 4, 120_000)
  await page.waitForTimeout(3000)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

function sigWindows(page: any) {
  return page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
}

/** The report doc cell (by id) straight off the renderer state. */
async function docCell(page: any, cellId: string) {
  return await page.evaluate((id: string) => {
    const d = (window as any)._spyde_test_report?.()
    return d?.cells?.find((c: any) => c.id === id) ?? null
  }, cellId)
}

test('setup: embed one signal window as a figure cell', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()

  // Embed via the synthetic-DataTransfer path (known-good) — the REAL drag is
  // what the next test isolates.
  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-drag-sigA', '1'))
  await page.evaluate(() => {
    const src = document.querySelector('[data-drag-sigA="1"]') as HTMLElement
    const dst = document.querySelector('[data-testid="report-body"]') as HTMLElement
    if (!src || !dst) throw new Error('drag src/report-body not found')
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
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
  })

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '01-one-figure-cell.png') })
})

test('REAL drag: pill → figure cell left edge → 2-panel grid', async () => {
  const { page } = ctx

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  const cellId = (await figCell.getAttribute('data-testid'))!.replace('report-figcell-', '')
  const before = await docCell(page, cellId)
  console.log('[compose-drag] panels BEFORE =', before?.figure?.panels?.length ?? 0,
              'layout =', JSON.stringify(before?.figure?.layout ?? null))

  // Drag source: the SECOND signal window's breadcrumb pill.
  const pill = sigWindows(page).nth(1).getByTestId('window-breadcrumb')
  await expect(pill).toBeVisible()
  const src = (await pill.boundingBox())!
  const dst = (await figCell.boundingBox())!

  // A real HTML5 drag: press on the pill, move in steps (Chromium needs several
  // moves past the drag threshold to promote it to a drag), hover the LEFT edge
  // of the figure cell, then release.
  const leftEdgeX = dst.x + dst.width * 0.12
  const midY = dst.y + dst.height * 0.5

  await page.mouse.move(src.x + src.width / 2, src.y + src.height / 2)
  await page.mouse.down()
  await page.mouse.move(src.x + src.width / 2 + 12, src.y + src.height / 2 + 12, { steps: 6 })
  await page.mouse.move(dst.x + dst.width * 0.5, midY, { steps: 20 })
  await page.waitForTimeout(300)

  // MID-DRAG diagnostics: did the shield mount, and did a zone light up?
  const midDrag = await page.evaluate((id: string) => ({
    shield: !!document.querySelector(`[data-testid="figcell-compose-shield-${id}"]`),
    zones: !!document.querySelector('[data-testid="figcell-zones"]'),
  }), cellId)
  console.log('[compose-drag] mid-drag over CENTER =', JSON.stringify(midDrag))

  await page.mouse.move(leftEdgeX, midY, { steps: 12 })
  await page.waitForTimeout(300)
  const onEdge = await page.evaluate(() => {
    const z = document.querySelector('[data-testid="figcell-zones"]')
    const hot = document.querySelector('[data-testid="figcell-zone-left"]')
    return { zones: !!z, leftZone: !!hot }
  })
  console.log('[compose-drag] mid-drag over LEFT EDGE =', JSON.stringify(onEdge))
  await page.screenshot({ path: join(SHOTS, '02-mid-drag-left-zone.png') })

  await page.mouse.up()
  await page.waitForTimeout(4000)
  await page.screenshot({ path: join(SHOTS, '03-after-drop.png') })

  const after = await docCell(page, cellId)
  console.log('[compose-drag] panels AFTER =', after?.figure?.panels?.length ?? 0,
              'layout =', JSON.stringify(after?.figure?.layout ?? null))

  expect(midDrag.shield, 'the compose drop shield never mounted — dragKind never became "window"').toBe(true)
  expect(onEdge.leftZone, 'the "Tile ←" zone never appeared on the left edge').toBe(true)
  expect(after?.figure?.panels?.length ?? 0,
    'the drop did not produce a 2-panel grid figure').toBe(2)
})

// The same gesture inside a PRESENTATION (slide-grouped) document. The user's
// deck is a presentation, and slides wrap every cell in a slide GROUP that
// carries its own reorder drag handlers — the suspect for a compose drop that
// works in a report and not in a deck.
test('REAL drag inside a PRESENTATION deck', async () => {
  const { page } = ctx
  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()
  await page.waitForTimeout(1000)

  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-drag-pres', '1'))
  await page.evaluate(() => {
    const src = document.querySelector('[data-drag-pres="1"]') as HTMLElement
    const dst = document.querySelector('[data-testid="report-body"]') as HTMLElement
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
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
  })

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '04-presentation-one-figure.png') })

  const cellId = (await figCell.getAttribute('data-testid'))!.replace('report-figcell-', '')

  const pill = sigWindows(page).nth(1).getByTestId('window-breadcrumb')
  const src = (await pill.boundingBox())!
  const dst = (await figCell.boundingBox())!
  const leftEdgeX = dst.x + dst.width * 0.12
  const midY = dst.y + dst.height * 0.5

  await page.mouse.move(src.x + src.width / 2, src.y + src.height / 2)
  await page.mouse.down()
  await page.mouse.move(src.x + src.width / 2 + 12, src.y + src.height / 2 + 12, { steps: 6 })
  await page.mouse.move(dst.x + dst.width * 0.5, midY, { steps: 20 })
  await page.waitForTimeout(300)
  const midDrag = await page.evaluate((id: string) => ({
    shield: !!document.querySelector(`[data-testid="figcell-compose-shield-${id}"]`),
    zones: !!document.querySelector('[data-testid="figcell-zones"]'),
  }), cellId)
  console.log('[compose-drag/pres] mid-drag over CENTER =', JSON.stringify(midDrag))

  await page.mouse.move(leftEdgeX, midY, { steps: 12 })
  await page.waitForTimeout(300)
  const onEdge = await page.evaluate(() => ({
    zones: !!document.querySelector('[data-testid="figcell-zones"]'),
    leftZone: !!document.querySelector('[data-testid="figcell-zone-left"]'),
  }))
  console.log('[compose-drag/pres] mid-drag over LEFT EDGE =', JSON.stringify(onEdge))
  await page.screenshot({ path: join(SHOTS, '05-pres-mid-drag.png') })

  await page.mouse.up()
  await page.waitForTimeout(4000)
  await page.screenshot({ path: join(SHOTS, '06-pres-after-drop.png') })

  const after = await docCell(page, cellId)
  console.log('[compose-drag/pres] panels AFTER =', after?.figure?.panels?.length ?? 0,
              'layout =', JSON.stringify(after?.figure?.layout ?? null))

  expect(midDrag.shield, 'PRESENTATION: compose shield never mounted').toBe(true)
  expect(onEdge.leftZone, 'PRESENTATION: "Tile ←" zone never appeared').toBe(true)
  expect(after?.figure?.panels?.length ?? 0,
    'PRESENTATION: the drop did not produce a 2-panel grid figure').toBe(2)
})

/**
 * THE REGRESSION THIS SPEC EXISTS FOR: a SPLIT block (text beside a figure —
 * the layout most presentation slides use) had a replace-only drop on its
 * figure side. No zones, no tiling: dropping a second window swapped the figure
 * instead of combining, so the subplot grid was unreachable from a deck.
 */
test('SPLIT block figure side: edge drop tiles into a grid', async () => {
  const { page } = ctx
  // Runnable standalone (`-g SPLIT`), so open the sidebar if the setup test
  // that normally does it was filtered out.
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
    await expect(page.getByTestId('report-sidebar')).toBeVisible()
  }
  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()
  await page.waitForTimeout(800)

  // A split slide, then fill its figure side from window A.
  await backendAction(page, 'report_add_split_cell', { slide_break: true })
  const splitCell = page.locator('[data-testid^="report-split-"]').first()
  await expect(splitCell).toBeVisible({ timeout: 10_000 })
  const splitId = (await page.locator('[data-testid^="report-split-dropzone-"]')
    .first().getAttribute('data-testid'))!.replace('report-split-dropzone-', '')

  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-drag-split', '1'))
  await page.evaluate((id: string) => {
    const src = document.querySelector('[data-drag-split="1"]') as HTMLElement
    const dst = document.querySelector(`[data-testid="report-split-dropzone-${id}"]`) as HTMLElement
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
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
  }, splitId)

  const splitFrame = page.getByTestId('report-sidebar')
    .locator('iframe[data-testid^="figure-"]').first()
  await expect(splitFrame).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(2500)
  await page.screenshot({ path: join(SHOTS, '07-split-filled.png') })

  const before = await docCell(page, splitId)
  console.log('[compose-drag/split] panels BEFORE =', before?.figure?.panels?.length ?? 0)

  // REAL drag of window B onto the split figure's RIGHT edge.
  const dst = (await splitFrame.boundingBox())!
  const pill = sigWindows(page).nth(1).getByTestId('window-breadcrumb')
  const src = (await pill.boundingBox())!
  const rightEdgeX = dst.x + dst.width * 0.88
  const midY = dst.y + dst.height * 0.5

  await page.mouse.move(src.x + src.width / 2, src.y + src.height / 2)
  await page.mouse.down()
  await page.mouse.move(src.x + src.width / 2 + 12, src.y + src.height / 2 + 12, { steps: 6 })
  await page.mouse.move(rightEdgeX, midY, { steps: 20 })
  await page.waitForTimeout(400)
  const onEdge = await page.evaluate((id: string) => ({
    shield: !!document.querySelector(`[data-testid="report-split-shield-${id}"]`),
    zones: !!document.querySelector('[data-testid="figcell-zones"]'),
    rightZone: !!document.querySelector('[data-testid="figcell-zone-right"]'),
  }), splitId)
  console.log('[compose-drag/split] mid-drag over RIGHT EDGE =', JSON.stringify(onEdge))
  await page.screenshot({ path: join(SHOTS, '08-split-mid-drag.png') })

  await page.mouse.up()
  await page.waitForTimeout(5000)
  await page.screenshot({ path: join(SHOTS, '09-split-after-drop.png') })

  const after = await docCell(page, splitId)
  console.log('[compose-drag/split] panels AFTER =', after?.figure?.panels?.length ?? 0,
              'layout =', JSON.stringify(after?.figure?.layout ?? null))

  expect(onEdge.zones, 'SPLIT: no compose zones on the split figure side').toBe(true)
  expect(onEdge.rightZone, 'SPLIT: "Tile →" zone never appeared').toBe(true)
  expect(after?.figure?.panels?.length ?? 0,
    'SPLIT: the edge drop did not tile into a 2-panel grid').toBe(2)
})

/**
 * The drag TRACE (dragDiag.ts) must record every stage of a healthy drag — it
 * is the instrument used to diagnose a drag that fails on a real machine but
 * not under Playwright, so it has to be known-good before its silence means
 * anything.
 */
test('the drag trace records all six stages of a working drag', async () => {
  const { page } = ctx
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
    await expect(page.getByTestId('report-sidebar')).toBeVisible()
  }
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()
  await page.waitForTimeout(800)
  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-drag-trace', '1'))
  await page.evaluate(() => {
    const s = document.querySelector('[data-drag-trace="1"]') as HTMLElement
    const d = document.querySelector('[data-testid="report-body"]') as HTMLElement
    const dt = new DataTransfer()
    const fire = (t: HTMLElement, type: string) => {
      const r = t.getBoundingClientRect()
      const ev = new DragEvent(type, { bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      t.dispatchEvent(ev)
    }
    fire(s, 'dragstart')
    fire(d, 'dragenter'); fire(d, 'dragover'); fire(d, 'drop'); fire(s, 'dragend')
  })

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(2500)

  const pill = sigWindows(page).nth(1).getByTestId('window-breadcrumb')
  const src = (await pill.boundingBox())!
  const dst = (await figCell.boundingBox())!
  await page.mouse.move(src.x + src.width / 2, src.y + src.height / 2)
  await page.mouse.down()
  await page.mouse.move(src.x + src.width / 2 + 12, src.y + src.height / 2 + 12, { steps: 6 })
  await page.mouse.move(dst.x + dst.width * 0.12, dst.y + dst.height * 0.5, { steps: 18 })
  await page.waitForTimeout(300)
  await page.mouse.up()
  await page.waitForTimeout(1500)

  const stages = await page.evaluate(() =>
    ((window as any).__spydeDragEntries?.() ?? []).map((e: any) => e.stage))
  console.log('[drag-trace] stages =', JSON.stringify(stages, null, 1))

  for (const want of ['1.dragstart/pill', '3.shield/mounted', '4.shield/dragover',
                      '5.shield/drop', '6.sendAction/repfig_compose']) {
    expect(stages.some((s: string) => s === want),
      `the trace never recorded "${want}" — the diagnostic itself is broken`).toBe(true)
  }
})

/**
 * THE FIX FOR THE REPORTED FAILURE: the figure iframe must become
 * pointer-events:none while a window pill is in flight.
 *
 * Why a CSS assertion rather than a behavioural one: the bug is that a REAL
 * drag is routed by the browser process via compositor surface hit-testing,
 * which answers with the out-of-process IFRAME's surface and delivers
 * dragover/drop to that frame instead of to the shield above it. Playwright's
 * synthesized drag goes through the renderer's input path and never reproduces
 * that — every compose e2e in this file passed while the feature was dead in
 * the user's hands. So the only honest thing to pin here is the contract that
 * fixes it: during a window drag the iframe is out of hit-testing.
 */
test('the figure iframe becomes pointer-events:none during a window drag', async () => {
  const { page } = ctx
  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })

  const hostPointerEvents = async () => await page.evaluate(() => {
    const frame = document.querySelector(
      '[data-testid^="report-figcell-"] iframe[data-testid^="figure-"]')
    const host = frame?.parentElement
    return host ? getComputedStyle(host).pointerEvents : null
  })

  expect(await hostPointerEvents(), 'the figure should be interactive when NOT dragging')
    .not.toBe('none')

  // Start a drag and hold it.
  const pill = sigWindows(page).nth(1).getByTestId('window-breadcrumb')
  const src = (await pill.boundingBox())!
  const dst = (await figCell.boundingBox())!
  await page.mouse.move(src.x + src.width / 2, src.y + src.height / 2)
  await page.mouse.down()
  await page.mouse.move(src.x + src.width / 2 + 12, src.y + src.height / 2 + 12, { steps: 6 })
  await page.mouse.move(dst.x + dst.width * 0.5, dst.y + dst.height * 0.5, { steps: 14 })
  await page.waitForTimeout(300)

  const during = await hostPointerEvents()
  console.log('[oopif-fix] frame host pointer-events DURING drag =', during)
  expect(during, 'the iframe must leave hit-testing during a drag, or a real OS drag '
    + 'is routed to it instead of to the compose shield').toBe('none')

  await page.mouse.up()
  await page.waitForTimeout(1500)
  const after = await hostPointerEvents()
  console.log('[oopif-fix] frame host pointer-events AFTER drag =', after)
  expect(after, 'the figure must be interactive again once the drag ends').not.toBe('none')
})

/**
 * THE REPORTED SYMPTOM: the drop zones light up correctly, you release, and
 * nothing happens.
 *
 * That is exactly what a drop whose `getData()` returns EMPTY looks like —
 * `types` still advertises the MIME (so the zones appear on dragover), but the
 * payload is unreadable at drop time, the handler resolves a null source window
 * and silently returns. A synthesized Playwright drag never reproduces it
 * because its DataTransfer stays inside the renderer; a real OS-level drag can.
 *
 * This drives that state directly: a DataTransfer that ADVERTISES the MIME in
 * `types` but hands back "" from getData(). It must still compose, via the
 * in-process payload stash (dnd.ts).
 */
test('drop with an unreadable payload still composes (types set, getData empty)', async () => {
  const { page } = ctx
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
    await expect(page.getByTestId('report-sidebar')).toBeVisible()
  }
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()
  await page.waitForTimeout(800)

  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-drag-empty', '1'))
  await page.evaluate(() => {
    const src = document.querySelector('[data-drag-empty="1"]') as HTMLElement
    const dst = document.querySelector('[data-testid="report-body"]') as HTMLElement
    const dt = new DataTransfer()
    const fire = (t: HTMLElement, type: string) => {
      const r = t.getBoundingClientRect()
      const ev = new DragEvent(type, { bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      t.dispatchEvent(ev)
    }
    fire(src, 'dragstart')
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
  })

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(2500)
  const cellId = (await figCell.getAttribute('data-testid'))!.replace('report-figcell-', '')

  // Now the crippled drag: real dragstart on the pill (so the source stashes its
  // payload), then a hand-built DataTransfer over the figure whose getData()
  // returns "" while types still lists the MIME.
  // 1) A real dragstart on the pill — populates the in-process stash AND flips
  //    dragKind to 'window', which is what MOUNTS the compose shield. Must be
  //    its own turn: the shield does not exist until React re-renders.
  const started = await page.evaluate(() => {
    const pill = document.querySelectorAll('[data-testid="window-breadcrumb"]')[2] as HTMLElement
    if (!pill) return false
    const dt = new DataTransfer()
    const ev = new DragEvent('dragstart', { bubbles: true, cancelable: true })
    Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
    pill.dispatchEvent(ev)
    return true
  })
  expect(started, 'no third window pill to drag').toBe(true)
  await page.waitForTimeout(500)

  // 2) Now the shield is mounted. Drop on its LEFT edge with a DataTransfer
  //    that LIES: types advertises the MIME, getData() hands back "".
  const composed = await page.evaluate((id: string) => {
    const shield = document.querySelector(
      `[data-testid="figcell-compose-shield-${id}"]`) as HTMLElement | null
    if (!shield) return { ok: false, why: 'compose shield never mounted' }
    const crippled = {
      types: ['application/x-spyde-figure', 'application/x-spyde-window'],
      getData: () => '',
      dropEffect: 'copy', effectAllowed: 'copy',
      setData: () => {},
    }
    const r = shield.getBoundingClientRect()
    const x = r.left + r.width * 0.1, y = r.top + r.height * 0.5   // left edge → tile-left
    for (const type of ['dragenter', 'dragover', 'drop']) {
      const ev = new DragEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y })
      Object.defineProperty(ev, 'dataTransfer', { value: crippled, configurable: true })
      shield.dispatchEvent(ev)
    }
    return { ok: true, why: '' }
  }, cellId)
  console.log('[empty-payload] dispatch =', JSON.stringify(composed))
  expect(composed.ok, composed.why).toBe(true)

  await expect.poll(async () => (await docCell(page, cellId))?.figure?.panels?.length ?? 0, {
    timeout: 20_000,
    message: 'a drop whose getData() was empty did NOT compose — the stash fallback is not working',
  }).toBe(2)
  const after = await docCell(page, cellId)
  console.log('[empty-payload] panels AFTER =', after?.figure?.panels?.length,
              'layout =', JSON.stringify(after?.figure?.layout))
  await page.screenshot({ path: join(SHOTS, '12-empty-payload-composed.png') })
})

/**
 * The CLICK path: ＋ on the cell chrome → window picker → tiles it in. This is
 * the target you cannot miss — no 28%-wide edge strip to aim at — so it is the
 * one that has to work every time.
 */
test('＋ Add figure: click path builds the grid without any drag', async () => {
  const { page } = ctx
  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
    await expect(page.getByTestId('report-sidebar')).toBeVisible()
  }
  await backendAction(page, 'report_new', {})
  await expect(page.getByTestId('report-body')).toBeVisible()
  await page.waitForTimeout(800)

  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-drag-click', '1'))
  await page.evaluate(() => {
    const src = document.querySelector('[data-drag-click="1"]') as HTMLElement
    const dst = document.querySelector('[data-testid="report-body"]') as HTMLElement
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
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
  })

  const figCell = page.locator('[data-testid^="report-figcell-"]').first()
  await expect(figCell).toBeVisible({ timeout: 15_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(2500)
  const cellId = (await figCell.getAttribute('data-testid'))!.replace('report-figcell-', '')

  // Hover to reveal the chrome, then ＋ → the picker.
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const addBtn = page.getByTestId(`cell-add-figure-${cellId}`)
  await expect(addBtn).toBeVisible()
  await addBtn.click()
  const menu = page.getByTestId(`add-figure-menu-${cellId}`)
  await expect(menu).toBeVisible({ timeout: 5_000 })
  await page.screenshot({ path: join(SHOTS, '10-add-figure-menu.png') })

  // Pick the SECOND signal window from the list.
  const winButtons = menu.locator('[data-testid^="add-figure-win-"]')
  const nWin = await winButtons.count()
  console.log('[add-figure] windows offered =', nWin)
  expect(nWin, 'the picker listed no windows').toBeGreaterThanOrEqual(2)
  await winButtons.nth(1).click()

  await page.waitForTimeout(5000)
  await page.screenshot({ path: join(SHOTS, '11-add-figure-result.png') })
  const after = await docCell(page, cellId)
  console.log('[add-figure] panels AFTER =', after?.figure?.panels?.length ?? 0,
              'layout =', JSON.stringify(after?.figure?.layout ?? null))
  expect(after?.figure?.panels?.length ?? 0,
    '＋ Add figure did not tile a second panel in').toBe(2)
})
