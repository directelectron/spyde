/**
 * grid_present.spec.ts — a multi-panel (subplot-grid) figure cell must render
 * EVERY panel's data when PRESENTED, not just the last one.
 *
 * Reported: building a 3-panel grid with the ＋ picker and then entering Present
 * mode shows two panels as empty black boxes (axes + scale bar drawn, no image),
 * with only the last panel carrying pixels — plus editing grips/outlines visible
 * on the presented slide.
 *
 * The sidebar cell and the presented slide are DIFFERENT render paths (the cell
 * hosts the live figure; a slide re-mounts it in the full-screen stage), so a
 * grid that looks right in the editor can still present broken. This spec
 * measures the pixels PER PANEL on the presented slide.
 *
 * Screenshots to grid_present_shots/ — each Read by the author.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'grid_present_shots')

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

async function docCell(page: any, cellId: string) {
  return await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    return d?.cells?.find((c: any) => c.id === cid) ?? null
  }, cellId)
}

/**
 * Sample the presented figure iframe's canvas and report, for each of `cols`
 * vertical bands, how many pixels are non-background. An EMPTY panel reads ~0.
 */
async function panelPixelCounts(page: any, cols: number): Promise<number[]> {
  // The figure iframe is a separate frame (file://), so `contentDocument` is
  // null from the top document — it has to be reached through page.frames().
  //
  // CRITICAL: scope to the PRESENTED slide's own frame. Simply taking the first
  // frame that owns a canvas picks up an MDI window's figure instead, and then
  // this function happily reports "both panels filled" about a completely
  // different plot — which it did, giving two different tests byte-identical
  // counts. Resolve the active slide's iframe src and match the frame by URL.
  const src: string | null = await page.evaluate(() => {
    const slide = document.querySelector('[data-testid="present-slide"][data-active="1"]')
    const iframe = slide?.querySelector('iframe') as HTMLIFrameElement | null
    return iframe?.src ?? null
  })
  if (!src) return []
  const target = page.frames().filter((f: any) => f.url() === src)
  if (!target.length) return []
  for (const frame of target) {
    try {
      const out = await frame.evaluate((nCols: number) => {
        const canvases = Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]
        if (!canvases.length) return null
        const cv = canvases.sort((a, b) => (b.width * b.height) - (a.width * a.height))[0]
        const c2 = cv.getContext('2d')
        if (!c2) return null
        const { width, height } = cv
        const bandW = Math.floor(width / nCols)
        const res: number[] = []
        for (let c = 0; c < nCols; c++) {
          const img = c2.getImageData(c * bandW, 0, bandW, height).data
          let n = 0
          for (let i = 0; i < img.length; i += 4) {
            if (img[i] > 60 || img[i + 1] > 60 || img[i + 2] > 60) n++
          }
          res.push(n)
        }
        return res
      }, cols)
      if (out && out.length) return out
    } catch { /* cross-origin / detached frame */ }
  }
  return []
}

test('build a 2-panel grid with ＋, then present it', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()

  // Embed window A as a figure cell.
  const sigA = sigWindows(page).nth(0)
  await sigA.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-gp', '1'))
  await page.evaluate(() => {
    const src = document.querySelector('[data-gp="1"]') as HTMLElement
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

  // ＋ → pick the second signal window → 1×2 grid.
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  await page.getByTestId(`cell-add-figure-${cellId}`).click()
  const menu = page.getByTestId(`add-figure-menu-${cellId}`)
  await expect(menu).toBeVisible({ timeout: 5_000 })
  await menu.locator('[data-testid^="add-figure-win-"]').nth(1).click()

  await expect.poll(async () => (await docCell(page, cellId))?.figure?.panels?.length ?? 0, {
    timeout: 20_000, message: '＋ did not build a 2-panel grid',
  }).toBe(2)
  await page.waitForTimeout(4000)
  await page.screenshot({ path: join(SHOTS, '01-grid-in-sidebar.png') })

  // THE SUSPECT: leave the figure EDITOR open. The reported screenshot shows
  // per-panel drag grips and selection outlines ON the presented slide, which
  // are edit-mode widgets — and edit mode makes the BACKEND rebuild the figure
  // with draggable annotation widgets. If that rebuild is what loses the image
  // layers, a deck presented with the editor still open shows empty panels.
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  await page.getByTestId(`report-figcell-edit-toggle-${cellId}`).click()
  await expect(page.getByTestId(`figcell-edit-${cellId}`)).toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(4000)
  await page.screenshot({ path: join(SHOTS, '01b-grid-edit-mode.png') })

  // Present it.
  await page.getByTestId('report-present').click()
  await expect(page.locator('[data-testid="present-slide"][data-active="1"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(6000)
  await page.screenshot({ path: join(SHOTS, '02-grid-presented.png') })

  const counts = await panelPixelCounts(page, 2)
  console.log('[grid-present] per-panel non-background pixel counts =', JSON.stringify(counts))

  // Editing affordances must not leak onto a presented slide. The grips are
  // anyplotlib widgets living INSIDE the figure frame, so count them there.
  let widgets = -1
  for (const frame of page.frames()) {
    try {
      const n = await frame.evaluate(() =>
        document.querySelectorAll('[class*="drag"],[class*="handle"],[class*="widget"]').length)
      if (n >= 0 && frame !== page.mainFrame()) { widgets = Math.max(widgets, n) }
    } catch { /* detached */ }
  }
  console.log('[grid-present] editing widgets inside the presented figure =', widgets)

  // The editor must have been closed by entering Present mode.
  const stillEditing = await page.getByTestId(`figcell-edit-${cellId}`).count()
  console.log('[grid-present] figure editor still open behind the deck =', stillEditing)

  expect(counts.length, 'could not sample the presented figure canvas').toBe(2)
  for (const [i, n] of counts.entries()) {
    expect(n, `presented panel ${i} is EMPTY (no image pixels)`).toBeGreaterThan(500)
  }
  expect(stillEditing, 'presenting must close the figure editor (its widgets draw on the slide)').toBe(0)

  // Leave Present mode — the full-screen deck covers the sidebar, so a following
  // test in this serial file cannot reach any report control while it is up.
  await page.keyboard.press('Escape')
  await expect(page.locator('[data-testid="present-slide"][data-active="1"]'))
    .toBeHidden({ timeout: 10_000 })
})

/**
 * THE REPORTED CASE: on SPED-Ag (a LAZY 4-D scan) a navigator+DP grid presents
 * with the NAVIGATOR panel filled and the DIFFRACTION panel EMPTY — axes and
 * scale bar drawn, no pixels.
 *
 * The earlier repro used eager `si_grains` and two DP panels, and passed. Two
 * things differ here and both are candidates: the source is LAZY (its frame may
 * not be resident when the panel is snapshotted) and the panels are a
 * navigator + a signal rather than two of a kind.
 */
test('LAZY dataset: navigator + DP grid presents with BOTH panels filled', async () => {
  const { page } = ctx
  // Defensive: never start with a deck up (see the Escape at the end of the
  // previous test) — the overlay would swallow every sidebar interaction.
  if (await page.locator('[data-testid="present-slide"][data-active="1"]').count()) {
    await page.keyboard.press('Escape')
    await page.waitForTimeout(500)
  }
  await backendAction(page, 'load_test_data_lazy_chunked')
  await waitForSubwindowCount(page, 6, 120_000)
  await page.waitForTimeout(4000)

  if (!(await page.getByTestId('report-sidebar').count())) {
    await page.getByTestId('toggle-report').click()
    await expect(page.getByTestId('report-sidebar')).toBeVisible()
  }
  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()
  await page.waitForTimeout(800)

  // The lazy dataset's NAVIGATOR window is the newest N- window.
  const navWins = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
  const sigWins = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
  const nNav = await navWins.count(), nSig = await sigWins.count()
  console.log('[lazy-grid] nav windows =', nNav, 'sig windows =', nSig)

  await navWins.nth(nNav - 1).getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-lazy-nav', '1'))
  await page.evaluate(() => {
    const s = document.querySelector('[data-lazy-nav="1"]') as HTMLElement
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
  await expect(figCell).toBeVisible({ timeout: 20_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 25_000 })
  await page.waitForTimeout(3000)
  const cellId = (await figCell.getAttribute('data-testid'))!.replace('report-figcell-', '')

  // ＋ → tile the lazy dataset's DIFFRACTION (S-) window in beside it.
  //
  // NB a real `.hover()` does NOT work here: it targets the cell's centre,
  // which is the figure IFRAME, and the out-of-process iframe swallows the
  // mouseover so the cell root's onMouseEnter never fires and the chrome never
  // appears. (That is a real usability wart — the ＋/Copy/Delete chrome is
  // unreachable while the pointer is over the plot itself.) Dispatch on the
  // cell root instead.
  await figCell.scrollIntoViewIfNeeded()
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const addBtn = page.getByTestId(`cell-add-figure-${cellId}`)
  await expect(addBtn).toBeVisible({ timeout: 10_000 })
  // A lazy figure keeps re-rendering as its frames swap in, which MOVES the
  // chrome — Playwright's stability check then times out on an element that is
  // perfectly visible. Let the box settle (same rect twice) before clicking.
  let lastBox = ''
  await expect.poll(async () => {
    const b = await addBtn.boundingBox()
    const key = b ? `${Math.round(b.x)},${Math.round(b.y)}` : 'none'
    const stable = key !== 'none' && key === lastBox
    lastBox = key
    return stable
  }, { timeout: 30_000, message: 'the ＋ button never stopped moving' }).toBe(true)
  await addBtn.click()
  const menu = page.getByTestId(`add-figure-menu-${cellId}`)
  await expect(menu).toBeVisible({ timeout: 5_000 })
  const winBtns = menu.locator('[data-testid^="add-figure-win-"]')
  await winBtns.nth(await winBtns.count() - 1).click()   // newest = the lazy S- window

  await expect.poll(async () => (await docCell(page, cellId))?.figure?.panels?.length ?? 0, {
    timeout: 25_000, message: '＋ did not build a 2-panel grid on the lazy dataset',
  }).toBe(2)
  await page.waitForTimeout(6000)
  await page.screenshot({ path: join(SHOTS, '03-lazy-grid-sidebar.png') })

  await page.getByTestId('report-present').click()
  await expect(page.locator('[data-testid="present-slide"][data-active="1"]')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(8000)
  await page.screenshot({ path: join(SHOTS, '04-lazy-grid-presented.png') })

  const counts = await panelPixelCounts(page, 2)
  console.log('[lazy-grid] per-panel non-background pixel counts =', JSON.stringify(counts))

  expect(counts.length, 'could not sample the presented figure canvas').toBe(2)
  for (const [i, n] of counts.entries()) {
    expect(n, `LAZY: presented panel ${i} is EMPTY (no image pixels)`).toBeGreaterThan(500)
  }
})
