/**
 * add_slide_menu.spec.ts — the "+ Add slide ▾" starter menu must never be
 * CLIPPED by the report body's scroll container.
 *
 * The bug: the menu was `position:absolute; bottom:100%` inside
 * `ReportSidebar`'s `styles.body`, which is `overflowY:'auto'`. On a short deck
 * the add row sits near the TOP of that scroller, so a menu opening UPWARD ran
 * past the scroller's top edge and was sliced in half ("Add text slide" cut
 * through the middle). It is now a `position:fixed` popover anchored to the
 * button rect (the CaretBox idiom), so no overflow ancestor can clip it.
 *
 * Asserted at THREE deck heights — empty (button near the top of the body),
 * mid, and a long deck scrolled to the bottom (button near the window bottom) —
 * because the fix must also FLIP: drop down when there is no room above, drop up
 * when there is no room below, and always stay inside the viewport.
 *
 * The checks are pixel-level, not selector-level: `toBeVisible()` passes on a
 * half-clipped menu. We compare the menu's client rect to the viewport AND
 * hit-test its four corners with elementFromPoint — a clipped corner is painted
 * by the ancestor, so the hit-test returns something outside the menu.
 *
 * Test 4b covers the per-slide Background picker, the second popover in that
 * same scroller. (The third, the split cell's layout picker, is driven by
 * presentation_editing.spec.ts.)
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdirSync } from 'fs'
const { launchApp, backendErrorLines } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'add_slide_menu_shots')

interface Rect { top: number; bottom: number; left: number; right: number; height: number }

interface MenuGeom {
  menu: Rect
  /** The FIRST starter row — the one the bug sliced through the middle. */
  firstItem: Rect
  vw: number
  vh: number
  /** Do all four corners of the menu land on the menu itself? */
  cornersHit: boolean[]
  /** Same for the first item — a half-clipped row fails here. */
  itemCornersHit: boolean[]
  /** Rects of its scroll/clip ancestors, reported on failure for diagnosis. */
  clips: Rect[]
}

/** Measure the open menu. The load-bearing check is the HIT-TEST, not the rect:
 *  a clipped region is painted by the ancestor, so elementFromPoint there
 *  returns something outside the menu. (`toBeVisible()` is happy either way,
 *  which is exactly why this bug shipped.) */
async function menuGeom(
  page: any,
  menuTestid = 'report-add-slide-menu',
  itemTestid = 'add-slide-text',
): Promise<MenuGeom> {
  return await page.evaluate(({ menuTestid, itemTestid }: any) => {
    const rectOf = (e: Element) => {
      const r = e.getBoundingClientRect()
      return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, height: r.height }
    }
    const el = document.querySelector(`[data-testid="${menuTestid}"]`) as HTMLElement
    if (!el) throw new Error(`${menuTestid} not in the DOM`)
    const item = document.querySelector(`[data-testid="${itemTestid}"]`) as HTMLElement
    if (!item) throw new Error(`${itemTestid} not in the DOM`)
    // Probe a couple of px inside each corner so a 1px border / AA doesn't matter.
    const hitCorners = (target: HTMLElement) => {
      const r = target.getBoundingClientRect()
      const P = 3
      const pts: [number, number][] = [
        [r.left + P, r.top + P], [r.right - P, r.top + P],
        [r.left + P, r.bottom - P], [r.right - P, r.bottom - P],
      ]
      return pts.map(([x, y]) => {
        const hit = document.elementFromPoint(x, y)
        return !!hit && (el === hit || el.contains(hit))
      })
    }
    const clips: any[] = []
    for (let p = el.parentElement; p; p = p.parentElement) {
      const cs = getComputedStyle(p)
      if (/(auto|scroll|hidden)/.test(cs.overflowY + cs.overflowX)) clips.push(rectOf(p))
    }
    return {
      menu: rectOf(el), firstItem: rectOf(item),
      vw: window.innerWidth, vh: window.innerHeight,
      cornersHit: hitCorners(el), itemCornersHit: hitCorners(item),
      clips,
    }
  }, { menuTestid, itemTestid })
}

/** The whole contract: inside the viewport, and every corner of the panel AND of
 *  its first row actually painted by the menu (i.e. clipped by nothing). */
function expectUnclipped(g: MenuGeom, where: string) {
  const ctx = ` [menu ${JSON.stringify(g.menu)} clips ${JSON.stringify(g.clips)}]`
  expect(g.menu.height, `${where}: menu has no height${ctx}`).toBeGreaterThan(40)
  expect(g.menu.top, `${where}: menu top above the viewport${ctx}`).toBeGreaterThanOrEqual(0)
  expect(g.menu.bottom, `${where}: menu bottom below the viewport${ctx}`).toBeLessThanOrEqual(g.vh)
  expect(g.menu.left, `${where}: menu left of the viewport${ctx}`).toBeGreaterThanOrEqual(0)
  expect(g.menu.right, `${where}: menu right of the viewport${ctx}`).toBeLessThanOrEqual(g.vw)
  expect(g.cornersHit, `${where}: a menu corner is not painted by the menu (clipped)${ctx}`)
    .toEqual([true, true, true, true])
  expect(g.itemCornersHit,
    `${where}: the first starter row is clipped (this IS the reported bug)${ctx}`)
    .toEqual([true, true, true, true])
}

async function openMenu(page: any) {
  await page.getByTestId('report-add-slide').click()
  await expect(page.getByTestId('report-add-slide-menu')).toBeVisible()
  // All four starters are on the menu.
  for (const t of ['add-slide-text', 'add-slide-split', 'add-slide-title', 'add-slide-figure']) {
    await expect(page.getByTestId(t), `${t} missing`).toBeVisible()
  }
}

async function closeMenu(page: any) {
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('report-add-slide-menu')).toHaveCount(0, { timeout: 4_000 })
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

// ── 1) EMPTY deck — the button sits near the TOP of the scrolling body ────────
// This is the reported case: a menu opening upward had nowhere to go and the
// scroller sliced its first item in half.

test('1) empty deck — the add-slide menu is fully visible, not clipped', async () => {
  const { page } = ctx
  await openMenu(page)
  await page.screenshot({ path: join(SHOTS, '01-empty-deck-menu.png') })
  expectUnclipped(await menuGeom(page), 'empty deck')
  // Room below (the deck is empty) → it opens DOWN, matching the ▾ caret,
  // instead of upward over the sidebar's title bar.
  expect(await page.getByTestId('report-add-slide-menu').getAttribute('data-placement'))
    .toBe('down')
  await closeMenu(page)
  ctx.assertNoJsErrors()
})

// ── 2) A few slides — the button has moved down the body ─────────────────────

test('2) short deck — still fully visible', async () => {
  const { page } = ctx
  for (let i = 0; i < 2; i++) {
    await page.getByTestId('report-add-slide').click()
    await page.getByTestId('add-slide-text').click()
    await page.waitForTimeout(300)
  }
  await expect(page.getByTestId('report-slide-1')).toBeVisible({ timeout: 8_000 })
  await openMenu(page)
  await page.screenshot({ path: join(SHOTS, '02-short-deck-menu.png') })
  expectUnclipped(await menuGeom(page), 'short deck')
  await closeMenu(page)
  ctx.assertNoJsErrors()
})

// ── 3) Long deck scrolled to the BOTTOM — the button is near the window edge ──
// Here there is no room BELOW, so the menu must flip up and still fit.

test('3) long deck scrolled to the bottom — flips up, stays inside the viewport', async () => {
  const { page } = ctx
  for (let i = 0; i < 5; i++) {
    await page.getByTestId('report-add-slide').click()
    await page.getByTestId('add-slide-title').click()
    await page.waitForTimeout(250)
  }
  // Park the scroller at the very bottom so the add row hugs the window edge.
  await page.evaluate(() => {
    const b = document.querySelector('[data-testid="report-body"]') as HTMLElement
    if (b) b.scrollTop = b.scrollHeight
  })
  await page.waitForTimeout(200)
  const btnTop = await page.getByTestId('report-add-slide').evaluate(
    (el: HTMLElement) => el.getBoundingClientRect().top)
  const vh = await page.evaluate(() => window.innerHeight)
  // Sanity: the button really is in the lower part of the window, otherwise this
  // test is not exercising the flip at all.
  expect(btnTop, 'add-slide button is not near the window bottom').toBeGreaterThan(vh * 0.5)

  await openMenu(page)
  await page.screenshot({ path: join(SHOTS, '03-bottom-menu.png') })
  const g = await menuGeom(page)
  expectUnclipped(g, 'bottom of the window')
  // With no room below, it opened UPWARD (above the button).
  expect(await page.getByTestId('report-add-slide-menu').getAttribute('data-placement'))
    .toBe('up')
  expect(g.menu.bottom, 'menu did not flip up at the window bottom')
    .toBeLessThanOrEqual(btnTop + 1)
  await closeMenu(page)
  ctx.assertNoJsErrors()
})

// ── 4) A SHORT window — neither side has room for the whole menu ──────────────
// The degenerate case: it must take the roomier side, scroll internally, and
// still not put a single pixel outside the window.

test('4) short window — the menu is capped to the viewport, never overflows it', async () => {
  const { page, app } = ctx
  const before = await app.evaluate(({ BrowserWindow }: any) =>
    BrowserWindow.getAllWindows()[0].getSize())
  await app.evaluate(({ BrowserWindow }: any) =>
    BrowserWindow.getAllWindows()[0].setSize(1280, 420))
  await page.waitForTimeout(500)
  await page.evaluate(() => {
    const b = document.querySelector('[data-testid="report-body"]') as HTMLElement
    if (b) b.scrollTop = b.scrollHeight
  })
  await page.waitForTimeout(200)

  await openMenu(page)
  await page.screenshot({ path: join(SHOTS, '04-short-window-menu.png') })
  const g = await menuGeom(page)
  expectUnclipped(g, 'short window')
  await closeMenu(page)

  await app.evaluate(({ BrowserWindow }: any, size: number[]) =>
    BrowserWindow.getAllWindows()[0].setSize(size[0], size[1]), before)
  await page.waitForTimeout(500)
  ctx.assertNoJsErrors()
})

// ── 4b) The per-slide Background picker lives in the same scroller ────────────
// Same class of bug, same container — it opens DOWNWARD, so a slide header near
// the bottom of the body used to have its options cut off.

test('4b) the per-slide Background menu is not clipped either', async () => {
  const { page } = ctx
  // Put a slide header as low in the body as it will go.
  await page.evaluate(() => {
    const b = document.querySelector('[data-testid="report-body"]') as HTMLElement
    if (b) b.scrollTop = b.scrollHeight
  })
  await page.waitForTimeout(200)
  // The lowest header that is actually on screen.
  const n = await page.evaluate(() => {
    const heads = Array.from(document.querySelectorAll('[data-testid^="report-slide-header-"]'))
    let best: { n: number; top: number } | null = null
    for (const h of heads) {
      const r = h.getBoundingClientRect()
      const id = Number((h.getAttribute('data-testid') || '').split('-').pop())
      if (r.top > 0 && r.bottom < window.innerHeight && (!best || r.top > best.top)) {
        best = { n: id, top: r.top }
      }
    }
    return best?.n ?? 0
  })
  await page.getByTestId(`report-slide-bg-toggle-${n}`).click()
  await expect(page.getByTestId(`report-slide-bg-menu-${n}`)).toBeVisible()
  await page.screenshot({ path: join(SHOTS, '05-bg-menu.png') })
  expectUnclipped(
    await menuGeom(page, `report-slide-bg-menu-${n}`, `report-slide-bg-${n}-default`),
    'slide background picker')
  // …and it still applies a choice.
  await page.getByTestId(`report-slide-bg-${n}-accent`).click()
  await expect(page.getByTestId(`report-slide-bg-menu-${n}`)).toHaveCount(0, { timeout: 4_000 })
  await expect(page.getByTestId(`report-slide-${n}`))
    .toHaveAttribute('data-slide-style', 'accent', { timeout: 8_000 })
  ctx.assertNoJsErrors()
})

// ── 5) The menu still WORKS (it is a popover, not a decoration) ───────────────

test('5) picking a starter from the popover still adds the slide', async () => {
  const { page } = ctx
  const before = await page.evaluate(() =>
    (window as any)._spyde_test_report?.()?.cells?.length ?? 0)
  await openMenu(page)
  await page.getByTestId('add-slide-split').click()
  await expect(page.getByTestId('report-add-slide-menu')).toHaveCount(0, { timeout: 4_000 })
  await expect.poll(async () => await page.evaluate(() =>
    (window as any)._spyde_test_report?.()?.cells?.length ?? 0),
  { timeout: 8_000, message: 'no slide added' }).toBe(before + 1)
  const doc = await page.evaluate(() => (window as any)._spyde_test_report?.())
  const last = doc.cells[doc.cells.length - 1]
  expect(last.cell_type).toBe('split')
  expect(last.slide_break).toBe(true)
  ctx.assertNoJsErrors()
})

// ── 6) Clicking outside dismisses it ─────────────────────────────────────────

test('6) an outside click closes the popover', async () => {
  const { page } = ctx
  await openMenu(page)
  // Press somewhere well outside the sidebar (the empty MDI area).
  await page.mouse.click(220, 400)
  await expect(page.getByTestId('report-add-slide-menu')).toHaveCount(0, { timeout: 4_000 })
  ctx.assertNoJsErrors()
})

test('7) no Python tracebacks in the backend log', async () => {
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[add_slide_menu] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
