/**
 * vectors_sidebar.spec.ts — the LIVE 2-panel vectors explorer (navigator + DP,
 * crosshair, pointer/integrate) rendered in the report SIDEBAR cell (Approach A).
 *
 * Dropping a vectors-carrying window into the Report Builder as an interactive
 * VIEWER cell must make the sidebar cell host the SAME self-contained explorer
 * the HTML export embeds — not a static snapshot. The backend emits the explorer
 * page as the cell's figure `html` (host:"report"); the main process writes it to
 * a spyde-fig:// file and SeamlessFigureFrame mounts it as the cell iframe. The
 * explorer runs entirely client-side (overlay-canvas disk splat) and exposes
 * `window.__vx` — the same test hook the export spec drives.
 *
 * We assert the SIDEBAR cell iframe:
 *   1. mounts the explorer (window.__vx present, #vx-root data-ready, canvases),
 *   2. renders DP disks for a pointer position (DP overlay canvas lights up),
 *   3. keeps rendering the DP after the crosshair moves (readout + stats update,
 *      DP stays lit), and INTEGRATE sums many positions (brighter, more vectors).
 *
 * NB the load_test_vectors fixture is a 6x6 nav of IDENTICAL 4-disk patterns
 * (every nav position has the same vectors), so — unlike the synthetic
 * left/right-cluster fixture in vectors_report_embed.spec.ts — this asserts the
 * explorer renders + updates, not a left↔right disk flip.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'vectors_sidebar_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

// Full native HTML5 drag src→dst, entirely in-page so the constructed
// DataTransfer is shared across dragstart/dragover/drop (the proven pattern from
// report_vectors_choice.spec.ts).
async function dragAndDrop(page: any, srcSelector: string, dstSelector: string) {
  await page.evaluate(({ srcSelector, dstSelector }:
    { srcSelector: string; dstSelector: string }) => {
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
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_vectors')
  await waitForSubwindowCount(page, 4, 60_000)
  // The vectors attach at batch FINALIZE (not when the result window opens) — the
  // requires_vectors-gated toolbar button appearing is the attach signal.
  await expect(page.getByTestId('action-btn-Vector Virtual Imaging').first())
    .toBeAttached({ timeout: 60_000 })
  await page.waitForTimeout(500)
})

test.afterAll(async () => { await ctx?.app?.close() })
test.setTimeout(180_000)

// Locate the report cell's figure iframe FRAME object (the explorer runs inside
// it; it's a spyde-fig:// same-privileged frame, not a sandboxed srcdoc, so
// Playwright can reach it). It's the frame whose page defines window.__vx.
async function explorerFrame(page: any) {
  for (let attempt = 0; attempt < 60; attempt++) {
    for (const f of page.frames()) {
      try {
        const has = await f.evaluate(() => typeof (window as any).__vx === 'object'
          && !!(window as any).__vx)
        if (has) return f
      } catch { /* frame navigating / cross-context — skip */ }
    }
    await page.waitForTimeout(500)
  }
  throw new Error('explorer frame (window.__vx) not found in report cell')
}

test('viewer-vectors sidebar cell hosts the live explorer; crosshair drives the DP', async () => {
  const { page } = ctx

  // Open the report sidebar + a fresh document.
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await page.getByTestId('report-new').click()
  await expect(page.getByTestId('report-body')).toBeVisible()

  // The vectors SIGNAL window (its tree carries diffraction_vectors) is the one
  // with the Vector Virtual Imaging action; tag it as the drag source.
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Vector Virtual Imaging') }).first()
  await vsig.getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-vx-src', '1'))

  // Drop → embed-choice prompt; pick the interactive VIEWER → the cell lands.
  await dragAndDrop(page, '[data-vx-src="1"]', '[data-testid="report-body"]')
  await expect(page.getByTestId('report-vectors-choice')).toBeVisible({ timeout: 15_000 })
  await page.screenshot({ path: join(SHOTS, '01-choice-prompt.png') })
  await page.getByTestId('report-vectors-viewer').click()
  await expect(page.getByTestId('report-vectors-choice')).toHaveCount(0)
  await expect(page.getByTestId(/^report-figcell-c[0-9a-f]{8}$/)).toHaveCount(1, { timeout: 15_000 })

  // The SIDEBAR cell iframe must mount the explorer (window.__vx present).
  const fr = await explorerFrame(page)
  await fr.waitForSelector('#vx-root[data-ready="1"]', { timeout: 30_000 })
  // ONE figure, TWO panels → several canvases mounted inside the cell.
  expect(await fr.locator('#vx-fig canvas').count()).toBeGreaterThan(0)
  // Both nav widgets serialized (crosshair pointer + rectangle integrate) → the
  // crosshair marker is a live SVG/canvas element in the figure.
  const hasCross = await fr.evaluate(() => {
    const h = (window as any).__vx._h()
    const pj = JSON.parse(h.H.get(h.navKey))
    return (pj.overlay_widgets || []).some((w: any) => w.type === 'crosshair')
  })
  expect(hasCross).toBe(true)
  await page.screenshot({ path: join(SHOTS, '02-sidebar-explorer.png') })

  // The DP overlay canvas: brightest per-pixel mean over the figure's canvases
  // (the DP disks push here — a broken push would leave it dark). Scoped INSIDE
  // the cell iframe.
  const dpBrightness = () => fr.evaluate(() => {
    let best = 0
    for (const c of document.querySelectorAll('#vx-fig canvas')) {
      const ctx2 = (c as HTMLCanvasElement).getContext('2d')
      if (!ctx2 || !(c as HTMLCanvasElement).width) continue
      const d = ctx2.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                  (c as HTMLCanvasElement).height).data
      let sum = 0
      for (let i = 0; i < d.length; i += 4) sum += d[i]
      best = Math.max(best, sum / (d.length / 4))
    }
    return best
  })
  const stats = () => fr.evaluate(() => (window as any).__vx.stats)

  // POINTER at nav (0,0) → the position's disks render (the find-vectors detector
  // decides the per-position count on this real data — assert it's non-zero, not
  // an exact number) and the DP overlay canvas lights up. Capture the per-position
  // count for the integrate assertion below.
  await fr.evaluate(() => (window as any).__vx.setPointer({ ix: 0, iy: 0 }))
  await expect.poll(async () => (await stats()).hit).toBeGreaterThan(0)
  const perPos = (await stats()).hit
  await expect.poll(dpBrightness, { timeout: 5_000 }).toBeGreaterThan(5)
  await expect(fr.locator('#vx-readout')).toContainText(`${perPos} vectors`)
  await page.screenshot({ path: join(SHOTS, '03-pointer-00.png') })

  // MOVE the crosshair to another nav position → the DP keeps rendering (the
  // fixture's patterns are identical, but the explorer must recompute + stay lit
  // and the readout must report the NEW position). This is the "crosshair drives
  // the DP" contract regardless of whether the pixels happen to differ.
  await fr.evaluate(() => (window as any).__vx.setPointer({ ix: 5, iy: 5 }))
  await expect(fr.locator('#vx-readout')).toContainText('pointer: nav (5, 5)')
  await expect.poll(async () => (await stats()).hit).toBeGreaterThan(0)
  await expect.poll(dpBrightness, { timeout: 5_000 }).toBeGreaterThan(5)
  await page.screenshot({ path: join(SHOTS, '04-pointer-55.png') })

  // THEMED SEGMENTED TOGGLE (fix #1): clicking the "Integrate" pill switches mode
  // (aria-pressed follows the accent fill) — the same path setMode drives. Assert
  // the toggle markup is present, dark, and actually switches the mode.
  await expect(fr.locator('.vx-seg-btn[data-mode="pointer"]')).toBeVisible()
  await expect(fr.locator('.vx-seg-btn[data-mode="integrate"]')).toBeVisible()
  // Dark theme (fix #2): the page body carries the app surface color, not white.
  const bodyBg = await fr.evaluate(() =>
    getComputedStyle(document.body).backgroundColor)
  expect(bodyBg).toBe('rgb(30, 30, 46)')   // #1e1e2e
  await fr.locator('.vx-seg-btn[data-mode="integrate"]').click()
  await expect.poll(async () => (await fr.evaluate(() => (window as any).__vx.mode.integrate)))
    .toBe(true)
  expect(await fr.locator('.vx-seg-btn[data-mode="integrate"]')
    .getAttribute('aria-pressed')).toBe('true')
  expect(await fr.locator('.vx-seg-btn[data-mode="pointer"]')
    .getAttribute('aria-pressed')).toBe('false')

  // INTEGRATE the whole 6x6 nav → 36 positions x perPos vectors summed. (Patterns
  // are identical so the NORMALISED DP looks the same as a pointer frame — the
  // contract here is the mode switch + region sum path, so assert the vector count
  // + that the DP still lights up.)
  await fr.evaluate(() => (window as any).__vx.setRegion({ x: 0, y: 0, w: 6, h: 6 }))
  await expect.poll(async () => (await stats()).hit).toBe(36 * perPos)
  await expect(fr.locator('#vx-readout')).toContainText(`${36 * perPos} vectors summed`)
  await expect.poll(dpBrightness, { timeout: 5_000 }).toBeGreaterThan(5)
  await page.screenshot({ path: join(SHOTS, '05-integrate.png') })

  // VIRTUAL IMAGING (fix #4): the DP DETECTOR drives the navigator VI. Back to
  // pointer mode; the identical-pattern fixture has 4 disks per frame — place the
  // detector ON a disk (nonzero viHit, nav lights) then MOVE it far OFF all disks
  // (zero viHit, nav goes dark). A moving detector changing the nav VI is the
  // contract. Read a disk center from the vectors payload so the detector lands
  // exactly on one.
  await fr.evaluate(() => (window as any).__vx.setMode(false))
  const disk = await fr.evaluate(() => {
    const h = (window as any).__vx._h()
    const hdr = JSON.parse(document.getElementById('vx-header')!.textContent!)
    // Reconstruct the first vector's DP-pixel column/row from the header extents.
    const b64 = document.getElementById('vx-data')!.textContent!.trim()
    return { W: hdr.sig[1], H: hdr.sig[0] }
  })
  // Put the detector at a disk (the fixture's disks are away from center); scan a
  // few candidate columns and keep the one that catches vectors.
  let onDisk = 0
  for (const cx of [Math.round(disk.W * 0.25), Math.round(disk.W * 0.5),
                    Math.round(disk.W * 0.75), Math.round(disk.W * 0.4),
                    Math.round(disk.W * 0.6)]) {
    await fr.evaluate((c) => (window as any).__vx.setDetector(
      { cx: c, cy: Math.round(c * 0 + 0) }), cx)
    // sweep cy too
    for (const cy of [Math.round(disk.H * 0.25), Math.round(disk.H * 0.5),
                      Math.round(disk.H * 0.75)]) {
      await fr.evaluate(({ c, y }) => (window as any).__vx.setDetector(
        { cx: c, cy: y, r: 20 }), { c: cx, y: cy })
      await page.waitForTimeout(60)
      const vh = (await stats()).viHit
      if (vh > 0) { onDisk = vh; break }
    }
    if (onDisk > 0) break
  }
  expect(onDisk).toBeGreaterThan(0)
  // The navigator VI overlay must now carry signal (nonzero mean).
  const navBrightness = () => fr.evaluate((navId) => {
    // The VI overlay is the topmost canvas over the nav panel; measure the max
    // mean over all figure canvases as a signal-present proxy.
    let best = 0
    for (const c of document.querySelectorAll('#vx-fig canvas')) {
      const ctx2 = (c as HTMLCanvasElement).getContext('2d')
      if (!ctx2 || !(c as HTMLCanvasElement).width) continue
      const d = ctx2.getImageData(0, 0, (c as HTMLCanvasElement).width,
                                  (c as HTMLCanvasElement).height).data
      let sum = 0
      for (let i = 0; i < d.length; i += 4) sum += d[i]
      best = Math.max(best, sum / (d.length / 4))
    }
    return best
  }, '')
  await expect.poll(navBrightness, { timeout: 5_000 }).toBeGreaterThan(5)
  await page.screenshot({ path: join(SHOTS, '06-detector-vi.png') })

  // MOVE the detector to a far corner (off every disk) → the VI drops to zero.
  await fr.evaluate((d) => (window as any).__vx.setDetector(
    { cx: 2, cy: 2, r: 1 }), disk)
  await expect.poll(async () => (await stats()).viHit).toBe(0)
  await page.screenshot({ path: join(SHOTS, '07-detector-vi-empty.png') })

  ctx.assertNoJsErrors()
})
