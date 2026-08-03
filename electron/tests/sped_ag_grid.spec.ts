/**
 * sped_ag_grid.spec.ts — the REPORTED empty-panel bug, on the dataset it was
 * reported on.
 *
 * Reported: on SPED-Ag, composing a navigator + diffraction grid and presenting
 * it shows the navigator panel filled and the DIFFRACTION panel EMPTY — a box
 * with a scale bar and no image.
 *
 * It does NOT reproduce on the bundled si_grains (eager) or lazy_chunked
 * datasets — both present with every panel filled. So this drives the real
 * thing: pyxem's sped_ag, a 208x64 scan of 112x112 patterns, lazy behind dask.
 *
 * The measurement is per-panel: sample the presented figure's canvas in
 * vertical bands and count non-background pixels. An empty panel reads ~0 while
 * its neighbour reads tens of thousands, which is exactly the reported shape.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'sped_ag_shots')

let ctx: Awaited<ReturnType<typeof launchApp>>

test.describe.configure({ mode: 'serial' })
test.setTimeout(600_000)

test.beforeAll(async () => {
  // SPYDE_GPU_IMAGE=0 forces anyplotlib's Canvas2D reference renderer.
  // Load-bearing for the MEASUREMENT, not the behaviour: a full-size SPED-Ag
  // frame renders on the GPU canvas, where getContext('2d') returns null and
  // canvas readback yields nothing at all — the per-panel probe reported []
  // for a figure that visibly drew both panels. The CPU path answers the only
  // question this spec asks ("does the panel carry image content?") and is
  // readable.
  ctx = await launchApp({
    dask: true,
    env: { SPYDE_LOG_LEVEL: 'INFO' },
  })
  const { page } = ctx
  await page.waitForTimeout(1500)
  // The real scan — downloads on first run, then pooch-cached.
  await backendAction(page, 'load_test_data_sped_ag')
  await waitForSubwindowCount(page, 2, 300_000)
  // A lazy 208x64 scan: give the navigator time to fill and the DP to paint.
  await page.waitForTimeout(15000)
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally { await ctx?.app?.close() }
})

async function docCell(page: any, cellId: string) {
  return await page.evaluate((cid: string) => {
    const d = (window as any)._spyde_test_report?.()
    return d?.cells?.find((c: any) => c.id === cid) ?? null
  }, cellId)
}

/** Per-panel non-background pixel counts, scoped to the ACTIVE slide's frame. */
async function panelPixelCounts(page: any, cols: number): Promise<number[]> {
  const src: string | null = await page.evaluate(() => {
    const slide = document.querySelector('[data-testid="present-slide"][data-active="1"]')
    const iframe = slide?.querySelector('iframe') as HTMLIFrameElement | null
    return iframe?.src ?? null
  })
  if (!src) return []
  for (const frame of page.frames().filter((f: any) => f.url() === src)) {
    try {
      const out = await frame.evaluate((nCols: number) => {
        // Score EVERY 2-D canvas and keep the one carrying the most drawn
        // pixels. Picking the LARGEST canvas is wrong: anyplotlib stacks a
        // transparent widget-overlay canvas at the same size over the plot, so
        // "largest" can land on a canvas that is empty by construction — which
        // reported [0, 0] for a figure that visibly draws both panels.
        const cvs = Array.from(document.querySelectorAll('canvas')) as HTMLCanvasElement[]
        let best: number[] | null = null
        let bestTotal = -1
        for (const cv of cvs) {
          if (!cv.width || !cv.height) continue
          const c2 = cv.getContext('2d')
          if (!c2) continue                      // WebGL/WebGPU surface
          const bandW = Math.floor(cv.width / nCols)
          if (bandW <= 0) continue
          const res: number[] = []
          let total = 0
          try {
            for (let c = 0; c < nCols; c++) {
              const img = c2.getImageData(c * bandW, 0, bandW, cv.height).data
              let n = 0
              for (let i = 0; i < img.length; i += 4) {
                if (img[i] > 60 || img[i + 1] > 60 || img[i + 2] > 60) n++
              }
              res.push(n); total += n
            }
          } catch { continue }                   // tainted canvas
          if (total > bestTotal) { bestTotal = total; best = res }
        }
        return best
      }, cols)
      if (out && out.length) return out
    } catch { /* detached */ }
  }
  return []
}

test('SPED-Ag: navigator + DP grid, presented, both panels must draw', async () => {
  const { page } = ctx

  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await backendAction(page, 'report_new', { type: 'presentation' })
  await expect(page.getByTestId('report-body')).toBeVisible()

  // A TITLE SLIDE FIRST, so the grid lands on slide 2 — matching the reported
  // deck. This is load-bearing, not decoration: Present mode renders EVERY
  // slide but hides the inactive ones with `display:none`, so a figure on any
  // slide but the first mounts its iframe inside a zero-size subtree. A figure
  // that lays out at 0x0 and is only revealed later is a completely different
  // situation from one that mounts visible, and the first repro (grid on slide
  // 1) could not see it.
  await backendAction(page, 'report_add_cell', {
    cell_type: 'markdown', source: '# SpyDE\n\nCarter Francis', slide_kind: 'title',
  })
  await page.waitForTimeout(800)

  const navWins = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^N-/ }) })
  const sigWins = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-/ }) })
  console.log('[sped-ag] nav windows =', await navWins.count(),
              'sig windows =', await sigWins.count())

  // Embed the NAVIGATOR as the first panel.
  await navWins.first().getByTestId('window-breadcrumb')
    .evaluate((el: HTMLElement) => el.setAttribute('data-sa-nav', '1'))
  await page.evaluate(() => {
    const s = document.querySelector('[data-sa-nav="1"]') as HTMLElement
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
  await expect(figCell).toBeVisible({ timeout: 30_000 })
  await expect(figCell.locator('iframe[data-testid^="figure-"]')).toBeVisible({ timeout: 60_000 })
  await page.waitForTimeout(6000)
  const cellId = (await figCell.getAttribute('data-testid'))!.replace('report-figcell-', '')
  await page.screenshot({ path: join(SHOTS, '01-nav-embedded.png') })

  // ＋ → tile the DIFFRACTION window in beside it.
  await figCell.scrollIntoViewIfNeeded()
  await figCell.dispatchEvent('mouseover', { bubbles: true })
  const addBtn = page.getByTestId(`cell-add-figure-${cellId}`)
  await expect(addBtn).toBeVisible({ timeout: 15_000 })
  await addBtn.click()
  const menu = page.getByTestId(`add-figure-menu-${cellId}`)
  await expect(menu).toBeVisible({ timeout: 10_000 })
  // The S- (signal / diffraction) entry.
  const sigEntry = menu.locator('[data-testid^="add-figure-win-"]')
    .filter({ hasText: /sped|test_data/i }).last()
  await sigEntry.click()

  await expect.poll(async () => (await docCell(page, cellId))?.figure?.panels?.length ?? 0, {
    timeout: 60_000, message: 'the DP never tiled in',
  }).toBe(2)
  await page.waitForTimeout(8000)
  await page.screenshot({ path: join(SHOTS, '02-grid-in-sidebar.png') })

  // What the SPEC says about each panel — a panel with no layer, or a layer
  // with a degenerate clim, would explain an empty box while its axes and
  // scale bar still draw.
  const cell = await docCell(page, cellId)
  console.log('[sped-ag] panels =', JSON.stringify(
    (cell?.figure?.panels ?? []).map((p: any) => ({
      id: p.id, kind: p.kind, grid_pos: p.grid_pos,
      layers: (p.layers ?? []).map((l: any) => ({ id: l.id, cmap: l.cmap, clim: l.clim })),
      hasAxes: !!p.axes, units: p.axes?.units,
    })), null, 1))

  // MATCH THE REPORTER'S WINDOW SIZE before presenting.
  //
  // "If I press it it doesn't work but if you do it does" — the remaining
  // difference between that session and this one is the stage the figure has to
  // re-lay out onto. A figure slide now takes a 96rem column, so on a large
  // display the presented iframe is far wider than the sidebar cell it was
  // built in, and the figure has to relayout to a size this suite never
  // exercised at ~1400x811.
  await ctx.app.evaluate(async ({ BrowserWindow }: any) => {
    const w = BrowserWindow.getAllWindows()[0]
    if (w) { w.setSize(2000, 1250); w.center() }
  })
  await page.waitForTimeout(2500)

  // Present, then ADVANCE to slide 2 — the figure's iframe has been mounted
  // inside a display:none slide the whole time.
  await page.getByTestId('report-present').click()
  await expect(page.locator('[data-testid="present-slide"][data-active="1"]'))
    .toBeVisible({ timeout: 30_000 })
  await page.waitForTimeout(3000)
  await page.screenshot({ path: join(SHOTS, '03a-title-slide.png') })
  await page.keyboard.press('ArrowRight')
  await page.waitForTimeout(12000)
  await page.screenshot({ path: join(SHOTS, '03-presented.png') })

  // What the presented iframe was given to draw from. On a WORKING 2-panel
  // figure this is the baseline the reported failure has to be compared
  // against: if a broken deck shows fewer binary pixel states than panels,
  // that is the bug.
  // The ON-SCREEN readout (press D) — the whole point is that it needs no
  // DevTools, so it has to be verified as visible, not just as a function that
  // returns data.
  // Via the BUTTON, not the key: a keypress that silently does nothing is
  // indistinguishable from a stale build, and that ambiguity cost a round.
  await page.getByTestId('present-diag-toggle').click()
  const diagPanel = page.getByTestId('present-figure-diag')
  await expect(diagPanel).toBeVisible({ timeout: 5_000 })
  const diagText = (await diagPanel.innerText()).replace(/\n/g, ' | ')
  console.log('[sped-ag] on-screen diagnostic =', diagText)
  // NB deliberately NOT asserting which mount holds the figId registration.
  // Sidebar cell and presented slide both register under the same id and the
  // winner is a race — it came out 'present-slide' locally and 'report-sidebar'
  // on CI, on the same build. It no longer decides anything: a frame replays
  // into ITSELF on load (replayState's `target`). What matters is the stash.
  expect(diagText).toContain('2 panel(s)')
  await page.screenshot({ path: join(SHOTS, '04-diagnostic.png') })
  await page.getByTestId('present-diag-toggle').click()
  await expect(diagPanel).toBeHidden({ timeout: 5_000 })

  const dump = await page.evaluate(() => (window as any).__spydeFigureDump?.() ?? null)
  console.log('[sped-ag] figure replay stash =', JSON.stringify(dump, null, 1))
  // The GRID figure — identified by carrying more than one panel's pixels, not
  // by which mount won the registration race (see the note above).
  const presented = (dump ?? []).find((r: any) => String(r.binaryKeyNames).includes(','))
    ?? (dump ?? []).find((r: any) => r.registeredIn === 'present-slide')
  expect(presented, 'no figure carried the composed grid').toBeTruthy()
  console.log('[sped-ag] presented figure: panels=2',
              'binaryKeys=', presented.binaryKeys, 'jsonKeys=', presented.jsonKeys)

  // THE REGRESSION ASSERTION: one stashed pixel frame PER PANEL.
  //
  // Not the pixel probe. That counts anything brighter than the background, so
  // an empty panel's own fill and its HTML scale-bar overlay score as
  // "content" — it reported healthy numbers for the broken figure all the way
  // through this investigation and is why the bug survived so long. The stash
  // count is exact: `binary` short of the panel count means a panel's pixels
  // were overwritten and it WILL present blank.
  expect(presented.binaryKeys,
    `the presented figure stashed ${presented.binaryKeys} pixel frame(s) for 2 panels — `
    + 'a panel will render as an empty box with only its scale bar').toBe(2)
  const names = String(presented.binaryKeyNames)
  expect(names, 'the stash is not keyed per panel (geom::field)').toContain('::image_b64')
  expect(names.split(',').length, 'both panels must be represented in the stash').toBe(2)

  const counts = await panelPixelCounts(page, 2)
  console.log('[sped-ag] per-panel non-background pixel counts (indicative only) =',
              JSON.stringify(counts))
})
