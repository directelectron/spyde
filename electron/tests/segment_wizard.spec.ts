/**
 * segment_wizard.spec.ts — the Segment Particles caret, end-to-end on the
 * bundled synthetic particle movie.
 *
 * What this actually proves (headless tests + tsc cannot see any of it):
 *   1. The caret opens from the real toolbar button and the backend previews
 *      the DISPLAYED frame — the size histogram has bars, not an empty box.
 *   2. The floating brush strip renders NEXT TO THE PLOT (plan B0) with one
 *      swatch per backend class, and is not clipped by the window.
 *   3. `min_size` = 0 is FLOORED by the backend and the caret shows the
 *      EFFECTIVE value, not the 0 the user typed (plan §0.9 — at 0 the split
 *      returns background speckle as particles).
 *   4. Run All opens a real particle result window.
 *   5. A brush stroke reaches `seg_paint` and the per-class labelled-pixel
 *      counts in the caret update — the counts are how a user notices an
 *      under-trained class, so a count stuck at 0 is a real failure.
 *
 * Real Dask + `load_test_data_particles` (lazy, 1 frame/chunk, ground truth
 * stamped into metadata) — the path a user actually drags.
 */
import { test, expect } from '@playwright/test'
import { mkdirSync } from 'fs'
const {
  launchApp, backendAction, waitForSubwindowCount, sigWindow, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = 'segment_wizard_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>
/** `data-testid="figure-<figId>"` on the signal window's iframe. */
let figId = ''

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  // INFO tees `logging` to stderr, which the harness captures — backend
  // emit_error goes over the PLOTAPP protocol and never reaches this buffer.
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  // backend-ready can land a beat before the stdin pump is live; the lazy specs
  // settle the same way so the first action isn't dropped.
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_particles', { frames: 6 })
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2000)
})

test.afterAll(async () => {
  await ctx?.app?.close()
})

/** The SOURCE movie's signal window specifically — `sigWindow` picks the first
 *  `S-` window, and Run All adds a second one ("S-Particles — 6 frames"). */
function srcWindow() {
  const { page } = ctx
  return page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-Synthetic/ }),
  }).first()
}

/** Raise the source window above the result windows Run All cascades on top of
 *  it. The caret lives in that window's stacking context, so a newer window
 *  stacked above ALSO covers the caret and swallows clicks meant for it — the
 *  same thing a user does (click the window) fixes it. */
async function raiseSource() {
  await srcWindow().getByTestId('subwindow-title').click()
}

/** Open the caret from the REAL toolbar button (hover the titlebar to reveal
 *  the floating bar), exactly as a user does. */
async function openCaret() {
  const { page } = ctx
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Segment Particles').click()
  await expect(page.getByTestId('segment-wizard')).toBeVisible()
  return sig
}

test('caret opens, previews the displayed frame, and shows the brush strip', async () => {
  const { page } = ctx
  await page.screenshot({ path: `${SHOTS}/01-movie-loaded.png` })

  const sig = await openCaret()
  const tid = await sig.locator('iframe').first().getAttribute('data-testid')
  figId = (tid ?? '').replace(/^figure-/, '')
  expect(figId).not.toBe('')

  await page.screenshot({ path: `${SHOTS}/02-caret-open.png` })

  // The preview is a real backend round trip (seg_open → worker → seg_preview),
  // so wait for the stats line to stop saying "no preview yet".
  await expect.poll(
    () => page.getByTestId('seg-preview-stats').textContent(),
    { timeout: 60_000, message: 'seg_preview never reached the caret' },
  ).toMatch(/found/)

  // An EMPTY histogram is the classic "it rendered but says nothing" failure —
  // assert on the bar count the component publishes, not on pixels.
  await expect.poll(
    async () => Number(await page.getByTestId('seg-histogram').getAttribute('data-nonzero')),
    { timeout: 30_000, message: 'size histogram has no populated bins' },
  ).toBeGreaterThan(0)

  // The brush strip is next to the PLOT, not in the caret (plan B0).
  const strip = page.getByTestId('seg-class-strip')
  await expect(strip).toBeVisible()
  await expect(page.getByTestId('seg-strip-class-0')).toBeVisible()
  await expect(page.getByTestId('seg-strip-brush')).toBeVisible()
  await expect(page.getByTestId('seg-strip-eraser')).toBeVisible()

  // The class list carries NAMES + per-class pixel counts (the caret is the
  // authoritative list; the strip is swatches only).
  await expect(page.getByTestId('seg-class-0')).toBeVisible()
  await expect(page.getByTestId('seg-class-pixels-0')).toBeVisible()

  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/03-caret-detail.png` })
  await page.screenshot({ path: `${SHOTS}/04-preview.png` })
  ctx.assertNoJsErrors()
})

test('sensitivity re-previews and min_size=0 is floored to the EFFECTIVE value', async () => {
  const { page } = ctx
  const stats = page.getByTestId('seg-preview-stats')

  const before = await stats.textContent()
  await page.getByTestId('seg-sensitivity').fill('0.85')
  await expect.poll(() => stats.textContent(), {
    timeout: 60_000, message: 'dragging sensitivity did not re-preview',
  }).not.toBe(before)
  await page.screenshot({ path: `${SHOTS}/05-sensitivity.png` })

  // min_size=0 is the footgun plan §0.9 measured (33 instances where 9 are
  // real). The backend floors it to 10 and the caret must show 10.
  const minSize = page.getByTestId('seg-min-size')
  await minSize.fill('0')
  await minSize.blur()
  await expect(page.getByTestId('seg-min-size-floor')).toBeVisible({ timeout: 60_000 })
  await expect.poll(() => minSize.inputValue(), {
    timeout: 30_000, message: 'caret still shows 0 while the backend ran 10',
  }).toBe('10')

  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/06-minsize-floored.png` })

  // Back to a sane value for the batch below.
  await minSize.fill('20')
  await minSize.blur()
  ctx.assertNoJsErrors()
})

test('Run All segments the movie into a new particle window', async () => {
  const { page } = ctx
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('seg-run').click()
  await expect.poll(() => page.getByTestId('subwindow').count(), {
    timeout: 180_000, message: 'the particle result window never opened',
  }).toBeGreaterThan(before)
  await page.screenshot({ path: `${SHOTS}/07a-run-early-window.png` })

  // The window opens EARLY with an empty store; `tree.particles` attaches only
  // at _finalize, which re-sends the toolbar — so the requires_particles-gated
  // buttons appearing IS the "batch finished" signal, exactly as
  // `particles_action._rebuild_toolbars` documents ("the e2e specs wait on
  // exactly this appearing"). NB the status bar is NOT usable here: seg_run's
  // last emit_progress leaves busy=true, and StatusBar shows loading.text over
  // status while busy — so "Found N particles" never becomes visible.
  // Depends on the requires_particles toolbar entries (plan B9, Particle
  // Overlay / Particle Lanes) being present in spyde/toolbars.yaml.
  await expect.poll(
    () => page.getByTestId('action-btn-Particle Overlay').count(),
    { timeout: 180_000, message: 'the segmentation batch never finalized' },
  ).toBeGreaterThan(0)
  await page.waitForTimeout(2000)
  await page.screenshot({ path: `${SHOTS}/07-run-all.png` })
  ctx.assertNoJsErrors()
})

test('a brush stroke reaches seg_paint and the class pixel counts update', async () => {
  const { page } = ctx

  await raiseSource()
  await page.getByTestId('seg-tab-scribble').click()
  await expect(page.getByTestId('seg-scribble-note')).toBeVisible()

  const pixels0 = page.getByTestId('seg-class-pixels-0')
  expect(await pixels0.textContent()).toContain('0')

  // Fatten the brush first, then paint with the strip's ACTIVE class — i.e.
  // drive the same state the strip owns, not a synthetic payload.
  await page.getByTestId('seg-strip-brush').fill('7')
  await page.getByTestId('seg-strip-class-1').click()

  // The anyplotlib brush widget (plan B0) is not landed, so post the widget
  // event the caret listens for. Points are IMAGE PIXELS [[y, x], …] with no
  // scale/offset applied — plan trap 6, and what seg_paint documents.
  const stroke = async (y: number, x0: number, x1: number) => {
    await page.evaluate(({ id, y, x0, x1 }) => {
      const points: number[][] = []
      for (let x = x0; x <= x1; x += 1) points.push([y, x])
      window.dispatchEvent(new CustomEvent('spyde:figure_event', {
        detail: { figId: id, event: { type: 'brush_stroke', points } },
      }))
    }, { id: figId, y, x0, x1 })
  }
  // Selecting on the strip drives the caret's class list too — ONE active
  // class, not two highlighted rows.
  await expect(page.getByTestId('seg-class-1')).toHaveAttribute('data-active', 'true')
  await expect(page.getByTestId('seg-class-0')).toHaveAttribute('data-active', 'false')

  await stroke(20, 20, 90)                       // class 1 (support film)
  await expect.poll(() => page.getByTestId('seg-class-pixels-1').textContent(), {
    timeout: 30_000, message: 'seg_paint never updated the class pixel counts',
  }).not.toMatch(/^!?\s*0$/)

  await page.getByTestId('seg-strip-class-0').click()
  await expect(page.getByTestId('seg-class-0')).toHaveAttribute('data-active', 'true')
  await expect(page.getByTestId('seg-class-1')).toHaveAttribute('data-active', 'false')
  await stroke(48, 30, 80)                       // class 0 (particle)
  await expect.poll(() => pixels0.textContent(), {
    timeout: 30_000, message: 'painting class 0 did not update its count',
  }).not.toMatch(/^!?\s*0$/)

  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/08-painted.png` })
  // A close-up of the class list on its own: the per-class counts and the
  // active-row indication are the two things here that have to be legible.
  await page.getByTestId('seg-class-list').screenshot({ path: `${SHOTS}/08b-class-list.png` })
  // EXACTLY ONE row may look selected. A previously-active row used to keep a
  // stale white 1px border (React clears a dropped `borderColor` longhand but
  // leaves the base `border` shorthand's width/style), so two rows read as
  // selected at once — invisible to every attribute assertion above.
  const borders = await page.evaluate(() =>
    [...document.querySelectorAll('[data-testid^="seg-class-"]')]
      .filter(e => /^seg-class-\d+$/.test(e.getAttribute('data-testid') ?? ''))
      .map(e => getComputedStyle(e).borderTopColor))
  expect(borders.filter(c => c !== 'rgba(0, 0, 0, 0)' && c !== 'transparent'),
    `rows with a visible border: ${JSON.stringify(borders)}`).toHaveLength(1)
  await page.screenshot({ path: `${SHOTS}/09-painted-full.png` })

  // The counts line is the "did I label enough" readout.
  await expect(page.getByTestId('seg-counts')).toContainText('frames labelled')
  ctx.assertNoJsErrors()
})

test('Train fits the scribble classifier and the caret reports it', async () => {
  const { page } = ctx
  await raiseSource()
  const train = page.getByTestId('seg-train')
  await expect(train).toBeEnabled()
  await train.click()

  // Assert on the PERSISTENT report line, not the status: the backend follows
  // seg_trained with a re-preview whose status overwrites it milliseconds later
  // (a poll on the status races that and loses).
  await expect(page.getByTestId('seg-trained-note')).toBeVisible({ timeout: 180_000 })
  await expect(page.getByTestId('seg-trained-note')).toContainText(/Trained on \d+ px/)
  // Training flips the engine to scribble and unlocks the batch.
  await expect(page.getByTestId('seg-scribble-note')).toHaveCount(0)
  await expect(page.getByTestId('seg-run')).toBeEnabled()

  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/10-trained.png` })
  await page.screenshot({ path: `${SHOTS}/11-trained-full.png` })

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})
