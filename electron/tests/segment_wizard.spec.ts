/**
 * segment_wizard.spec.ts — the Segment Particles caret, end-to-end on the
 * bundled synthetic particle movie.
 *
 * What this actually proves (headless tests + tsc cannot see any of it):
 *   1. The caret opens from the real toolbar button and the backend previews
 *      the DISPLAYED frame — the count line names a number of particles.
 *   2. The DEFAULT face is calm: one slider, one count, one button, one
 *      disclosure. Everything else is behind `▸ Advanced`, which is collapsed
 *      on open and remembers its state.
 *   3. The floating brush strip renders NEXT TO THE PLOT (plan B0) with one
 *      swatch per backend class, and is not clipped by the window.
 *   4. The size histogram (inside Advanced) has bars, not an empty box.
 *   5. `min_size` = 0 is FLOORED by the backend and the caret shows the
 *      EFFECTIVE value, not the 0 the user typed (plan §0.9 — at 0 the split
 *      returns background speckle as particles). Both the field and the warning
 *      live inside Advanced.
 *   6. "Find in all frames" opens a real particle result window.
 *   7. A brush stroke reaches `seg_paint` and the per-class labelled-pixel
 *      counts in the caret update — the counts are how a user notices an
 *      under-trained class, so a count stuck at 0 is a real failure. The class
 *      list is the SCRIBBLE tab's business and is not shown on Classical.
 *   8. The BOUNDARY class is offered, is paintable, and painting it flips the
 *      split route — the caret says `watershed split` before and `seam split`
 *      after, which is the only thing on screen that distinguishes a 0.33 s
 *      split from a 1.78 s one at 4096². Its hover text has to carry the
 *      "paint the seam, not the outline" warning, because the intuitive reading
 *      trains a head that MERGES touching particles (benchmarks.md) and nothing
 *      else on screen says which reading is right.
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

/**
 * Drive the `▸ Advanced` disclosure to a known state. Everything the primary
 * face no longer shows is in there, so most of the assertions below have to open
 * it first — which is the point of the redesign, not an inconvenience.
 *
 * ONE click has to do it, deliberately: the caret is a DOM child of
 * FloatingToolbar, whose placement effect moves it (below ↔ right) when its
 * height changes. Toggling the disclosure changes that height from the WIZARD's
 * own state, which does not re-render the toolbar — so before the toolbar's
 * ResizeObserver existed the placement stayed stale and the caret jumped on the
 * NEXT render, i.e. between the mousedown and the mouseup of the following
 * click. The browser then emits no `click` at all: the control takes focus and
 * silently does nothing, and every other click is ignored. A retry loop here
 * would hide exactly that, so there isn't one.
 */
async function setAdvanced(open: boolean) {
  const { page } = ctx
  const adv = page.getByTestId('seg-advanced')
  const toggle = page.getByTestId('seg-advanced-toggle')
  if (((await adv.count()) > 0) === open) return
  await toggle.click()
  await expect(adv).toHaveCount(open ? 1 : 0)
  await expect(toggle).toHaveAttribute('aria-expanded', String(open))
}

/**
 * The caret box has NO scroller of its own — the Threshold dropdown's menu is
 * absolutely positioned, so an `overflow:auto` ancestor would clip it. That
 * makes "does it fit" a real assertion rather than a cosmetic one: anything
 * past the bottom of the MDI area is simply unreachable, and an expanded
 * Advanced is where it happens.
 *
 * It HAS happened: single-column Advanced measured 907 px in an 805 px area,
 * putting the size histogram and Commit Frame off-screen with no way to reach
 * them. The two-column layout (plan B7) is what buys the room back, so this
 * assertion is the thing holding that layout in place — if a future edit
 * re-stacks Advanced into one column, this is what says so.
 *
 * BOTH AXES, because checking only the bottom is how the next one got through:
 * the two-column caret fit vertically and then FloatingToolbar placed it off
 * the LEFT edge of the app (a side-placed caret anchors its right edge to the
 * window's left, so one wider than the room beside the window walks straight
 * out of the viewport). Vertically-only, this function passed while half the
 * caret was unreachable.
 */
async function expectCaretFits() {
  const { page } = ctx
  const box = await page.getByTestId('segment-wizard').boundingBox()
  const mdi = await page.getByTestId('mdi-area').boundingBox()
  const where = `caret ${JSON.stringify(box)} vs MDI area ${JSON.stringify(mdi)}`
  expect(box && mdi, where).toBeTruthy()
  expect(box!.y + box!.height <= mdi!.y + mdi!.height + 1,
    `${where}: runs past the BOTTOM`).toBe(true)
  expect(box!.x >= mdi!.x - 1, `${where}: runs past the LEFT edge`).toBe(true)
  expect(box!.x + box!.width <= mdi!.x + mdi!.width + 1,
    `${where}: runs past the RIGHT edge`).toBe(true)
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
  // so wait for the count line to stop saying "no preview yet".
  await expect.poll(
    () => page.getByTestId('seg-preview-stats').textContent(),
    { timeout: 60_000, message: 'seg_preview never reached the caret' },
  ).toMatch(/\d+ particles? on this frame/)

  // ── the DEFAULT face is the whole point of the redesign ──────────────────
  // Advanced is collapsed on open, and everything it holds is genuinely absent
  // from the DOM (not merely visually quiet).
  await expect(page.getByTestId('seg-advanced-toggle')).toHaveAttribute('aria-expanded', 'false')
  await expect(page.getByTestId('seg-advanced')).toHaveCount(0)
  for (const hidden of [
    'seg-min-size', 'seg-max-size', 'seg-watershed', 'seg-store-masks',
    'seg-track', 'seg-threshold', 'seg-gaussian', 'seg-rb-kernel',
    'seg-local-size', 'seg-min-separation', 'seg-marker-smooth', 'seg-max-dist',
    'seg-invert', 'seg-clear-border', 'seg-histogram', 'seg-counts',
    'seg-commit', 'seg-min-score',
  ]) {
    await expect(page.getByTestId(hidden),
      `${hidden} must be behind Advanced, not on the default face`).toHaveCount(0)
  }
  // THREE sliders and the button, nothing else besides the tabs, ✕ and the
  // disclosure. ✕, 3 tabs, sensitivity, merge-nm, min-nm, Find-in-all-frames,
  // Advanced = 9. This count is the guard against the face refilling one
  // reasonable-looking addition at a time (plan §0.9a), so it is exact on
  // purpose — if you add a control here, justify it in the diff.
  const primaryControls = await page.getByTestId('segment-wizard')
    .locator('input, select, textarea, button').count()
  expect(primaryControls, 'the default face grew a control back').toBe(9)
  await expect(page.getByTestId('seg-sensitivity')).toBeVisible()
  // The two PHYSICAL controls, the ones that are the same on every engine.
  await expect(page.getByTestId('seg-merge-nm')).toBeVisible()
  await expect(page.getByTestId('seg-min-nm')).toBeVisible()
  await expect(page.getByTestId('seg-run')).toHaveText('Find in all frames')

  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/03-caret-detail.png` })

  // ── Advanced still holds everything, and it still works ──────────────────
  await setAdvanced(true)
  // An EMPTY histogram is the classic "it rendered but says nothing" failure —
  // assert on the bar count the component publishes, not on pixels.
  await expect.poll(
    async () => Number(await page.getByTestId('seg-histogram').getAttribute('data-nonzero')),
    { timeout: 30_000, message: 'size histogram has no populated bins' },
  ).toBeGreaterThan(0)
  await expect(page.getByTestId('seg-min-size')).toBeVisible()
  await expect(page.getByTestId('seg-commit')).toBeVisible()
  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/03b-caret-advanced.png` })
  // Expanded, IN PLACE: Advanced makes the caret tall enough that the toolbar
  // re-places it beside the window. It must still fit inside the MDI area — the
  // caret box has no scroller of its own, so anything past the bottom is simply
  // unreachable.
  await page.screenshot({ path: `${SHOTS}/03c-advanced-full.png` })
  await expectCaretFits()
  await setAdvanced(false)

  // The brush strip belongs to the SCRIBBLE tab only. It floats over the image,
  // so on Classical — where there is nothing to paint — it would be chrome
  // covering the data for no reason.
  await expect(page.getByTestId('seg-class-strip'),
    'the brush strip is showing on Classical, where there is nothing to paint',
  ).toHaveCount(0)

  // The class list carries NAMES + per-class pixel counts (the caret is the
  // authoritative list; the strip is swatches only) — also Scribble's business.
  await expect(page.getByTestId('seg-class-list')).toHaveCount(0)

  await page.getByTestId('seg-tab-scribble').click()
  await expect(page.getByTestId('seg-class-0')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('seg-class-pixels-0')).toBeVisible()
  // ...and NOW the strip appears, next to the plot rather than in the caret.
  const strip = page.getByTestId('seg-class-strip')
  await expect(strip).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('seg-strip-class-0')).toBeVisible()
  await expect(page.getByTestId('seg-strip-brush')).toBeVisible()
  await expect(page.getByTestId('seg-strip-eraser')).toBeVisible()
  // No `+ add class`: the backend has no seg_add_class verb, and a permanently
  // disabled control is noise on a face this redesign just emptied out.
  await expect(page.getByTestId('seg-add-class')).toHaveCount(0)

  await page.getByTestId('seg-tab-classical').click()
  await expect(page.getByTestId('seg-sensitivity')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('seg-class-strip')).toHaveCount(0)

  await page.screenshot({ path: `${SHOTS}/04-preview.png` })
  ctx.assertNoJsErrors()
})

test('sensitivity re-previews and min_size=0 is floored to the EFFECTIVE value', async () => {
  const { page } = ctx
  const stats = page.getByTestId('seg-preview-stats')

  // The COUNT alone is not a reliable "did it re-run" signal — two sensitivities
  // can legitimately find the same number of particles — so poll the caret's own
  // monotonic preview counter instead of diffing the label text.
  const seq0 = Number(await stats.getAttribute('data-seq'))
  await page.getByTestId('seg-sensitivity').fill('0.85')
  await expect.poll(async () => Number(await stats.getAttribute('data-seq')), {
    timeout: 60_000, message: 'dragging sensitivity did not re-preview',
  }).toBeGreaterThan(seq0)
  await page.screenshot({ path: `${SHOTS}/05-sensitivity.png` })

  // min_size=0 is the footgun plan §0.9 measured (33 instances where 9 are
  // real). The backend floors it to 10 and the caret must show 10. Both the
  // field and the warning are inside Advanced now: the floor is applied
  // unconditionally, so the primary face never has to mention it.
  await setAdvanced(true)
  const minSize = page.getByTestId('seg-min-size')
  await minSize.fill('0')
  await minSize.blur()
  await expect(page.getByTestId('seg-min-size-floor')).toBeVisible({ timeout: 60_000 })
  await expect.poll(() => minSize.inputValue(), {
    timeout: 30_000, message: 'caret still shows 0 while the backend ran 10',
  }).toBe('10')

  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/06-minsize-floored.png` })

  // Back to a sane value for the batch below, and back to the calm face.
  await minSize.fill('20')
  await minSize.blur()
  await setAdvanced(false)
  ctx.assertNoJsErrors()
})

/**
 * The three filters that had NO coverage, which is how all three shipped broken
 * at once: the two nm sliders crashed the whole caret on first render (`Field`
 * was used but never imported — a blank window, and every headless test still
 * green), and Confidence was taken off the face and never re-added to Advanced,
 * so `min_score` sat at 0 with no control able to move it.
 *
 * Each assertion below is therefore "the control exists AND reaches the
 * backend", polling the caret's monotonic `data-seq`. Existence alone is what
 * the previous specs checked, and it is exactly what a dead slider passes.
 */
test('the nm face filters and the demoted Confidence slider all re-preview', async () => {
  const { page } = ctx
  const stats = page.getByTestId('seg-preview-stats')
  const reran = async (what: string, act: () => Promise<void>) => {
    const seq = Number(await stats.getAttribute('data-seq'))
    await act()
    await expect.poll(async () => Number(await stats.getAttribute('data-seq')), {
      timeout: 60_000, message: `${what} did not reach the backend`,
    }).toBeGreaterThan(seq)
  }

  // ── the face: both controls are PHYSICAL and read out in nm ───────────────
  // The readout is the assertion that matters. `_nm_to_px` converts with the
  // signal's scale, and nothing stashed that scale on the params — so the
  // backend took its uncalibrated branch and merged at N PIXELS while the label
  // said N nm. A label in nm is a claim about the scale bar; if the conversion
  // is skipped the caret is lying by exactly the magnification.
  await expect(page.getByTestId('seg-merge-nm')).toHaveValue('0')
  await reran('the merge-nm slider', () => page.getByTestId('seg-merge-nm').fill('25'))
  await expect(page.getByTestId('segment-wizard')).toContainText('25 nm')

  await reran('the min-nm slider', () => page.getByTestId('seg-min-nm').fill('4'))
  await expect(page.getByTestId('segment-wizard')).toContainText('4 nm')

  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/06b-nm-filters.png` })

  // Both are engine-independent, so unlike sensitivity they stay on the face
  // when the engine changes. (Scribble is untrained here, which does not matter
  // — this is about which controls render.)
  await page.getByTestId('seg-tab-scribble').click()
  await expect(page.getByTestId('seg-merge-nm')).toBeVisible()
  await expect(page.getByTestId('seg-min-nm')).toBeVisible()
  await expect(page.getByTestId('seg-sensitivity'),
    'sensitivity is classical-only — the scribble engine never reads it').toHaveCount(0)
  await page.getByTestId('seg-tab-classical').click()
  await expect(page.getByTestId('seg-sensitivity')).toBeVisible({ timeout: 30_000 })

  // ── Advanced: Confidence is demoted, NOT deleted (plan §0.9a) ─────────────
  await setAdvanced(true)
  const score = page.getByTestId('seg-min-score')
  await expect(score, 'Confidence left the face and never arrived in Advanced')
    .toBeVisible()
  await expect(page.getByTestId('seg-advanced')).toContainText('off')
  await reran('the Confidence slider', () => score.fill('0.5'))
  await expect(page.getByTestId('seg-advanced')).toContainText('50%')
  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/06c-confidence.png` })

  // Everything back to default so the batch run below is the plain path.
  await reran('resetting Confidence', () => score.fill('0'))
  await setAdvanced(false)
  await reran('resetting merge-nm', () => page.getByTestId('seg-merge-nm').fill('0'))
  await reran('resetting min-nm', () => page.getByTestId('seg-min-nm').fill('0'))
  ctx.assertNoJsErrors()
})

test('Find in all frames segments the movie into a new particle window', async () => {
  const { page } = ctx
  const before = await page.getByTestId('subwindow').count()
  await page.getByTestId('seg-run').click()
  // The button's own status line is the proof the React handler ran at all —
  // a swallowed click leaves the last preview's text sitting there and the
  // 3-minute window poll below then fails for the wrong reason.
  await expect(page.getByTestId('seg-status')).toHaveText(/Segmenting the movie/)
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

  // The counts line is the "did I label enough" readout — demoted into Advanced
  // (the class list already carries the per-class numbers on the face).
  await setAdvanced(true)
  await expect(page.getByTestId('seg-counts')).toContainText('frames labelled')
  await setAdvanced(false)
  ctx.assertNoJsErrors()
})

test('Train fits the scribble classifier and the caret reports it', async () => {
  const { page } = ctx
  await raiseSource()
  const train = page.getByTestId('seg-train')
  await expect(train).toBeEnabled()
  await page.getByTestId('seg-train').click()

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

  // The Scribble face, expanded — its parameters share the SAME disclosure, so
  // nothing moved to a second place.
  await setAdvanced(true)
  await expect(page.getByTestId('seg-min-size')).toBeVisible()
  await expect(page.getByTestId('seg-track')).toBeVisible()
  // The classical MASK knobs are absent here, and not just to save room: the
  // scribble engine hands split_instances a probability map thresholded at 0.5
  // and never reads them (spyde/particles/classical.py::split_instances). Six
  // knobs that do nothing is the overload complaint in miniature.
  for (const dead of ['seg-threshold', 'seg-gaussian', 'seg-rb-kernel',
    'seg-local-size', 'seg-invert', 'seg-sensitivity']) {
    await expect(page.getByTestId(dead),
      `${dead} does not affect the scribble engine and must not be shown`).toHaveCount(0)
  }
  // Scribble's Advanced is the TALLEST state the caret has (class list + train
  // report + parameters); if anything is going to run off the bottom, it is this.
  await expectCaretFits()
  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/12-scribble-advanced.png` })
  await setAdvanced(false)
  await page.getByTestId('segment-wizard').screenshot({ path: `${SHOTS}/13-scribble-collapsed.png` })

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})

test('painting the boundary class flips the split to the seam route', async () => {
  const { page } = ctx
  await raiseSource()

  // The previous test trained with no boundary painted, so the caret is sitting
  // on the watershed route. That is the BEFORE half of this test — without it,
  // asserting "seam split" afterwards would not prove anything flipped.
  await expect(page.getByTestId('seg-trained-note')).toContainText('watershed split')
  await page.getByTestId('segment-wizard').screenshot({
    path: `${SHOTS}/14-before-boundary.png` })

  // The boundary class is offered at all. It is the 4th default class and it is
  // opt-in by construction: unpainted, the split falls back to the watershed.
  const swatch = page.getByTestId('seg-strip-class-3')
  await expect(swatch).toBeVisible()

  // Its hover text must say WHICH boundary to paint. "Boundary" reads as "the
  // outline of a particle" to almost everyone, and a head trained on outlines
  // learns "shrink everything" — measured, it merged the touching pair and lost
  // 40% of the median area while still reporting a trained boundary class and
  // still taking the fast route. The wrong reading is worse than not painting at
  // all AND it is silent, so this tooltip is the only guard there is.
  const tip = await swatch.getAttribute('title')
  expect(tip, `boundary swatch tooltip: ${tip}`).toMatch(/SEAM BETWEEN/)
  expect(tip, 'the tooltip must warn against the outline reading').toMatch(/never the outline/)

  await swatch.click()
  await expect(page.getByTestId('seg-class-3')).toHaveAttribute('data-active', 'true')

  // Paint a seam. Points are IMAGE PIXELS [[y, x], …] — plan trap 6.
  const stroke = async (y: number, x0: number, x1: number) => {
    await page.evaluate(({ id, y, x0, x1 }) => {
      const points: number[][] = []
      for (let x = x0; x <= x1; x += 1) points.push([y, x])
      window.dispatchEvent(new CustomEvent('spyde:figure_event', {
        detail: { figId: id, event: { type: 'brush_stroke', points } },
      }))
    }, { id: figId, y, x0, x1 })
  }
  await page.getByTestId('seg-strip-brush').fill('3')
  await stroke(56, 34, 76)
  await expect.poll(() => page.getByTestId('seg-class-pixels-3').textContent(), {
    timeout: 30_000, message: 'painting the boundary class did not update its count',
  }).not.toMatch(/^!?\s*0$/)

  await page.getByTestId('seg-train').click()
  // Assert on the PERSISTENT report line, not the status: the backend follows
  // seg_trained with a re-preview whose status overwrites it milliseconds later.
  await expect(page.getByTestId('seg-trained-note')).toContainText('seam split', {
    timeout: 180_000 })

  await page.getByTestId('segment-wizard').screenshot({
    path: `${SHOTS}/15-seam-route.png` })
  await page.getByTestId('seg-class-list').screenshot({
    path: `${SHOTS}/15b-class-list-boundary.png` })
  await page.screenshot({ path: `${SHOTS}/16-seam-route-full.png` })

  // A boundary that was painted must still segment — the fast route returning
  // nothing would be a "faster" result that found no particles.
  await expect.poll(async () =>
    Number(await page.getByTestId('seg-preview-stats').getAttribute('data-count')), {
    timeout: 120_000, message: 'no preview after switching to the seam route',
  }).toBeGreaterThan(0)

  const errors = backendErrorLines(ctx.backend)
  expect(errors, `backend errors:\n${errors.join('\n')}`).toEqual([])
  ctx.assertNoJsErrors()
})
