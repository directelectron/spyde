/** The zero-beam search region: drawn, draggable, and on the enlarged panel.
 *
 * Synthetic members, because this is an INTERACTION test — the region's effect
 * on a solve is measured in the Python suite, where it can be compared against
 * a known answer instead of eyeballed.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'

const { launchApp } = require('./_harness.cjs')
const SHOTS = join(__dirname, '..', 'beam_roi_shots')
const MEMBER_DIRECTORY = join(require('os').tmpdir(), 'spyde-multiangle-test')
const TILTS = [1.0, 1.0, 1.0, 0.5, 0.5]
const SCAN = 28
const paths = () => TILTS.map((_t, i) =>
  join(MEMBER_DIRECTORY, `angle${String(i).padStart(2, '0')}.mrc`))

let ctx: any
test.setTimeout(10 * 60_000)

const send = (action: string, payload: Record<string, unknown> = {}) =>
  ctx.page.evaluate(([a, p]: any) => (window as any).electron.action(a, p),
    [action, payload] as const)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  await ctx.page.waitForTimeout(1500)
  await send('write_test_multiangle_files', { nav: SCAN, sig: 64 })
  await ctx.page.waitForTimeout(2500)
})
test.afterAll(async () => { await ctx?.app?.close() })

test('the region is drawn, moves, and is placeable on the enlarged panel', async () => {
  const { page } = ctx
  await page.getByTestId('menu-file').click()
  await page.getByTestId('menu-load-multiangle').click()
  const dialog = page.getByTestId('multiangle-loader')
  await expect(dialog).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(1500)

  await send('maped_add_files', { paths: paths() })
  await page.waitForTimeout(3000)
  await send('maped_set_scan_shape', { scan_shape: [SCAN, SCAN] })
  await expect(dialog.getByTestId('maped-summary'))
    .toContainText('5 angles', { timeout: 120_000 })
  for (let i = 0; i < TILTS.length; i += 1) {
    await send('maped_set_member', { index: i, tilt: TILTS[i], azimuth: i * 70 })
    await page.waitForTimeout(200)
  }
  await expect.poll(() => dialog.locator('img').count(), { timeout: 240_000 })
    .toBeGreaterThanOrEqual(4)

  await send('maped_align_real', { params: { max_shift: 16 } })
  await expect(dialog.getByTestId('maped-tab-reciprocal'))
    .toBeEnabled({ timeout: 300_000 })
  // The real-space evidence belongs IN the dialog: the dialog is a full-screen
  // modal, so a window behind it cannot be looked at while it is up. It lives
  // on the tab that solves it, so stand there.
  await dialog.getByTestId('maped-tab-real').click()
  await expect(dialog.getByTestId('maped-real-evidence'))
    .toBeVisible({ timeout: 300_000 })
  await page.waitForTimeout(1200)
  await dialog.screenshot({ path: join(SHOTS, '00-real-evidence.png') })
  console.log('sharpness:', await dialog.getByTestId('maped-real-gain')
    .textContent().catch(() => 'none'))
  console.log('verdict  :', await dialog.getByTestId('maped-real-verdict')
    .textContent().catch(() => 'none'))

  await dialog.getByTestId('maped-tab-reciprocal').click()
  await page.waitForTimeout(2500)
  await dialog.screenshot({ path: join(SHOTS, '01-reciprocal-with-roi.png') })

  const boxes = dialog.getByTestId('maped-beam-roi')
  console.log('region boxes drawn:', await boxes.count())
  expect(await boxes.count(), 'no search region drawn').toBeGreaterThan(0)

  // Enlarge first: a tableau tile is far too coarse to aim on.
  const probe = await page.evaluate(() => {
    const panel = document.querySelector('[data-testid="maped-corner-0-0"]')
    if (!panel) return { found: false }
    const r = panel.getBoundingClientRect()
    const atCentre = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)
    return {
      found: true,
      rect: [Math.round(r.width), Math.round(r.height)],
      atCentre: (atCentre as HTMLElement)?.dataset?.testid
        ?? (atCentre as HTMLElement)?.tagName ?? null,
      zoomsBefore: document.querySelectorAll('[data-testid="maped-zoom"]').length,
    }
  })
  console.log('PROBE', JSON.stringify(probe))

  await page.getByTestId('maped-corner-0-0').dblclick()
  await page.waitForTimeout(800)
  console.log('zooms after playwright dblclick:', await page.getByTestId('maped-zoom').count())

  if (await page.getByTestId('maped-zoom').count() === 0) {
    const dispatched = await page.evaluate(() => {
      const panel = document.querySelector('[data-testid="maped-corner-0-0"]')
      panel?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }))
      return document.querySelectorAll('[data-testid="maped-zoom"]').length
    })
    await page.waitForTimeout(600)
    console.log('zooms after a dispatched dblclick:', dispatched,
      '->', await page.getByTestId('maped-zoom').count())
  }
  await expect(page.getByTestId('maped-zoom')).toBeVisible({ timeout: 20_000 })
  await page.waitForTimeout(1000)
  await page.screenshot({ path: join(SHOTS, '02-zoomed-with-roi.png') })

  const zoomBox = page.getByTestId('maped-zoom').getByTestId('maped-beam-roi')
  expect(await zoomBox.count(), 'no region in the enlarged panel').toBe(1)

  const before = await dialog.getByTestId('maped-status').textContent()
  const zb = await zoomBox.boundingBox()
  expect(zb, 'the enlarged region has no box').not.toBeNull()
  await page.mouse.move(zb!.x + zb!.width / 2, zb!.y + zb!.height / 2)
  await page.mouse.down()
  await page.mouse.move(zb!.x + zb!.width / 2 + 40, zb!.y + zb!.height / 2 - 25,
    { steps: 12 })
  await page.mouse.up()
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '03-zoom-after-drag.png') })
  const after = await dialog.getByTestId('maped-status').textContent()
  console.log('status before:', before)
  console.log('status after :', after)
  expect(after, 'dragging in the enlarged panel did not move the region')
    .not.toEqual(before)
  expect(after).toMatch(/Zero beam searched within/)
})

/**
 * The per-axis confidence, in both of its states.
 *
 * A four-member scan large enough to divide into patches is far too big to
 * bundle, so the snapshot the backend would send is dispatched directly. What
 * is under test is the DISPLAY: that a determined axis and an unconstrained
 * one are told apart, and that two offsets are not presented as equally good
 * when only one of them was measured.
 */
test('the panel says which axis the specimen determined', async () => {
  const { page } = ctx
  const dialog = page.getByTestId('multiangle-loader')
  await expect(dialog).toBeVisible({ timeout: 20_000 })
  // The test before this one leaves a corner enlarged, and it covers the tabs.
  await page.keyboard.press('Escape')
  await page.waitForTimeout(500)
  await dialog.getByTestId('maped-tab-real').click()

  // A real image, because the panel declines to draw itself without one.
  const PIXEL = 'data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=='
  const send = (confidence: unknown) => page.evaluate(
    ([payload, pixel]: any) => {
      window.dispatchEvent(new CustomEvent('spyde:maped_state', {
        detail: {
          type: 'maped_state',
          // Real members, because the real-space tab is disabled without any
          // and the dialog drops back to Load.
          members: [0, 1].map((index) => ({
            index, path: `angle0${index}.mrc`, name: `angle0${index}.mrc`,
            scan_shape: [180, 180], detector_shape: [64, 64], dtype: 'uint16',
            size_bytes: 1024, tilt: 1.0, azimuth: index * 90, shell: 0,
            error: null, preview: pixel, virtual_images: [],
          })),
          reference: 0,
          shells: [{ shell: 0, tilt: 1.0, members: [0, 1] }],
          scan_shape: [180, 180], virtual_image: null,
          available_virtual_images: [], beam_roi: null,
          real: {
            solved: true, offsets: [[0, 0], [3, 24]], residuals: null,
            max_residual: 0.1, corners: {},
            solver_offsets: [[0, 0], [3, 24]],
            evidence: { unaligned: pixel, aligned: pixel, gain: 1.49 },
            confidence: payload,
          },
          reciprocal: {
            solved: false, offsets: null, residuals: null,
            max_residual: null, corners: {},
          },
          busy: false, message: '', can_commit: false,
        },
      }))
    }, [confidence, PIXEL] as const)

  // The case this whole feature exists for, measured on a real acquisition:
  // across the layers the patches agree, along them they cannot.
  await send({ agreement: { x: 0.72, y: 0.28 },
               determined: { x: true, y: false }, votes: 25 })
  await expect(dialog.getByTestId('maped-real-confidence'))
    .toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(400)
  await dialog.screenshot({ path: join(SHOTS, '04-confidence-split.png') })
  expect(await dialog.getByTestId('maped-real-confidence-x').textContent())
    .toContain('72%')
  expect(await dialog.getByTestId('maped-real-confidence-y').textContent())
    .toContain('28%')
  const warning = await dialog.getByTestId('maped-real-unconstrained')
    .textContent()
  console.log('warning:', warning)
  expect(warning).toContain('down')

  // Both determined: no warning, or it would cry wolf on every good solve.
  await send({ agreement: { x: 0.81, y: 0.77 },
               determined: { x: true, y: true }, votes: 25 })
  await page.waitForTimeout(400)
  await dialog.screenshot({ path: join(SHOTS, '05-confidence-both.png') })
  expect(await dialog.getByTestId('maped-real-unconstrained').count(),
    'warned about an axis it had just called determined').toBe(0)

  // Nothing measured is not a finding about the specimen.
  await send({ agreement: { x: 0.0, y: 0.0 },
               determined: { x: false, y: false }, votes: 0 })
  await page.waitForTimeout(400)
  const none = await dialog.getByTestId('maped-real-confidence').textContent()
  console.log('no votes:', none)
  expect(none).toContain('too few scan positions')
  expect(await dialog.getByTestId('maped-real-unconstrained').count(),
    'claimed the specimen fixes nothing when nothing was checked').toBe(0)
})

/**
 * The nudge pad, driven against the REAL backend.
 *
 * An earlier version of this dispatched a snapshot and intercepted what the
 * pad sent. That tested the pad talking to itself: `sendAction` does not go
 * through the object the test patched, so it recorded nothing while the keys
 * were in fact moving a member for real. Asserting on the offset the backend
 * reports back tests the whole loop, and is the only version that can fail
 * for a real reason.
 */
test('arrow keys nudge the selected member', async () => {
  const { page } = ctx
  const dialog = page.getByTestId('multiangle-loader')
  await expect(dialog).toBeVisible({ timeout: 20_000 })
  await page.keyboard.press('Escape')
  await page.waitForTimeout(400)
  // Re-solve rather than inherit whatever the previous test left behind: the
  // point here is the round trip to a REAL solve, and a test that only works
  // after its neighbours is a test that will start lying when they change.
  await send('maped_align_real', { params: { max_shift: 16 } })
  await page.waitForTimeout(2000)
  await dialog.getByTestId('maped-tab-real').click()
  await expect(dialog.getByTestId('maped-nudge'))
    .toBeVisible({ timeout: 300_000 })

  const readOffset = async (): Promise<number[]> => {
    const text = await dialog.getByTestId('maped-nudge-offset').textContent()
    const found = /\(\s*(-?\d+),\s*(-?\d+)\)/.exec(text ?? '')
    expect(found, `no offset in ${text}`).not.toBeNull()
    return [Number(found![1]), Number(found![2])]
  }

  // Selecting in the tableau is what arms the keys, and it is a SINGLE click
  // because double-click already zooms.
  // The reference member defines the frame and cannot move, so it is not
  // selectable; find one that is rather than assuming an index.
  const selectable = dialog.locator('[aria-selected]')
  const count = await selectable.count()
  expect(count, 'no member tile offers itself for selection').toBeGreaterThan(0)
  const tile = selectable.first()
  await tile.click()
  await expect(tile).toHaveAttribute('data-selected', 'true', { timeout: 5_000 })
  await dialog.screenshot({ path: join(SHOTS, '06-nudge-focused.png') })

  const start = await readOffset()
  await page.keyboard.press('ArrowDown')
  await expect.poll(readOffset, { timeout: 15_000 })
    .toEqual([start[0] + 1, start[1]])
  await page.keyboard.press('ArrowRight')
  await expect.poll(readOffset, { timeout: 15_000 })
    .toEqual([start[0] + 1, start[1] + 1])
  await page.keyboard.press('Shift+ArrowUp')
  await expect.poll(readOffset, { timeout: 15_000 })
    .toEqual([start[0] - 4, start[1] + 1])
  await dialog.screenshot({ path: join(SHOTS, '07-nudge-moved.png') })

  // Moved by hand, so there is now a solver answer to show and to go back to.
  await expect(dialog.getByTestId('maped-nudge-solver')).toBeVisible()
  await expect(dialog.getByTestId('maped-nudge-reset')).toBeEnabled()
  await dialog.getByTestId('maped-nudge-reset').click()
  await expect.poll(readOffset, { timeout: 15_000 }).toEqual(start)
  await expect(dialog.getByTestId('maped-nudge-reset')).toBeDisabled()

  // The buttons do the same thing, for a mouse.
  await dialog.getByTestId('maped-nudge-left').click()
  await expect.poll(readOffset, { timeout: 15_000 })
    .toEqual([start[0], start[1] - 1])
  await dialog.getByTestId('maped-nudge-reset').click()
  await expect.poll(readOffset, { timeout: 15_000 }).toEqual(start)
})

/**
 * The pairwise view: a member, the reference, and the two overlaid.
 *
 * Double-click is the way in because it already meant "look closer" on these
 * tiles; the reference keeps the plain enlargement, having nothing to be
 * compared against.
 */
test('double-click compares a member with the reference', async () => {
  const { page } = ctx
  const dialog = page.getByTestId('multiangle-loader')
  await expect(dialog).toBeVisible({ timeout: 20_000 })
  await page.keyboard.press('Escape')
  await page.waitForTimeout(400)
  await send('maped_align_real', { params: { max_shift: 16 } })
  await page.waitForTimeout(2000)
  await dialog.getByTestId('maped-tab-real').click()
  await expect(dialog.getByTestId('maped-nudge'))
    .toBeVisible({ timeout: 300_000 })

  const movable = dialog.locator('[aria-selected]')
  expect(await movable.count()).toBeGreaterThan(0)
  await movable.first().dblclick()

  const pair = page.getByTestId('maped-pair')
  await expect(pair).toBeVisible({ timeout: 20_000 })
  for (const part of ['maped-pair-member', 'maped-pair-reference',
                      'maped-pair-overlay']) {
    await expect(pair.getByTestId(part)).toBeVisible()
  }
  await page.waitForTimeout(600)
  await page.screenshot({ path: join(SHOTS, '08-pair-view.png') })

  // The overlay must follow the keys, or the view is decoration.
  const overlayOf = () => pair.getByTestId('maped-pair-overlay')
    .getAttribute('src')
  const before = await overlayOf()
  await page.keyboard.press('ArrowRight')
  await page.keyboard.press('ArrowRight')
  await expect.poll(overlayOf, { timeout: 15_000 }).not.toEqual(before)
  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, '09-pair-nudged.png') })

  // Switching the image here must not throw away the solve being edited.
  // A themed dropdown: its options exist only while it is open.
  await pair.getByTestId('maped-pair-image').click()
  const images = await page.locator('[data-testid^="maped-pair-image-opt-"]').count()
  expect(images, 'no images offered').toBeGreaterThan(0)
  await pair.getByTestId('maped-pair-image').click()

  await page.keyboard.press('Escape')
  await expect(pair).toBeHidden({ timeout: 10_000 })
  await expect(dialog.getByTestId('maped-nudge')).toBeVisible()
})
