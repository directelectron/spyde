/**
 * _seg.ts — shared setup for the Segment Particles specs.
 *
 * With the classical engine gone the caret has NOTHING to show until a
 * classifier is trained, so "open the caret and read the count" — which every
 * one of these specs used to start with — is no longer a valid opening move.
 *
 * `trainFromGroundTruth` is that opening move now. It uses the `seg_autolabel`
 * TEST DOOR, which paints through the same rasteriser the real brush uses at
 * coordinates taken from the synthetic fixture's stamped ground truth, and then
 * presses Train like a user. Only the PLACEMENT of the strokes is scripted; the
 * training is the real thing.
 *
 * The genuine paint path — a brush event reaching `seg_paint`, the per-class
 * pixel counts updating, the eraser, the boundary class — is tested for real in
 * `segment_wizard.spec.ts`. It is deliberately NOT duplicated here: a spec about
 * the overlay has no opinion about where a stroke lands, and hand-placed
 * coordinates in three specs would all go stale together the day the fixture
 * moves a particle.
 */
import { expect, type Page } from '@playwright/test'

/** Open the Segment caret from the real toolbar button on a signal window. */
export async function openCaret(page: Page, sig: ReturnType<Page['locator']>) {
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Segment Particles').click()
  await expect(page.getByTestId('segment-wizard')).toBeVisible()
}

/**
 * Label from ground truth and Train, leaving the caret with a live preview.
 * Resolves once the count line reports a real number.
 *
 * `mislabel: true` SWAPS the classes, teaching the head that the support film
 * is what you are looking for. That is how a spec reproduces over-segmentation
 * now: a CORRECTLY trained head does not over-segment even a heavily noisy
 * frame, which is measured and is the argument for deleting the engine that
 * did.
 */
export async function trainFromGroundTruth(page: Page, windowId: number,
                                           opts: { mislabel?: boolean,
                                                   timeout?: number } = {}) {
  const timeout = opts.timeout ?? 180_000
  // `window.electron.action(action, payload, windowId)` is exactly what the
  // caret's own buttons call (SpyDEContext's sendAction) — the third argument
  // is what routes a staged `seg_` verb to the right plot, so `backendAction`
  // from the harness cannot be used here: it omits it.
  await page.evaluate(
    ({ id, mislabel }) => (window as unknown as {
      electron: { action: (a: string, p: unknown, w: number) => void }
    }).electron.action('seg_autolabel', { mislabel }, id),
    { id: windowId, mislabel: !!opts.mislabel })

  // The class list is the proof the labels landed — a class still on 0 px means
  // autolabel painted nothing and Train would fit on an empty store.
  await expect
    .poll(() => page.getByTestId('seg-class-pixels-0').textContent(),
      { timeout: 60_000, message: 'seg_autolabel painted no particle pixels' })
    .not.toMatch(/^!?\s*0$/)

  await page.getByTestId('seg-train').click()
  await expect(page.getByTestId('seg-trained-note')).toBeVisible({ timeout })
  await expect
    .poll(async () => Number(await page.getByTestId('seg-preview-stats')
      .getAttribute('data-count')),
      { timeout, message: 'no preview after training' })
    .toBeGreaterThan(0)
}

/** The `windowId` the caret is bound to, read off the owning subwindow. */
export async function windowIdOf(sig: ReturnType<Page['locator']>) {
  const attr = await sig.getAttribute('data-window-id')
  return Number(attr)
}
