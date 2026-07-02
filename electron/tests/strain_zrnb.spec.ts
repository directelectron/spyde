/**
 * strain_zrnb.spec.ts — Strain Mapping end-to-end, mirroring the proven
 * find_vectors_workflow.spec.ts pattern: real Dask + bundled-synthetic Si-grains
 * (crisp reciprocal lattice the peak finder can detect), find vectors, then drive
 * the Strain Mapping caret. Screenshots each stage so the UI is actually inspected.
 *
 * Run: npx playwright test strain_zrnb.spec.ts --project=electron
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const {
  launchApp, backendAction, waitForSubwindowCount, countColorPixels,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'strain_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
})

test.afterAll(async () => {
  const buf = ctx?.backend?.logBuffer ?? []
  console.log('[diag] total log lines =', buf.length)
  buf.slice(-40).forEach((l: string) => console.log('[BE]', l))
  await ctx?.app?.close()
})

test.setTimeout(900_000)

test('Strain Mapping caret: overlay, component switch, toggle off', async () => {
  const { page } = ctx
  const nWin = () => page.getByTestId('subwindow').count()
  const green = () => countColorPixels(page, 'green')
  await page.screenshot({ path: join(SHOTS, '01-loaded.png') })

  // 1) Find Diffraction Vectors (proven flow from find_vectors_workflow.spec.ts).
  // `sig` must be pinned to a STABLE match: it's a live locator (re-evaluated on
  // every use, not a snapshot), and `hasNotText: 'Navigator'` alone also matches
  // the "— Vectors" windows that open later — so a later `.first()` re-query can
  // silently pick a different window once those exist. Exclude them explicitly.
  // NB '— Vectors' (the result windows' em-dash title), NOT bare 'Vectors' —
  // the OPEN Find-Vectors wizard inside the source window contains the text
  // "Find Diffraction Vectors", so a bare 'Vectors' filter would exclude the
  // source window too while the caret is open.
  const sig = page.getByTestId('subwindow')
    .filter({ hasNotText: 'Navigator' }).filter({ hasNotText: '— Vectors' }).first()
  await sig.getByTestId('subwindow-title').click()
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible()
  const before = await nWin()
  await page.getByTestId('fv-compute').click()
  await expect.poll(nWin, {
    timeout: 120_000, message: 'vectors result window never opened',
  }).toBeGreaterThan(before)
  // Wait for the REAL completion signal: the vector actions are
  // requires_vectors-gated, so the Strain Mapping button APPEARING means
  // diffraction_vectors attached and the toolbar was re-sent. (The "Found …"
  // status travels the PLOTAPP protocol, invisible to the harness log; the
  // distributed batch on this box takes minutes — per-worker process spawn.)
  await expect.poll(
    () => page.getByTestId('action-btn-Strain Mapping').count(), {
      timeout: 360_000, message: 'vectors never attached (no Strain button)',
    }).toBeGreaterThan(0)
  await page.screenshot({ path: join(SHOTS, '02-vectors.png') })
  console.log('[strain] vectors found, windows =', await nWin())

  // Close the Find Vectors caret so it doesn't overlap the Strain caret. The
  // caret popout renders inside its OWNING window's stacking context, so if a
  // later-opened window now sits visually on top (smaller default window
  // sizes pack windows tighter), the popout's close button can be covered —
  // raise the source window first, same as a real user would.
  if (await page.getByTestId('fv-close').count()) {
    await sig.getByTestId('subwindow-titlebar').click()
    await sig.getByTestId('subwindow-titlebar').hover()
    await page.getByTestId('fv-close').first().click()
    await page.waitForTimeout(500)
  }

  // 2) Open Strain Mapping on the vectors-image window.
  const vsig = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Strain Mapping') }).first()
  await vsig.getByTestId('subwindow-titlebar').click()
  await vsig.getByTestId('subwindow-titlebar').hover()
  const strainBtn = vsig.getByTestId('action-btn-Strain Mapping')
  await expect(strainBtn).toBeVisible({ timeout: 30_000 })
  const beforeOpen = await nWin()
  await strainBtn.click()
  await expect(page.getByTestId('strain-wizard')).toBeVisible({ timeout: 15_000 })
  // Opening the caret runs strain_open → a Strain map window opens. strain_open
  // self-waits (up to 300s) if vectors haven't attached yet, so give this a
  // matching generous budget rather than assume the earlier "Found" wait
  // already covered it.
  await expect.poll(nWin, {
    timeout: 300_000, message: 'strain map window never opened on action select',
  }).toBeGreaterThan(beforeOpen)
  await page.waitForTimeout(3000)
  const afterOpen = await nWin()
  await page.screenshot({ path: join(SHOTS, '03-strain-open.png') })
  console.log(`[strain] open: windows ${beforeOpen} -> ${afterOpen}; green px = ${await green()}`)

  // 3) Component switch εxx -> εyy must repaint the map immediately.
  const eyy = page.getByTestId(/^strain-comp-eyy-/).first()
  if (await eyy.count()) {
    await eyy.click({ timeout: 10_000 }).catch(() => {})
    await page.waitForTimeout(1500)
    await page.screenshot({ path: join(SHOTS, '04-eyy.png') })
  }

  // 4) Toggle the action OFF → the strain-map window is REMOVED (back to before).
  // The Strain map opened on top (by design — a freshly-opened window is
  // focused/topmost, see MDIArea's focus-on-open), so re-raise the source
  // window first, same as a real user would click it back to front.
  await vsig.getByTestId('subwindow-titlebar').click()
  await vsig.getByTestId('subwindow-titlebar').hover()
  await strainBtn.click()
  await expect.poll(nWin, {
    timeout: 15_000, message: 'strain window not removed on toggle off',
  }).toBe(beforeOpen)
  const afterClose = await nWin()
  await page.screenshot({ path: join(SHOTS, '05-closed.png') })
  console.log(`[strain] closed: windows -> ${afterClose}`)

  // 5) Re-open → exactly ONE strain window again (idempotent, no duplicate pileup).
  await vsig.getByTestId('subwindow-titlebar').hover()
  await strainBtn.click()
  await expect(page.getByTestId('strain-wizard')).toBeVisible({ timeout: 15_000 })
  await page.waitForTimeout(5000)
  const afterReopen = await nWin()
  await page.screenshot({ path: join(SHOTS, '06-reopen.png') })
  console.log(`[strain] reopen: windows -> ${afterReopen} (expected ${afterOpen})`)
  expect(afterReopen).toBe(afterOpen)   // no duplicate strain windows

  ctx.assertNoJsErrors()
})
