/**
 * actions_lifecycle.spec.ts — the formalized action lifecycle end-to-end:
 *
 *  1. requires_vectors gating — the vector actions (Strain Mapping, Vector
 *     Orientation Mapping, Vector Virtual Imaging) must NOT be on the vectors
 *     window's toolbar while the find-vectors batch is still computing, and
 *     must appear once it finalizes ("Found N diffraction vectors" re-sends
 *     the toolbar config).
 *  2. The Commit affordance — the Strain caret's Commit button
 *     (strain_commit → commit_result_tree) freezes the live field as a NEW
 *     SignalTree window.
 *  3. Controller teardown — closing the Strain caret removes the strain map
 *     and reference windows, but the committed tree stays.
 *
 * Mirrors the proven strain_zrnb.spec.ts pattern (real Dask + bundled
 * synthetic Si grains). Screenshots each stage to actions_formalize_shots/.
 *
 * Run: npx playwright test actions_lifecycle.spec.ts --project=electron
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'actions_formalize_shots')
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'INFO' } })
  const { page } = ctx
  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2, 120_000)
})

test.afterAll(async () => {
  const buf = ctx?.backend?.logBuffer ?? []
  buf.slice(-80).forEach((l: string) => console.log('[BE]', l))
  await ctx?.app?.close()
})

test.setTimeout(900_000)

test('requires_vectors gate + Commit-to-new-tree + caret teardown', async () => {
  const { page } = ctx
  const nWin = () => page.getByTestId('subwindow').count()
  const found = () =>
    (ctx.backend.logBuffer ?? []).some((l: string) => l.includes('Found'))
  await page.screenshot({ path: join(SHOTS, '01-loaded.png') })

  // ── 1) Run Find Diffraction Vectors on the source DP.
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

  // ── 2) requires_vectors: while the batch is STILL computing, NO window's
  // toolbar may offer the vector actions (page-wide — the buttons simply
  // don't exist until diffraction_vectors attach re-sends the toolbar).
  // Timing-tolerant: on a fast box the batch may already be done — then the
  // "absent" half is skipped and we still verify "present after".
  if (!found()) {
    for (const name of ['Strain Mapping', 'Vector Orientation Mapping',
                        'Vector Virtual Imaging']) {
      expect(await page.getByTestId(`action-btn-${name}`).count(),
        `${name} must be hidden until diffraction_vectors attach`).toBe(0)
    }
    await page.screenshot({ path: join(SHOTS, '02-gated.png') })
    console.log('[gate] vector actions hidden during the batch — OK')
  } else {
    console.log('[gate] batch finished before the check — absence half skipped')
  }

  // The button APPEARING is the real completion signal (the "Found …" status
  // travels the PLOTAPP protocol, not the harness log — see CLAUDE.md). The
  // distributed batch on this box takes minutes (per-worker process spawn),
  // so give it a real budget.
  await expect.poll(
    () => page.getByTestId('action-btn-Strain Mapping').count(), {
      timeout: 360_000,
      message: 'vector actions never appeared (diffraction_vectors attach)',
    }).toBeGreaterThan(0)
  await page.screenshot({ path: join(SHOTS, '03-ungated.png') })
  console.log('[gate] vector actions appeared after the attach — OK')

  // Close the FV caret so it doesn't cover the strain caret.
  if (await page.getByTestId('fv-close').count()) {
    await sig.getByTestId('subwindow-titlebar').click()
    await sig.getByTestId('subwindow-titlebar').hover()
    await page.getByTestId('fv-close').first().click()
    await page.waitForTimeout(500)
  }

  // ── 3) Open the Strain caret; wait for the live strain map window.
  const vwin = page.getByTestId('subwindow')
    .filter({ has: page.getByTestId('action-btn-Strain Mapping') }).first()
  await vwin.getByTestId('subwindow-titlebar').click()
  await vwin.getByTestId('subwindow-titlebar').hover()
  const strainBtn = vwin.getByTestId('action-btn-Strain Mapping')
  const beforeStrain = await nWin()
  await strainBtn.click()
  await expect(page.getByTestId('strain-wizard')).toBeVisible({ timeout: 15_000 })
  await expect.poll(nWin, {
    timeout: 300_000, message: 'strain map window never opened',
  }).toBeGreaterThan(beforeStrain)
  await page.waitForTimeout(2000)
  const withStrain = await nWin()
  await page.screenshot({ path: join(SHOTS, '04-strain-live.png') })

  // ── 4) Commit → a NEW SignalTree window ("Strain") appears; the live
  // window stays open (commit is non-destructive).
  await expect(page.getByTestId('strain-commit')).toBeVisible()
  await page.getByTestId('strain-commit').click()
  await expect.poll(nWin, {
    timeout: 60_000, message: 'committed strain tree window never opened',
  }).toBeGreaterThan(withStrain)
  await page.waitForTimeout(1500)
  await page.screenshot({ path: join(SHOTS, '05-committed.png') })
  const committed = page.getByTestId('subwindow').filter({ hasText: 'Strain' })
  expect(await committed.count(), 'a committed Strain window exists')
    .toBeGreaterThan(0)
  const afterCommit = await nWin()
  console.log(`[commit] windows ${withStrain} -> ${afterCommit}`)

  // ── 5) Close the caret: the live strain map + reference windows tear down
  // (controller close), the COMMITTED tree window survives.
  await vwin.getByTestId('subwindow-titlebar').click()
  await vwin.getByTestId('subwindow-titlebar').hover()
  await strainBtn.click()
  await expect.poll(nWin, {
    timeout: 15_000, message: 'strain live windows not removed on toggle off',
  }).toBe(beforeStrain + (afterCommit - withStrain))
  await page.screenshot({ path: join(SHOTS, '06-caret-closed.png') })
  expect(await page.getByTestId('subwindow').filter({ hasText: 'Strain' }).count(),
    'the committed Strain tree must survive the caret close').toBeGreaterThan(0)

  ctx.assertNoJsErrors()
})
