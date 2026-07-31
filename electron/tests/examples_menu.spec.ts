/**
 * examples_menu.spec.ts — the Examples menu, built from em-database.
 *
 * The menu is the whole feature here, so this drives it the way a user does:
 * open Examples, walk into a technique submenu, and check each dataset row
 * carries its size, its shape where known, and a marker saying whether it is
 * already on disk. Also pins the Dummy Data submenu and the data-directory
 * entry.
 *
 * Deliberately does NOT click a real dataset — that would download gigabytes.
 */
import { test, expect } from '@playwright/test'
const { launchApp } = require('./_harness.cjs')

const SHOTS = 'examples_shots'
let ctx: Awaited<ReturnType<typeof launchApp>>

test.beforeAll(async () => {
  ctx = await launchApp({ env: { SPYDE_LOG_LEVEL: 'INFO' } })
})

test.afterAll(async () => {
  ctx?.assertNoJsErrors()
  await ctx?.app?.close()
})

test.setTimeout(120_000)

async function openExamples() {
  const { page } = ctx
  // The menu button TOGGLES, so a menu left open by the previous test would be
  // closed by this click. Escape first, then open.
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('menu-examples-items')).toBeHidden()
  await page.getByTestId('menu-examples').click()
  await expect(page.getByTestId('menu-examples-items')).toBeVisible()
}

test('the catalogue is prefetched, so the first open is already populated', async () => {
  const { page, backend } = ctx
  // MenuBar asks for the catalogue when the BACKEND REPORTS READY, not when the
  // menu opens, so the groups are already in state before the first click.
  // Without that prefetch the first open renders the empty "Loading examples…"
  // placeholder while its own request round-trips (~210 ms on a cold backend —
  // the first build imports em-database) — or, if the menu is opened before the
  // Python sidecar is up at all, forever: main/runner.ts's sendAction drops the
  // action when there is no stdin and nothing ever retries it.
  //
  // The backend's own catalogue log is the signal that the prefetch happened; no
  // menu has been opened at this point in the file, so nothing else could have
  // asked for it. On the un-prefetched build this wait is what fails.
  await backend.waitForLog('examples catalogue:', 30_000)

  // A MutationObserver rather than a poll because the placeholder is a one-frame
  // state — far too short for expect() to catch, but it cannot escape a subtree
  // observer armed BEFORE the menu opens.
  await page.evaluate(() => {
    const w = window as unknown as { _sawLoadingExamples?: boolean }
    w._sawLoadingExamples = false
    new MutationObserver(() => {
      if (document.body.innerText.includes('Loading examples')) {
        w._sawLoadingExamples = true
      }
    }).observe(document.body, { childList: true, subtree: true, characterData: true })
  })

  await openExamples()
  await expect(page.getByTestId('examples-tech-4d-stem'))
    .toBeVisible({ timeout: 30_000 })

  const sawPlaceholder = await page.evaluate(
    () => (window as unknown as { _sawLoadingExamples?: boolean })._sawLoadingExamples)
  expect(sawPlaceholder,
    'the menu rendered its empty "Loading examples…" state — the catalogue ' +
    'was not prefetched on backend-ready').toBe(false)
  await page.screenshot({ path: `${SHOTS}/00-first-open-populated.png` })
  ctx.assertNoJsErrors()
})

test('one menu open asks the backend for the catalogue exactly once', async () => {
  // `sendAction` is a new closure on every provider render, so listing it in the
  // effect's dep array re-fired the request on every unrelated context update —
  // measured at a dozen sends in ~40 ms for a single open, each spawning a
  // shape-warming thread in the backend. The log line is per SEND, so counting it
  // across one open is the regression test.
  const { backend } = ctx
  const count = () => backend.logBuffer.filter(
    (l: string) => l.includes('examples catalogue:')).length

  // SETTLE FIRST. Two things send `example_catalogue`: MenuBar's startup
  // prefetch (once, when the backend reports ready) and each menu open. This
  // test is about the OPEN, so the prefetch has to be fully landed before the
  // snapshot — otherwise its log line arrives inside the window below and the
  // assertion sees 2 for reasons that have nothing to do with re-renders.
  //
  // Waiting on the count to stop moving, rather than sleeping a fixed amount,
  // is what makes this independent of how long backend startup happens to take
  // on a given runner. (Seen in CI as a flat "Expected: 1, Received: 2" whose
  // real cause was startup timing.)
  let settled = count()
  await expect.poll(async () => {
    const now = count()
    const stable = now === settled
    settled = now
    return stable
  }, { timeout: 30_000, message: 'the catalogue prefetch never settled' }).toBe(true)

  const before = count()
  await openExamples()
  await expect(ctx.page.getByTestId('examples-tech-4d-stem')).toBeVisible()
  // A short settle so any render-triggered re-fires would have landed.
  await ctx.page.waitForTimeout(1000)
  expect(count() - before,
    'the Examples menu re-requested its catalogue on unrelated re-renders').toBe(1)
  ctx.assertNoJsErrors()
})

test('Examples groups the datasets into technique submenus', async () => {
  const { page } = ctx
  await openExamples()

  // The backend catalogue arrives async; the technique rows appear with it.
  await expect(page.getByTestId('examples-tech-4d-stem'))
    .toBeVisible({ timeout: 30_000 })
  for (const tech of ['4d-stem', 'eels', 'ebsd']) {
    await expect(page.getByTestId(`examples-tech-${tech}`),
      `no ${tech} submenu`).toBeVisible()
  }
  await page.screenshot({ path: `${SHOTS}/01-examples-techniques.png` })
  ctx.assertNoJsErrors()
})

test('each dataset shows its size, shape and download state', async () => {
  const { page } = ctx
  await openExamples()
  await page.getByTestId('examples-tech-4d-stem').hover()

  const items = page.getByTestId('examples-tech-4d-stem-items')
  await expect(items).toBeVisible({ timeout: 15_000 })

  // SPEDAg is the scan the 4D-STEM work is benchmarked on; it must be listed
  // with its size, and its shape once it has been downloaded and measured.
  const sped = page.getByTestId('example-SPEDAg')
  await expect(sped).toBeVisible()
  await expect(sped).toContainText(/\d+(\.\d+)?\s*[kMG]B/)

  // Every row carries exactly one state marker.
  const rows = await items.getByRole('button').all()
  expect(rows.length).toBeGreaterThan(3)
  let marked = 0
  for (const row of rows) {
    const text = (await row.textContent()) ?? ''
    if (text.includes('●') || text.includes('○')) marked++
  }
  expect(marked, 'dataset rows carry no downloaded/not-downloaded marker')
    .toBe(rows.length)

  await page.screenshot({ path: `${SHOTS}/02-4dstem-datasets.png` })
  ctx.assertNoJsErrors()
})

test('hovering a dataset shows a themed info card', async () => {
  const { page } = ctx
  await openExamples()
  await page.getByTestId('examples-tech-4d-stem').hover()
  await expect(page.getByTestId('examples-tech-4d-stem-items')).toBeVisible()

  await page.getByTestId('example-SPEDAg').hover()
  const card = page.getByTestId('menu-hover-card')
  await expect(card).toBeVisible({ timeout: 10_000 })
  await expect(card).toContainText('SPEDAg')
  await expect(card).toContainText('4D-STEM')
  await expect(card).toContainText(/Size/)
  // The camera is most of what tells you what to expect from a dataset.
  await expect(card).toContainText('Camera')
  await expect(card).toContainText('Merlin (Quantum Detectors)')
  await expect(card).toContainText(/On disk|Not downloaded/)
  await page.screenshot({ path: `${SHOTS}/04-hover-card.png` })

  // It is OUR panel, not the OS bubble — so the row must not also carry a
  // native title attribute racing it.
  await expect(page.getByTestId('example-SPEDAg')).not.toHaveAttribute('title', /./)

  // An undownloaded set reads differently.
  await page.getByTestId('example-FeAlStripes').hover()
  await expect(card).toContainText('Not downloaded')
  await page.screenshot({ path: `${SHOTS}/05-hover-card-undownloaded.png` })
  ctx.assertNoJsErrors()
})

test('Dummy Data is its own submenu and still loads', async () => {
  const { page } = ctx
  await openExamples()
  await page.getByTestId('examples-dummy-data').hover()
  const items = page.getByTestId('examples-dummy-data-items')
  await expect(items).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('tutorial-navigation')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/03-dummy-data.png` })

  // It is instant + no-download, so actually clicking one is fair game.
  await page.getByTestId('tutorial-navigation').click()
  await expect(page.getByTestId('subwindow').first())
    .toBeVisible({ timeout: 60_000 })
  ctx.assertNoJsErrors()
})

test('Show Example Data Directory reports a real path', async () => {
  const { page, backend } = ctx
  await openExamples()
  await expect(page.getByTestId('examples-show-dir')).toBeVisible()
  await page.getByTestId('examples-show-dir').click()
  // `open_path` is consumed by the renderer and never echoed back on the
  // PLOTAPP channel, so waiting on a message would wait forever — the backend
  // logs the reveal for exactly this reason.
  await backend.waitForLog('revealing example data directory', 20_000)
  const line = backend.logBuffer.find(
    (l: string) => l.includes('revealing example data directory'))
  expect(line, 'the backend never reported the directory').toBeTruthy()
  expect(line).toContain('em_database')

  // …and that the path actually reached the MAIN process, which is the half
  // the backend log cannot see. The harness sets SPYDE_NO_SHELL_OPEN so main
  // logs the resolved directory instead of handing it to the desktop — on a
  // headless runner xdg-open has no file manager to reach and left the app
  // unable to exit, so this test's afterAll timed out for 120s on app.close()
  // even though the assertion above had already passed.
  await backend.waitForLog('open-path (suppressed)', 20_000)
  const mainLine = backend.logBuffer.find(
    (l: string) => l.includes('open-path (suppressed)'))
  expect(mainLine, 'the main process never received the path').toContain('em_database')
  ctx.assertNoJsErrors()
})
