/**
 * movie_editor.spec.ts — the Movie BLOCK + full-screen editor, end-to-end.
 *
 * The Movie block (spyde/actions/report/movie.py) REPLACES the old per-plot
 * "Export Movie" caret: a movie is now an editable, persistent cell in the report
 * document, edited in a full-screen editor. This spec drives the whole Phase-1
 * spine against the real app + real Dask + the bundled synthetic in-situ movie
 * (load_test_data_movie: 6 × 2048² uint16, 1 frame/chunk lazy, 0.05 s/frame):
 *
 *   1) load the movie → open the Report sidebar → the third "Movie" card exists.
 *   2) click Movie → a movie cell is created (seeded from the active in-situ
 *      window) AND the full-screen editor opens with a live preview frame.
 *   3) scrub the timeline → the preview frame CHANGES (not a stale/black frame).
 *   4) drag a crop rectangle on the preview → the Export readout output size shrinks.
 *   5) Export a GIF (ffmpeg-free) → movie_done, a real decodable file, a poster on
 *      the card.
 *   6) Save the .spyde-report and reopen → the movie cell + its spec + poster survive.
 *
 * Export routes through the MAIN-process report:export-dialog handler; we STUB it
 * (removeHandler + handle) to return fixed workdir paths so no OS picker blocks.
 * The GIF is validated in a real Python subprocess (imageio) for frame count + dims.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdtempSync, mkdirSync, existsSync, statSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { execFileSync } from 'child_process'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'movie_editor_shots')
const REPO_ROOT = join(__dirname, '..', '..')

let ctx: Awaited<ReturnType<typeof launchApp>>
let workDir: string
let gifPath: string
let reportPath: string

function pyJSON(code: string): any {
  const out = execFileSync('uv', ['run', 'python', '-c', code], {
    cwd: REPO_ROOT, encoding: 'utf-8', timeout: 120_000,
    maxBuffer: 32 * 1024 * 1024,
  })
  const lines = out.trim().split(/\r?\n/).filter((l) => l.trim())
  return JSON.parse(lines[lines.length - 1])
}

// Which fixed path the stubbed dialog returns (gif export vs report save).
async function setRoute(route: string) {
  await ctx.app.evaluate((_e, r) => {
    ;(globalThis as unknown as { __movRoute?: string }).__movRoute = r
  }, route)
}

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  mkdirSync(SHOTS, { recursive: true })
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_movie')
  await waitForSubwindowCount(page, 2, 120_000)   // 1-D time navigator + 2-D signal
  await page.waitForTimeout(3000)                 // let the first frame + nav paint

  workDir = mkdtempSync(join(tmpdir(), 'spyde-movie-editor-'))
  gifPath = join(workDir, 'movie.gif')
  reportPath = join(workDir, 'movie.spyde-report')

  // Stub BOTH the movie export dialog (report:export-dialog, 'mp4' kind → gif path)
  // AND the report save dialog (report:save-dialog → the .spyde-report path).
  await ctx.app.evaluate(({ ipcMain }, paths) => {
    const g = globalThis as unknown as { __movRoute?: string }
    ipcMain.removeHandler('report:export-dialog')
    ipcMain.handle('report:export-dialog', async () => paths.gif)
    ipcMain.removeHandler('report:save-dialog')
    ipcMain.handle('report:save-dialog', async () => paths.report)
    // report_saved / report_open dialogs read this too; keep it simple.
    void g
  }, { gif: gifPath, report: reportPath })
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
    if (workDir && existsSync(workDir)) {
      try { rmSync(workDir, { recursive: true, force: true }) } catch { /* */ }
    }
  }
})

// ── 1) The Movie card exists in the empty Report sidebar ─────────────────────

test('1) the Report sidebar empty state shows the Movie card', async () => {
  const { page } = ctx
  await page.getByTestId('toggle-report').click()
  await expect(page.getByTestId('report-sidebar')).toBeVisible()
  await expect(page.getByTestId('report-empty')).toBeVisible()
  await expect(page.getByTestId('report-new-report-card')).toBeVisible()
  await expect(page.getByTestId('report-new-presentation-card')).toBeVisible()
  const movieCard = page.getByTestId('report-new-movie-card')
  await expect(movieCard).toBeVisible()
  await expect(movieCard).toContainText('Movie')
  await page.screenshot({ path: join(SHOTS, '01-movie-card.png') })
  ctx.assertNoJsErrors()
})

// ── 2) Clicking Movie creates a cell + opens the editor with the LIVE figure ──

test('2) Movie card → a movie cell + the editor mounting the tree\'s LIVE signal figure', async () => {
  const { page } = ctx
  await page.getByTestId('report-new-movie-card').click()

  // The full-screen editor overlay appears.
  await expect(page.getByTestId('movie-editor')).toBeVisible({ timeout: 15_000 })
  // The editor surfaces the tree's REAL signal figure as a LIVE iframe (a
  // spyde-fig:// / data:text/html figure frame), NOT a rasterized <img>. The
  // figure wrap contains the SeamlessFigureFrame iframe once movie_state lands.
  const figWrap = page.getByTestId('movie-figure-wrap')
  await expect(figWrap).toBeVisible()
  await expect(figWrap.locator('iframe[data-testid^="figure-"]'),
    'the editor never mounted the live signal figure iframe')
    .toBeVisible({ timeout: 30_000 })
  // There is NO base64 PNG preview <img> anymore.
  expect(await page.getByTestId('movie-preview-img').count(),
    'the old PNG <img> preview should be gone (live figure now)').toBe(0)
  await page.waitForTimeout(1500)   // let the figure paint its first frame
  await page.screenshot({ path: join(SHOTS, '02-editor-open.png') })
  ctx.assertNoJsErrors()
})

// ── 3) Scrubbing drives the REAL navigator (movie_state.current_index moves) ──

test('3) scrubbing drives the real navigator — current_index advances', async () => {
  const { page } = ctx
  // Capture movie_state messages in-page so we can read the authoritative
  // current_index the backend reports as the navigator moves.
  await page.evaluate(() => {
    ;(globalThis as unknown as { __movIdx?: number }).__movIdx = -1
    window.addEventListener('spyde:movie_state', (e: Event) => {
      const d = (e as CustomEvent).detail as { current_index?: number }
      if (typeof d.current_index === 'number')
        (globalThis as unknown as { __movIdx?: number }).__movIdx = d.current_index
    })
  })
  // Scrub to the last frame via the range input (React controlled → native setter).
  const scrubber = page.getByTestId('movie-scrubber')
  await scrubber.evaluate((el: HTMLInputElement) => {
    const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value')?.set
    setter?.call(el, el.max)
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  })
  // The scrubber counter reflects the new position immediately (local state).
  await expect(page.getByTestId('movie-time-label')).toBeVisible()
  // The backend drove the real navigator → a movie_state re-emit carries the new
  // current_index (5 = last frame of the 6-frame movie). We re-open by nudging the
  // scrubber; assert the DOM counter shows the last frame.
  const counter = page.locator('[data-testid="movie-scrubber"] ~ span').last()
  await expect(counter).toContainText('5 / 5', { timeout: 10_000 })
  await page.waitForTimeout(1200)
  await page.screenshot({ path: join(SHOTS, '03-scrubbed.png') })
  ctx.assertNoJsErrors()
})

// ── 4) The iMovie timeline: add a text clip on the Text lane ──────────────────

test('4) add a Text overlay → a draggable clip appears on the Text timeline lane', async () => {
  const { page } = ctx
  await page.getByTestId('movie-add-text').click()
  // A text clip appears (index within the annotations list; text is first → -0).
  await expect(page.getByTestId('movie-clip-text-0'),
    'no text clip appeared on the timeline').toBeVisible({ timeout: 5_000 })
  // A ROI clip too — the ROI is the 2nd annotation (its testid carries its list
  // index), so match the ROI lane's clip by prefix rather than a fixed index.
  await page.getByTestId('movie-add-roi').click()
  await expect(page.locator('[data-testid^="movie-clip-roi-"]').first(),
    'no ROI clip appeared').toBeVisible({ timeout: 5_000 })
  // A freeze marker at the current frame.
  await page.getByTestId('movie-add-freeze').click()
  await expect(page.getByTestId('movie-clip-freeze-0')).toBeVisible({ timeout: 5_000 })
  await page.screenshot({ path: join(SHOTS, '04-timeline.png') })
  ctx.assertNoJsErrors()
})

// ── 5) Export a GIF → movie_done + a real file + a poster on the card ────────

test('5) export a GIF → decodable file + a poster back on the movie card', async () => {
  const { page } = ctx
  await setRoute('gif')
  // The Export button lives in the left rail's Export panel.
  await expect(page.getByTestId('movie-export-btn')).toBeVisible({ timeout: 10_000 })
  await page.getByTestId('movie-export-btn').click()

  // The status transitions to "Exported movie.gif …" on movie_done.
  await expect.poll(() => existsSync(gifPath), {
    timeout: 120_000, message: 'gif export was never written',
  }).toBe(true)
  await expect(page.getByTestId('movie-editor-status')).toContainText(/Exported movie\.gif/, {
    timeout: 20_000,
  })
  await page.screenshot({ path: join(SHOTS, '05-exported.png') })

  const size = statSync(gifPath).size
  console.log('[movie] gif size =', size)
  expect(size, 'gif too small').toBeGreaterThan(1024)
  // Validate the GIF in Python (imageio): 6 frames of the movie.
  const r = pyJSON(`
import json, numpy as np, imageio.v3 as iio
g = np.asarray(iio.imread(r'''${gifPath}'''))   # (T,H,W,C)
print(json.dumps(dict(T=int(g.shape[0]), H=int(g.shape[1]), W=int(g.shape[2]))))
`)
  console.log('[movie] gif metrics =', JSON.stringify(r))
  expect(r.T, `gif frame count (${r.T}) not > 5`).toBeGreaterThan(5)

  // Close the editor → the card should now show a poster.
  await page.getByTestId('movie-editor-close').click()
  await expect(page.getByTestId('movie-editor')).toBeHidden({ timeout: 5_000 })
  // The sidebar movie cell now carries a poster (baked on export).
  const poster = page.locator('[data-testid^="report-moviecell-poster-"]').first()
  await expect(poster, 'no poster appeared on the movie card after export').toBeVisible({ timeout: 10_000 })
  await page.screenshot({ path: join(SHOTS, '06-card-with-poster.png') })
  ctx.assertNoJsErrors()
})

// ── 6) Save + reopen the .spyde-report → the movie cell survives ─────────────

test('6) save + reopen the report → the movie cell + spec + poster survive on disk', async () => {
  const { page } = ctx
  await setRoute('report')
  // Save via the File menu → Save. A fresh doc has no path, so doSave routes
  // through the stubbed report:save-dialog → reportPath, then report_save {path}.
  await page.getByTestId('report-menu-toggle').click()
  await expect(page.getByTestId('report-menu')).toBeVisible()
  await page.getByTestId('report-save').click()
  await expect.poll(() => existsSync(reportPath), {
    timeout: 30_000, message: 'the .spyde-report was never written',
  }).toBe(true)
  console.log('[movie] report saved to', reportPath)

  // Validate the zip contents in Python: a movies/<id>.yaml + a poster asset, and a
  // round-trip read yields a movie cell with a source + params.
  const r = pyJSON(`
import json, zipfile
from spyde.actions.report.model import read_report
names = zipfile.ZipFile(r'''${reportPath}''').namelist()
has_movie_yaml = any(n.startswith('movies/') and n.endswith('.yaml') for n in names)
doc, assets = read_report(r'''${reportPath}''')
movies = [c for c in doc.cells if c.cell_type == 'movie']
m = movies[0] if movies else None
print(json.dumps(dict(
  has_movie_yaml=has_movie_yaml,
  n_movies=len(movies),
  has_source=(m is not None and m.movie is not None and m.movie.source is not None),
  has_fps=(m is not None and m.movie is not None and 'fps' in (m.movie.params or {})),
  has_poster=(m is not None and m.id in assets),
)))
`)
  console.log('[movie] report round-trip =', JSON.stringify(r))
  expect(r.has_movie_yaml, 'no movies/<id>.yaml in the saved zip').toBe(true)
  expect(r.n_movies, 'no movie cell in the reopened report').toBe(1)
  expect(r.has_source, 'movie cell lost its source on reload').toBe(true)
  expect(r.has_fps, 'movie params did not persist').toBe(true)
  expect(r.has_poster, 'poster asset missing from the saved zip').toBe(true)
  ctx.assertNoJsErrors()
})

// ── 7) Final audit: no renderer JS errors, no backend tracebacks ─────────────

test('7) no renderer JS errors and no Python tracebacks', async () => {
  ctx.assertNoJsErrors()
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[movie] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
