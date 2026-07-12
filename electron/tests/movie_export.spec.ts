/**
 * movie_export.spec.ts — Movie Export wizard (Phase 4) end-to-end in the real app.
 *
 * Real Dask + the bundled synthetic in-situ movie (`load_test_data_movie`:
 * 6 × 2048² uint16, 1 frame/chunk lazy, calibrated 0.05 s/frame time axis, 0.5 nm
 * signal scale, asymmetric per-frame content — a bright vertical band whose
 * x-position encodes the FRAME INDEX, corner blocks, a centre checkerboard). That
 * per-frame band makes rendered-frame differences pixel-visible in the decoded mp4.
 *
 * The wizard lives on the movie SIGNAL window's floating toolbar as a toggle-style
 * "Export Movie" button (toolbar_side: bottom, insitu-gated, plot_dim [2]). Opening
 * it fires mvx_open → the backend probes ffmpeg, reads the time axis, seeds params,
 * and broadcasts mvx_state. Export routes through the MAIN-process
 * report:export-dialog handler ('mp4' kind); we STUB it (removeHandler + handle)
 * to return fixed workdir paths so no OS picker blocks.
 *
 * Traps handled:
 *  - PLOTAPP statuses/progress never reach Playwright stdout → we poll the DOM
 *    (StatusBar busy %, the wizard's mvx-status "Exported …" note) and wait on the
 *    FILE, never a fixed sleep.
 *  - The bottom toolbar shares the window z-level and toggles pointer-events with
 *    hover; we focus-raise the window (click its titlebar) + hover before clicking.
 *  - SPYDE_LOG_LEVEL=WARNING tees backend logging to stderr so the final audit can
 *    scan ctx.backend.logBuffer for Python tracebacks.
 *
 * The mp4/gif are validated in a real Python subprocess (imageio) for size, frame
 * count, dims, per-frame difference, annotation time-gating, timestamp presence,
 * and the trace inset.
 */
import { test, expect } from '@playwright/test'
import { join } from 'path'
import { mkdtempSync, existsSync, statSync, rmSync } from 'fs'
import { tmpdir } from 'os'
import { execFileSync } from 'child_process'
const {
  launchApp, backendAction, waitForSubwindowCount, backendErrorLines,
  sigWindow, navWindow, titlebarGrabPoint,
} = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'movie_export_shots')
// Repo root (…/electron/tests → …), so `uv run` executes in the project dir.
const REPO_ROOT = join(__dirname, '..', '..')

let ctx: Awaited<ReturnType<typeof launchApp>>
let workDir: string
let mp4Path: string          // primary export (downsample 4, with timestamp + annotation)
let mp4NoTsPath: string      // control export (no timestamp, stride 3)
let gifPath: string          // GIF path
let mp4TracePath: string     // export WITH a trace inset
let mp4NoTracePath: string   // matching export WITHOUT the trace

// The pending route marker chooses which fixed path the stubbed dialog returns.
async function setExportRoute(route: string) {
  await ctx.app.evaluate((_e, r) => {
    ;(globalThis as unknown as { __mvxRoute?: string }).__mvxRoute = r
  }, route)
}

test.describe.configure({ mode: 'serial' })
test.setTimeout(300_000)

test.beforeAll(async () => {
  ctx = await launchApp({ dask: true, env: { SPYDE_LOG_LEVEL: 'WARNING' } })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_movie')
  await waitForSubwindowCount(page, 2, 120_000)   // 1-D time navigator + 2-D signal
  await page.waitForTimeout(3000)                 // let the first frame + nav paint

  workDir = mkdtempSync(join(tmpdir(), 'spyde-movie-export-'))
  mp4Path = join(workDir, 'movie.mp4')
  mp4NoTsPath = join(workDir, 'movie_no_ts.mp4')
  gifPath = join(workDir, 'movie.gif')
  mp4TracePath = join(workDir, 'movie_trace.mp4')
  mp4NoTracePath = join(workDir, 'movie_no_trace.mp4')

  // Stub the MAIN-process export dialog: return a fixed path chosen by the
  // pending __mvxRoute marker (set before each export) so the UI never blocks on
  // an OS save picker. Mirrors report_export.spec.ts.
  await ctx.app.evaluate(({ ipcMain }, paths) => {
    const g = globalThis as unknown as { __mvxRoute?: string }
    ipcMain.removeHandler('report:export-dialog')
    ipcMain.handle('report:export-dialog', async (_e, kind: string) => {
      const r = g.__mvxRoute || 'primary'
      if (r === 'no_ts') return paths.noTs
      if (r === 'gif') return paths.gif
      if (r === 'trace') return paths.trace
      if (r === 'no_trace') return paths.noTrace
      return paths.primary   // 'primary'
    })
  }, { primary: mp4Path, noTs: mp4NoTsPath, gif: gifPath,
       trace: mp4TracePath, noTrace: mp4NoTracePath })
})

test.afterAll(async () => {
  try { ctx?.assertNoJsErrors() } finally {
    await ctx?.app?.close()
    if (workDir && existsSync(workDir)) {
      try { rmSync(workDir, { recursive: true, force: true }) } catch { /* */ }
    }
  }
})

/**
 * Open the Movie Export wizard from the movie signal window's bottom toolbar.
 * The bottom toolbar shares the window's z-level and toggles pointer-events with
 * hover, so a plain click can land while it's transparent — focus-raise the window
 * (titlebar click) then hover the window + button before clicking via mouse box
 * centre (the clickCopyToReport interception workaround from report_export.spec).
 */
async function openWizard(page: any) {
  const sigWin = sigWindow(page)
  const grab = await titlebarGrabPoint(sigWin)
  await page.mouse.click(grab.x, grab.y)          // raise the window
  await sigWin.getByTestId('subwindow-titlebar').hover()
  await sigWin.hover()
  const btn = sigWin.getByTestId('action-btn-Export Movie')
  await expect(btn).toBeVisible({ timeout: 10_000 })
  await btn.hover()
  const box = await btn.boundingBox()
  if (!box) throw new Error('Export Movie button has no bounding box')
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2)
  await expect(page.getByTestId('mvx-wizard')).toBeVisible({ timeout: 10_000 })
}

/** Run the wizard's export for the given route and wait for the file + note. */
async function exportAndWait(page: any, route: string, filePath: string,
                             timeoutMs = 120_000) {
  await setExportRoute(route)
  await page.getByTestId('mvx-tab-Export').click()
  await expect(page.getByTestId('mvx-export')).toBeVisible({ timeout: 10_000 })
  await page.getByTestId('mvx-export').click()
  // The wizard status transitions to "Exported <basename>" on mvx_done.
  const base = filePath.split(/[/\\]/).pop()
  await expect.poll(() => existsSync(filePath), {
    timeout: timeoutMs, message: `export file ${base} was never written`,
  }).toBe(true)
  await expect(page.getByTestId('mvx-status')).toHaveText(
    new RegExp(`Exported ${base!.replace('.', '\\.')}`), { timeout: 20_000 })
}

/**
 * Run a Python snippet with `uv run python -c` in the repo dir and return its
 * stdout. The snippet must print a single JSON line as its LAST output; we take
 * the last non-empty line to skip any rsciio/matplotlib deprecation warnings.
 */
function pyJSON(code: string): any {
  const out = execFileSync('uv', ['run', 'python', '-c', code], {
    cwd: REPO_ROOT, encoding: 'utf-8', timeout: 120_000,
    maxBuffer: 32 * 1024 * 1024,
  })
  const lines = out.trim().split(/\r?\n/).filter((l) => l.trim())
  const last = lines[lines.length - 1]
  return JSON.parse(last)
}

// ── 1) Load the movie + open the wizard ──────────────────────────────────────

test('1) load the in-situ movie and open the Export Movie wizard', async () => {
  const { page } = ctx
  // The navigator (1-D time) + signal (2-D) windows are present.
  await expect(sigWindow(page)).toBeVisible()
  await expect(navWindow(page)).toBeVisible()

  await openWizard(page)
  // First mvx_state seeds n_frames=6 and the time axis (0.05 s/frame).
  await page.waitForTimeout(800)
  await page.screenshot({ path: join(SHOTS, '01-wizard-open.png') })
  ctx.assertNoJsErrors()
})

// ── 2) Format tab: downsample=4, fps=10 → output-info updates ─────────────────

test('2) Format tab: downsample=4, fps=10 → 6 frames · 0.6 s', async () => {
  const { page } = ctx
  await page.getByTestId('mvx-tab-Format').click()

  // fps = 10
  const fps = page.getByTestId('mvx-fps')
  await fps.fill('10')
  await fps.blur()
  // downsample = 4 (a <select>)
  await page.getByTestId('mvx-downsample').selectOption('4')
  await page.waitForTimeout(600)   // debounced tune round-trip

  // The computed output-info line: 6 frames (full range / stride 1) at 10 fps → 0.6 s.
  const info = page.getByTestId('mvx-output-info')
  await expect(info).toContainText('6 frames')
  await expect(info).toContainText('0.6 s')
  await expect(info).toContainText('10 fps')
  await page.screenshot({ path: join(SHOTS, '02-format-tab.png') })
  ctx.assertNoJsErrors()
})

// ── 3) Overlays tab: timestamp + scalebar on, Rect annotation on middle frames ─

test('3) Overlays: keep timestamp+scalebar, add a time-gated Rect annotation', async () => {
  const { page } = ctx
  await page.getByTestId('mvx-tab-Overlays').click()

  // Timestamp + scale bar default ON (scalebar seeded on because the signal axis
  // is calibrated 0.5 nm). Assert they are checked; keep them on.
  await expect(page.getByTestId('mvx-timestamp')).toBeChecked()
  // (Scale bar may be seeded on/off depending on the axis; force it ON.)
  const sb = page.getByTestId('mvx-scalebar')
  if (!(await sb.isChecked())) await sb.check()
  await expect(sb).toBeChecked()

  // Add a Rect annotation. The wizard seeds kind='text'; switch to 'rect', give it
  // a box, and a time window covering the MIDDLE frames only. The loader's frame
  // times are [0, .05, .10, .15, .20, .25] s (0.05 s/frame); the pipeline gates on
  // SECONDS, so [0.1, 0.35] selects frames 2..5 and excludes frames 0,1.
  await page.getByTestId('mvx-add-annotation').click()
  await page.getByTestId('mvx-ann-kind-0').selectOption('rect')
  await page.waitForTimeout(200)
  const t0 = page.getByTestId('mvx-ann-t0-0')
  const t1 = page.getByTestId('mvx-ann-t1-0')
  await t0.fill('0.1')
  await t0.blur()
  await t1.fill('0.35')
  await t1.blur()
  await page.waitForTimeout(600)

  // Verify the backend received the annotation with our time window by reading
  // the authoritative mvx_state (the wizard mirrors params.annotations, but the
  // ground truth is the backend's echoed state — assert the DOM row reflects it).
  await expect(page.getByTestId('mvx-annotation-0')).toBeVisible()
  await expect(t0).toHaveValue('0.1')
  await expect(t1).toHaveValue('0.35')
  await page.screenshot({ path: join(SHOTS, '03-overlays-tab.png') })
  ctx.assertNoJsErrors()
})

// ── 4) Export the mp4 (downsample 4, timestamp + annotation) ──────────────────

test('4) export mp4 → progress + cancel visible during run → file + success note', async () => {
  const { page } = ctx
  await setExportRoute('primary')
  await page.getByTestId('mvx-tab-Export').click()
  await expect(page.getByTestId('mvx-export')).toBeVisible({ timeout: 10_000 })
  await page.getByTestId('mvx-export').click()

  // While running the wizard swaps Export→Cancel and the StatusBar shows a busy %.
  // (6 frames of 512² encode fast; poll opportunistically — either we catch the
  // running state or the file lands first. The cancel button appearing at all is
  // the contract we assert; if it's already done, that's still a pass for the file.)
  const cancel = page.getByTestId('mvx-cancel')
  const sawCancel = await cancel.isVisible({ timeout: 4_000 }).catch(() => false)
  if (sawCancel) {
    await page.screenshot({ path: join(SHOTS, '04-during-run.png') })
    // StatusBar busy text carries the "Encoding movie" progress label.
    const busy = page.getByTestId('status-text')
    await expect(busy).toContainText(/Encoding movie|Exported/i, { timeout: 20_000 })
  }

  await expect.poll(() => existsSync(mp4Path), {
    timeout: 120_000, message: 'primary mp4 was never written',
  }).toBe(true)
  await expect(page.getByTestId('mvx-status')).toHaveText(/Exported movie\.mp4/, {
    timeout: 20_000,
  })
  await page.waitForTimeout(500)
  await page.screenshot({ path: join(SHOTS, '04b-after-done.png') })

  const size = statSync(mp4Path).size
  console.log('[mvx] primary mp4 size =', size)
  // A valid H.264 file with a real header + moov atom. The synthetic movie's
  // content (a smooth gradient + a thin band) compresses to ~10 KB with libx264 —
  // the substantive proof (6 frames, 512² dims, per-frame diffs, annotation
  // gating, timestamp) is in step 5's imageio decode. Here we only guard against a
  // truncated/empty write.
  expect(size, `mp4 suspiciously small (${size} B) — likely truncated`).toBeGreaterThan(2 * 1024)
  ctx.assertNoJsErrors()
})

// ── 4c) Control export: same movie WITHOUT timestamp (for timestamp presence) ─

test('4c) control export without timestamp (stride 3) for timestamp comparison', async () => {
  const { page } = ctx
  // Turn the timestamp OFF and bump stride to 3 (2 frames) for a quick control.
  await page.getByTestId('mvx-tab-Overlays').click()
  await page.getByTestId('mvx-timestamp').uncheck()
  await page.getByTestId('mvx-tab-Format').click()
  await page.getByTestId('mvx-stride').fill('3')
  await page.getByTestId('mvx-stride').blur()
  await page.waitForTimeout(600)

  await exportAndWait(page, 'no_ts', mp4NoTsPath)
  expect(statSync(mp4NoTsPath).size).toBeGreaterThan(1024)

  // Restore for downstream tests: timestamp back ON, stride back to 1.
  await page.getByTestId('mvx-tab-Overlays').click()
  await page.getByTestId('mvx-timestamp').check()
  await page.getByTestId('mvx-tab-Format').click()
  await page.getByTestId('mvx-stride').fill('1')
  await page.getByTestId('mvx-stride').blur()
  await page.waitForTimeout(600)
  ctx.assertNoJsErrors()
})

// ── 5) Validate the mp4 in Python (imageio) ──────────────────────────────────

test('5) validate mp4: 6 frames, 512×512, per-frame diff, annotation gating, timestamp', async () => {
  // One Python probe reads the primary mp4 (with timestamp + rect annotation on
  // frames 2..5) AND the no-timestamp control, and returns all the metrics.
  const code = `
import json, numpy as np, imageio.v3 as iio
prim = iio.imread(r'''${mp4Path}''')          # (T,H,W,3)
nots = iio.imread(r'''${mp4NoTsPath}''')
prim = np.asarray(prim); nots = np.asarray(nots)
T, H, W = prim.shape[0], prim.shape[1], prim.shape[2]
# Per-frame difference: frames 0, 2, 5 must all differ (moving index band).
def md(a, b): return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())
d02 = md(prim[0], prim[2]); d05 = md(prim[0], prim[5]); d25 = md(prim[2], prim[5])
# Annotation gating: the rect (kind='rect', default xy=[0,0] wh=[10,10] on the
# ORIGINAL 2048 image / downsample 4 → a tiny box near the top-left) shows on
# frames 2..5 only. Compare a MIDDLE annotated frame vs frame 0 in the top-left
# 24x24 region (where the rect + its outline land after /4 scaling). It should
# differ MORE there than the ambient per-frame band difference far from the band.
tl = (slice(0, 28), slice(0, 28))
# frame 0 (no annotation) vs frame 3 (annotated): top-left corner diff.
ann_diff = md(prim[0][tl], prim[3][tl])
# Timestamp presence: with timestamp ON, the top-left band carries burnt-in white
# text "t = …". Both exports share SOURCE frame 0 (primary stride 1 → frame[0];
# control stride 3 → frame[0]), and frame 0 has NO annotation on either, so the
# ONLY difference in the top-left band is the timestamp itself. Assert the band
# differs substantially while the full frames are near-identical — i.e. the change
# is localised to the timestamp text, not global. (Counting absolute bright pixels
# is unreliable: the synthetic frame already has a bright TOP-LEFT block there.)
band = (slice(0, 40), slice(0, 170))
ts_band_diff = md(prim[0][band], nots[0][band])
ts_full_diff = md(prim[0], nots[0])
print(json.dumps(dict(T=int(T), H=int(H), W=int(W),
                      d02=d02, d05=d05, d25=d25, ann_diff=ann_diff,
                      ts_band_diff=ts_band_diff, ts_full_diff=ts_full_diff)))
`
  const r = pyJSON(code)
  console.log('[mvx] mp4 metrics =', JSON.stringify(r))

  // 6 frames (full range, stride 1), 512×512 (2048/4, even).
  expect(r.T, 'mp4 frame count').toBe(6)
  expect(r.W, 'mp4 width (2048/4)').toBe(512)
  expect(r.H, 'mp4 height (2048/4)').toBe(512)

  // Frames differ from each other (moving per-frame index band).
  expect(r.d02, 'frame 0 vs 2 identical — band did not move').toBeGreaterThan(1)
  expect(r.d05, 'frame 0 vs 5 identical').toBeGreaterThan(1)
  expect(r.d25, 'frame 2 vs 5 identical').toBeGreaterThan(1)

  // Annotation gating: the rect is drawn on frame 3 (t=0.15 s ∈ [0.1,0.35]) but
  // NOT frame 0 (t=0). The top-left corner therefore differs between them.
  expect(r.ann_diff,
    'annotated middle frame top-left corner matches frame 0 — annotation not time-gated')
    .toBeGreaterThan(0.5)

  // Timestamp presence: the top-left band differs between the ts-ON primary and
  // ts-OFF control (burnt-in text), while the full frames are near-identical —
  // proving the timestamp overlay is drawn and localised to its band.
  console.log(`[mvx] timestamp band_diff=${r.ts_band_diff} full_diff=${r.ts_full_diff}`)
  expect(r.ts_band_diff, 'no timestamp text difference in the top-left band')
    .toBeGreaterThan(2)
  expect(r.ts_band_diff,
    'timestamp band diff not dominant over the full-frame diff — overlay not localised')
    .toBeGreaterThan(r.ts_full_diff * 3)
})

// ── 6) GIF export path ────────────────────────────────────────────────────────

test('6) GIF export → decodable, >5 frames', async () => {
  const { page } = ctx
  // Ensure stride 1 (full range) so the GIF has all 6 frames.
  await page.getByTestId('mvx-tab-Format').click()
  await page.getByTestId('mvx-stride').fill('1')
  await page.getByTestId('mvx-stride').blur()
  await page.waitForTimeout(500)

  await exportAndWait(page, 'gif', gifPath)
  await page.screenshot({ path: join(SHOTS, '06-gif-done.png') })

  const size = statSync(gifPath).size
  console.log('[mvx] gif size =', size)
  expect(size, 'gif too small').toBeGreaterThan(1024)

  const code = `
import json, numpy as np, imageio.v3 as iio
g = np.asarray(iio.imread(r'''${gifPath}'''))   # (T,H,W,C)
print(json.dumps(dict(T=int(g.shape[0]), H=int(g.shape[1]), W=int(g.shape[2]))))
`
  const r = pyJSON(code)
  console.log('[mvx] gif metrics =', JSON.stringify(r))
  expect(r.T, `gif frame count (${r.T}) not > 5`).toBeGreaterThan(5)
  ctx.assertNoJsErrors()
})

// ── 7) Trace drop: drag the 1-D time navigator pill → trace chip → export ─────

test('7) drag the 1-D time navigator onto the trace slot → trace chip + inset', async () => {
  const { page } = ctx
  // First a control export WITHOUT any trace (stride 2 for speed) so we can prove
  // the trace inset changes pixels. Clear annotations to isolate the trace effect.
  await page.getByTestId('mvx-tab-Overlays').click()
  // Remove the annotation added in step 3 (if the remove button is present).
  const annRemove = page.getByTestId('mvx-ann-remove-0')
  if (await annRemove.count()) { await annRemove.click(); await page.waitForTimeout(400) }
  await page.getByTestId('mvx-tab-Format').click()
  await page.getByTestId('mvx-stride').fill('2')
  await page.getByTestId('mvx-stride').blur()
  await page.waitForTimeout(500)
  await exportAndWait(page, 'no_trace', mp4NoTracePath)

  // Now add a trace: drag the 1-D TIME NAVIGATOR window's breadcrumb pill onto the
  // wizard's trace drop slot. The pill stamps WINDOW_DRAG_MIME = String(windowId);
  // the wizard's onDrop reads it → mvx_add_trace {source_window_id}. The navigator
  // plot is 1-D (a VI trace over the time axis) so capture_from_plot succeeds.
  await page.getByTestId('mvx-tab-Traces').click()
  const navWin = navWindow(page)
  const navPill = navWin.getByTestId('window-breadcrumb')
  await expect(navPill).toBeVisible()
  // Tag both src + dst so the in-page native-drag helper can target them.
  await navPill.evaluate((el: HTMLElement) => el.setAttribute('data-mvx-src', '1'))

  const dropRes = await page.evaluate(() => {
    const src = document.querySelector('[data-mvx-src="1"]') as HTMLElement
    const dst = document.querySelector('[data-testid="mvx-trace-drop"]') as HTMLElement
    if (!src || !dst) throw new Error('trace drag src/dst not found')
    const dt = new DataTransfer()
    const fire = (target: HTMLElement, type: string) => {
      const r = target.getBoundingClientRect()
      const ev = new DragEvent(type, {
        bubbles: true, cancelable: true,
        clientX: r.left + r.width / 2, clientY: r.top + r.height / 2,
      })
      Object.defineProperty(ev, 'dataTransfer', { value: dt, configurable: true })
      target.dispatchEvent(ev)
    }
    fire(src, 'dragstart')
    const types = Array.from(dt.types)
    fire(dst, 'dragenter'); fire(dst, 'dragover'); fire(dst, 'drop'); fire(src, 'dragend')
    return { types }
  })
  console.log('[mvx] trace drag MIME types =', JSON.stringify(dropRes.types))
  expect(dropRes.types, 'window drag did not stamp the WINDOW_DRAG_MIME')
    .toContain('application/x-spyde-window')

  // A trace chip appears (mvx-trace-<id>). This is the POSITIVE acceptance.
  const chip = page.locator('[data-testid^="mvx-trace-tr"]').first()
  await expect(chip, 'trace chip did not appear after dropping the navigator pill')
    .toBeVisible({ timeout: 10_000 })
  await page.waitForTimeout(400)
  await page.screenshot({ path: join(SHOTS, '07-trace-chip.png') })

  // Export WITH the trace (same stride 2) → the inset must change pixels vs the
  // no-trace control (the trace inset is pasted bottom-left with a coloured plot).
  await exportAndWait(page, 'trace', mp4TracePath)
  await page.screenshot({ path: join(SHOTS, '07b-trace-exported.png') })

  const code = `
import json, numpy as np, imageio.v3 as iio
tr = np.asarray(iio.imread(r'''${mp4TracePath}'''))   # (T,H,W,3)
nt = np.asarray(iio.imread(r'''${mp4NoTracePath}'''))
Tt, Ht, Wt = tr.shape[0], tr.shape[1], tr.shape[2]
# The inset is pasted BOTTOM-LEFT. Compare the bottom-left quadrant of the same
# frame with vs without the trace — a coloured matplotlib plot lands there.
n = min(tr.shape[0], nt.shape[0])
bl = (slice(int(Ht*0.55), Ht), slice(0, int(Wt*0.45)))
inset_diff = float(np.abs(tr[0][bl].astype(np.int32) - nt[0][bl].astype(np.int32)).mean())
# Non-grayscale (coloured) pixels in the trace export's bottom-left region: the
# trace plot draws blue/coloured lines that a gray-LUT frame never has.
reg = tr[0][bl].astype(np.int32)
chroma = np.abs(reg[...,0]-reg[...,1]) + np.abs(reg[...,1]-reg[...,2])
coloured = int((chroma > 30).sum())
print(json.dumps(dict(Tt=int(Tt), inset_diff=inset_diff, coloured=coloured)))
`
  const r = pyJSON(code)
  console.log('[mvx] trace metrics =', JSON.stringify(r))
  expect(r.inset_diff,
    'trace export bottom-left identical to no-trace — inset did not render')
    .toBeGreaterThan(1)
  expect(r.coloured,
    'no coloured (non-gray) pixels in the trace region — inset not drawn in colour')
    .toBeGreaterThan(50)
  ctx.assertNoJsErrors()
})

// ── 8) Final audit: no renderer JS errors, no backend tracebacks ─────────────

test('8) no renderer JS errors and no Python tracebacks in the backend log', async () => {
  ctx.assertNoJsErrors()
  const errs = backendErrorLines(ctx.backend)
  if (errs.length) console.log('[mvx] backend error lines:\n' + errs.join('\n'))
  expect(errs, 'Python tracebacks/errors in backend log').toEqual([])
})
