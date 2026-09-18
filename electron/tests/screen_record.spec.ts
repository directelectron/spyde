/**
 * screen_record.spec.ts — Help → Record Screen writes a real video of the app.
 *
 * The save dialog is a NATIVE modal Playwright cannot drive, so main's
 * `dialog.showSaveDialog` is stubbed in-process for the test (the product code
 * has no test-only branch). Everything after that is the real path: a real
 * click — which is what gives getDisplayMedia its user activation — real tab
 * capture, real chunks over IPC, real appends to disk.
 *
 * The dataset matters. What capture has to prove is that the FIGURE iframes
 * come through: they are a separate `spyde-fig://` origin in their own
 * processes, so if tab capture missed out-of-process frames the recording would
 * be app chrome around an empty hole.
 *
 * Proving that means DECODING the file, which needs ffmpeg — present on a dev
 * box, absent on the CI runners. So the checks are tiered: the capture itself,
 * and that the bytes are a real container, are asserted everywhere from the
 * file header alone; the frame-level assertions run where ffmpeg exists and
 * skip loudly where it doesn't. Do NOT collapse this back to one tier — a
 * recording that is merely non-empty is not evidence the figures are in it.
 */
import { test, expect } from '@playwright/test'
import { existsSync, statSync, mkdtempSync, mkdirSync, openSync, readSync, closeSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { execFileSync } from 'child_process'

const {
  launchApp, backendAction, waitForSubwindowCount,
} = require('./_harness.cjs')

/** ffmpeg/ffprobe are a dev-box convenience, not a CI dependency. */
const HAS_FFMPEG = (() => {
  try {
    execFileSync('ffprobe', ['-version'], { stdio: 'ignore' })
    execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' })
    return true
  } catch { return false }
})()

/** mp4 ('....ftyp') or Matroska/WebM (0x1A45DFA3) — read from the header, no decoder. */
function containerOf(path: string): 'mp4' | 'webm' | 'unknown' {
  const head = Buffer.alloc(12)
  const fd = openSync(path, 'r')
  try { readSync(fd, head, 0, 12, 0) } finally { closeSync(fd) }
  if (head.subarray(4, 8).toString('latin1') === 'ftyp') return 'mp4'
  if (head[0] === 0x1a && head[1] === 0x45 && head[2] === 0xdf && head[3] === 0xa3) return 'webm'
  return 'unknown'
}

let ctx: Awaited<ReturnType<typeof launchApp>>
const shots = join(__dirname, '..', 'screen_record_shots')
const scratch = mkdtempSync(join(tmpdir(), 'spyde-rec-'))
const outPath = join(scratch, 'capture.mp4')
const motionPath = join(scratch, 'motion.mp4')

test.beforeAll(async () => {
  mkdirSync(shots, { recursive: true })
  ctx = await launchApp()
})
test.afterAll(async () => { await ctx?.app.close() })

test('records the window to a real video file', async () => {
  const page = ctx.page
  await ctx.app.evaluate(({ dialog }, filePath) => {
    dialog.showSaveDialog = async () => ({ canceled: false, filePath })
  }, outPath)

  await backendAction(page, 'load_test_data_si_grains')
  await waitForSubwindowCount(page, 2)
  await page.waitForTimeout(2000)   // let the figures paint before capture

  await page.getByTestId('menu-help').click()
  await page.getByTestId('menu-item-record-screen').click()

  const pill = page.getByTestId('screen-recorder-pill')
  await expect(pill).toBeVisible()
  await expect(pill).toContainText('REC')
  await page.screenshot({ path: join(shots, '01-recording.png') })

  // Past one CHUNK_MS (3 s), so at least one chunk has been appended.
  await expect(pill).toContainText('0:05', { timeout: 15000 })
  await pill.click()
  await expect(pill).toContainText('Saved', { timeout: 15000 })
  await page.screenshot({ path: join(shots, '02-saved.png') })

  expect(existsSync(outPath)).toBe(true)
  const bytes = statSync(outPath).size
  const container = containerOf(outPath)
  console.log(`[screen-record] ${outPath} — ${bytes} bytes, container=${container}`)
  // A real muxed container, not a stub or a truncated first chunk.
  expect(container).not.toBe('unknown')
  expect(bytes).toBeGreaterThan(10_000)

  await ctx.assertNoJsErrors()
})

test('the recording contains the figures', async () => {
  test.skip(!HAS_FFMPEG, 'needs ffmpeg to decode the recording written by the previous test')

  const probe = execFileSync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration:stream=codec_name,width,height',
    '-of', 'default=noprint_wrappers=1', outPath,
  ]).toString()
  console.log(`[screen-record] ${probe.replace(/\n/g, ' ')}`)
  expect(probe).toMatch(/codec_name=(h264|vp9|vp8|av1)/)
  expect(Number(probe.match(/duration=([\d.]+)/)![1])).toBeGreaterThan(3)

  execFileSync('ffmpeg', ['-v', 'error', '-y', '-ss', '3', '-i', outPath,
                          '-vframes', '1', join(shots, '03-frame-from-video.png')])
  expect(existsSync(join(shots, '03-frame-from-video.png'))).toBe(true)
  await ctx.assertNoJsErrors()
})

test('the recording is live, not a still', async () => {
  test.skip(!HAS_FFMPEG, 'needs ffmpeg to pull frames back out of the recording')
  const page = ctx.page
  await ctx.app.evaluate(({ dialog }, filePath) => {
    dialog.showSaveDialog = async () => ({ canceled: false, filePath })
  }, motionPath)

  await page.getByTestId('menu-help').click()
  await page.getByTestId('menu-item-record-screen').click()
  const pill = page.getByTestId('screen-recorder-pill')
  await expect(pill).toContainText('REC')

  // Change the diffraction pattern's colormap WHILE recording. The repaint
  // happens inside the figure's own out-of-process frame, so two frames pulled
  // back out of the file either show gray then viridis — capture is following
  // the live iframe — or they don't, and it isn't.
  await page.waitForTimeout(2000)   // so the t=1s frame is still the OLD colormap
  await page.getByTestId("colormap-select").click()
  await page.getByText('viridis', { exact: true }).click()
  await page.waitForTimeout(2500)
  await expect(pill).toContainText('0:06', { timeout: 20000 })
  await pill.click()
  await expect(pill).toContainText('Saved', { timeout: 20000 })

  for (const t of ['1', '5']) {
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-ss', t, '-i', motionPath,
                            '-vframes', '1', join(shots, `1${t}-t${t}s.png`)])
  }
  // "The frames differ" would be satisfied by the REC clock alone. What has to
  // be true is that the FIGURE repainted in the capture, so count strongly blue
  // pixels: viridis floods the pattern with them, gray has none, and the app
  // chrome around it is identical in both frames.
  const blueCount = (png: string) => {
    const rgb = execFileSync('ffmpeg', ['-v', 'error', '-i', png,
                                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                             { maxBuffer: 1 << 28 })
    let n = 0
    for (let i = 0; i < rgb.length; i += 3) if (rgb[i + 2] > rgb[i] + 60) n++
    return n
  }
  const [early, late] = ['1', '5'].map((t) => blueCount(join(shots, `1${t}-t${t}s.png`)))
  console.log(`[screen-record] strongly-blue pixels: ${early} -> ${late}`)
  expect(late).toBeGreaterThan(early * 5)
  await ctx.assertNoJsErrors()
})
