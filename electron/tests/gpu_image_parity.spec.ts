/**
 * gpu_image_parity.spec.ts — GPU-vs-CPU parity + zoom/pan stability of the
 * large-image signal window, end-to-end in the real Electron app.
 *
 * Uses the SYNTHETIC in-situ movie (load_test_data_movie: 6 × 2048² frames,
 * no file, no download) whose content makes coordinate bugs visible: an x+y
 * gradient + unique corner blocks (any mirror/flip shows), a per-frame index
 * band (a stale frame shows), and a fine checkerboard (a blurry overview
 * masquerading as the crisp tile shows).
 *
 * Test 1 renders the SAME scene twice — default (WebGPU path) and with
 * SPYDE_GPU_IMAGE=0 (Canvas2D reference) — drives IDENTICAL zoom/pan through
 * the real wheel/drag handlers, waits for the detail tile to cover the view,
 * and pixel-compares the signal-window screenshots at each viewpoint.
 *
 * Test 2 is self-contained on the GPU path: pan direction must follow the
 * cursor (regression: the shader v-mirror inverted pan-y), NO blank/black
 * frame may appear during zoom/pan/scrub (data-flash audit), and scrubbing
 * while zoomed must paint FRESH detail tiles (regression: detail_seq dedup
 * froze the zoomed view on the first frame).
 */
import { test, expect } from '@playwright/test'
import { mkdirSync, writeFileSync } from 'fs'
const { launchApp, backendAction, waitForSubwindowCount } = require('./_harness.cjs')

const SHOTS = 'gpu_parity_shots'
mkdirSync(SHOTS, { recursive: true })

// ── helpers ─────────────────────────────────────────────────────────────────

/** Find the figure iframe whose panel is the LARGE signal image; return
 *  { fr, panelId, iframeEl } or null. Works on both GPU and CPU runs — draw2d
 *  records __apl_gpu2d[pid] (wanted/active/iw/ih) on every frame either way.
 *  Polls until `timeout` — the 2048² overview build + iframe boot can outlast
 *  any fixed sleep on a cold first launch. */
async function findSignalFrame(page: any, minEdge = 2000, timeout = 90_000) {
  const deadline = Date.now() + timeout
  do {
    for (const fr of page.frames()) {
      try {
        const rec = await fr.evaluate((edge: number) => {
          const g: any = (globalThis as any).__apl_gpu2d
          if (!g) return null
          for (const k of Object.keys(g))
            if (g[k].iw >= edge && g[k].ih >= edge) return { id: k, ...g[k] }
          return null
        }, minEdge)
        if (rec) {
          const iframeEl = await fr.frameElement()
          return { fr, panelId: rec.id, rec, iframeEl }
        }
      } catch { /* frame gone */ }
    }
    await page.waitForTimeout(1000)
  } while (Date.now() < deadline)
  return null
}

/** Set an exact zoom via the test hook, then fire ONE real wheel tick so the
 *  genuine wheel handler runs (tile mode: centre-zoom ×1.1) and emits the
 *  debounced view_changed → SpyDE samples + ships a hi-res detail tile.
 *  Ends at exactly `zoom` with center (cx, cy). */
async function zoomTo(fr: any, panelId: string, zoom: number, cx: number, cy: number) {
  await fr.evaluate(
    ({ pid, z, cx, cy }: any) => {
      const setZoom = (globalThis as any).__apl_setZoom
      setZoom(pid, z / 1.1, cx, cy)
      const ov = Array.from(document.querySelectorAll('canvas'))
        .find((c: any) => c.style && c.style.zIndex === '5') as HTMLCanvasElement
      ov.dispatchEvent(new WheelEvent('wheel',
        { deltaY: -100, bubbles: true, cancelable: true }))
    },
    { pid: panelId, z: zoom, cx, cy })
}

/** Wait until the detail tile fully covers the current visible window (the
 *  crisp-zoom settled state), then let the 90 ms blend ramp finish. */
async function waitDetailCovers(fr: any, panelId: string, timeout = 30_000) {
  await fr.waitForFunction((pid: string) => {
    const f = (globalThis as any).__apl_viewStateJson
    if (typeof f !== 'function') return false
    const stj = f(pid)
    if (!stj) return false
    const st = JSON.parse(stj)
    const reg = st.detail_region || []
    if (reg.length !== 4) return false
    const iw = st.image_width, ih = st.image_height, z = st.zoom || 1
    if (z < 1.05) return true
    const cx = st.center_x ?? 0.5, cy = st.center_y ?? 0.5
    const visW = iw / z, visH = ih / z
    const sx = Math.max(0, Math.min(iw - visW, cx * iw - visW / 2))
    const sy = Math.max(0, Math.min(ih - visH, cy * ih - visH / 2))
    const [tx0, tx1, ty0, ty1] = reg
    return sx >= tx0 - 0.5 && sx + visW <= tx1 + 0.5
        && sy >= ty0 - 0.5 && sy + visH <= ty1 + 0.5
  }, panelId, { timeout })
  await fr.page().waitForTimeout(300)
}

/** Screenshot the signal figure iframe element; returns a PNG Buffer. */
async function snapSignal(iframeEl: any, name: string): Promise<Buffer> {
  const buf: Buffer = await iframeEl.screenshot()
  writeFileSync(`${SHOTS}/${name}.png`, buf)
  return buf
}

/** Decode + diff two PNG buffers INSIDE the app page (no node png dep):
 *  returns { w, h, meanDiff, maxDiff, fracOver } where fracOver = fraction of
 *  pixels with any channel differing by > tol. */
async function diffPngs(page: any, a: Buffer, b: Buffer, tol = 10) {
  return await page.evaluate(async ({ b64a, b64b, tol }: any) => {
    const load = (b64: string) => new Promise<HTMLImageElement>((res, rej) => {
      const img = new Image()
      img.onload = () => res(img)
      img.onerror = rej
      img.src = 'data:image/png;base64,' + b64
    })
    const ia = await load(b64a), ib = await load(b64b)
    if (ia.width !== ib.width || ia.height !== ib.height)
      return { err: `size ${ia.width}x${ia.height} vs ${ib.width}x${ib.height}` }
    const w = ia.width, h = ia.height
    const cv = document.createElement('canvas')
    cv.width = w; cv.height = h
    const ctx = cv.getContext('2d')!
    ctx.drawImage(ia, 0, 0)
    const da = ctx.getImageData(0, 0, w, h).data
    ctx.clearRect(0, 0, w, h)
    ctx.drawImage(ib, 0, 0)
    const db = ctx.getImageData(0, 0, w, h).data
    let sum = 0, mx = 0, over = 0
    for (let i = 0; i < w * h; i++) {
      let pd = 0
      for (let c = 0; c < 3; c++) {
        const d = Math.abs(da[i * 4 + c] - db[i * 4 + c])
        sum += d
        if (d > pd) pd = d
      }
      if (pd > mx) mx = pd
      if (pd > tol) over++
    }
    return { w, h, meanDiff: +(sum / (w * h * 3)).toFixed(3),
             maxDiff: mx, fracOver: +(over / (w * h)).toFixed(4) }
  }, { b64a: a.toString('base64'), b64b: b.toString('base64'), tol })
}

/** Luminance stats of a PNG buffer (blank-frame detector), computed in-page. */
async function pngStats(page: any, buf: Buffer) {
  return await page.evaluate(async (b64: string) => {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image()
      i.onload = () => res(i)
      i.onerror = rej
      i.src = 'data:image/png;base64,' + b64
    })
    const cv = document.createElement('canvas')
    cv.width = img.width; cv.height = img.height
    const ctx = cv.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    const d = ctx.getImageData(0, 0, cv.width, cv.height).data
    let sum = 0, sum2 = 0
    const n = cv.width * cv.height
    for (let i = 0; i < n; i++) {
      const L = 0.3 * d[i * 4] + 0.59 * d[i * 4 + 1] + 0.11 * d[i * 4 + 2]
      sum += L; sum2 += L * L
    }
    const mean = sum / n
    return { mean: +mean.toFixed(1), std: +Math.sqrt(sum2 / n - mean * mean).toFixed(1) }
  }, buf.toString('base64'))
}

/** Drag the mouse over the iframe (pan): from its centre by (dx, dy). */
async function dragPan(page: any, iframeEl: any, dx: number, dy: number) {
  const box = await iframeEl.boundingBox()
  const sx = box.x + box.width / 2, sy = box.y + box.height / 2
  await page.mouse.move(sx, sy)
  await page.mouse.down()
  await page.mouse.move(sx + dx, sy + dy, { steps: 8 })
  await page.mouse.up()
}

/** Launch → load synthetic movie → scrub to frame 2 → return context. */
async function openMovie(env: Record<string, string>) {
  const expectGpu = env.SPYDE_GPU_IMAGE !== '0'
  const ctx = await launchApp({ dask: true, env })
  const { page } = ctx
  await page.waitForTimeout(1500)
  await backendAction(page, 'load_test_data_movie', {})
  await waitForSubwindowCount(page, 2, 120_000)
  await page.waitForTimeout(2500)
  // Scrub to a mid-movie frame so a real frame (not the t=0 placeholder) is up.
  await backendAction(page, 'test_nav_drag', { targets: [[2, 0]] })
  await page.waitForTimeout(2500)
  const sig = await findSignalFrame(page)
  expect(sig, 'no large signal image panel found').toBeTruthy()
  if (expectGpu) {
    // The WebGPU device init is async (first frame is Canvas2D by contract) —
    // wait for the activation redraw before driving the scenario, so every
    // screenshot is the GPU rendering.
    //
    // This activation IS the WebGPU-availability probe: it runs inside the
    // figure iframe (figure_esm's _gpuDevice: adapter + device + a real
    // GPUImage init), which is the ONLY context that agrees with the render
    // path. A top-page `navigator.gpu.requestAdapter()` probe was unreliable
    // on hosted runners — it could hand back a software adapter/device while
    // the iframe's real context creation still failed ("Failed to create
    // WebGPU Context Provider"), so the test proceeded and then hung its full
    // 600s waiting for an activation that never came. When the iframe never
    // activates within the window, SKIP (don't fail, don't hang) — hosted CI
    // runners have no usable WebGPU device. Library-level GPU render math is
    // CI-covered in anyplotlib's own test_gpu_parity_playwright.py
    // (chromium + --enable-unsafe-webgpu).
    try {
      await sig.fr.waitForFunction(
        (pid: string) => (globalThis as any).__apl_gpu2d?.[pid]?.active === true,
        sig.panelId, { timeout: 30_000 })
    } catch {
      await ctx.app.close().catch(() => {})
      test.skip(true, 'WebGPU image path never activated (software/CI runner) — GPU parity cannot be tested')
    }
  }
  return { ...ctx, sig }
}

/** Drive the shared zoom/pan scenario; screenshot at 3 viewpoints. */
async function runScenario(env: Record<string, string>, tag: string) {
  const ctx = await openMovie(env)
  const { page, sig, assertNoJsErrors } = ctx as any
  const shots: Record<string, Buffer> = {}
  const views: Record<string, any> = {}
  const readView = async () => {
    const st = JSON.parse(await sig.fr.evaluate(
      (pid: string) => (globalThis as any).__apl_viewStateJson(pid), sig.panelId))
    return { zoom: st.zoom, cx: st.center_x, cy: st.center_y }
  }
  try {
    // V1: zoomed in 3× at centre, detail tile settled.
    await zoomTo(sig.fr, sig.panelId, 3.0, 0.5, 0.5)
    await waitDetailCovers(sig.fr, sig.panelId)
    shots.v1 = await snapSignal(sig.iframeEl, `${tag}-v1-zoom3-center`)
    views.v1 = await readView()

    // V2: pan down-right via a REAL mouse drag (vertically off-centre window —
    // the shader v-mirror regression case), detail settled.
    await dragPan(page, sig.iframeEl, -70, -55)
    await waitDetailCovers(sig.fr, sig.panelId)
    shots.v2 = await snapSignal(sig.iframeEl, `${tag}-v2-panned`)
    views.v2 = await readView()

    // V3: zoom back out below the tile threshold (overview base path).
    await zoomTo(sig.fr, sig.panelId, 1.0, 0.5, 0.5)
    await page.waitForTimeout(800)
    shots.v3 = await snapSignal(sig.iframeEl, `${tag}-v3-zoomout`)
    views.v3 = await readView()
    console.log(`${tag} views:`, JSON.stringify(views))
    // DIAG: base freshness at v3 — token + bytes fingerprint + blit cache key.
    const diag = await sig.fr.evaluate(
      (pid: string) => (globalThis as any).__apl_panelDiag?.(pid), sig.panelId)
    console.log(`${tag} v3 diag:`, JSON.stringify(diag))

    const gpuActive = (await sig.fr.evaluate(
      (pid: string) => (globalThis as any).__apl_gpu2d[pid], sig.panelId)).active
    assertNoJsErrors()
    return { shots, views, gpuActive, app: ctx.app, page }
  } catch (e) {
    await ctx.app.close().catch(() => {})
    throw e
  }
}

// ── tests ───────────────────────────────────────────────────────────────────

test('movie signal window: GPU render matches the Canvas2D reference under zoom/pan', async () => {
  test.setTimeout(600_000)

  // GPU run first (default), then the CPU reference (SPYDE_GPU_IMAGE=0).
  const gpu = await runScenario({}, 'gpu')
  await gpu.app.close()
  const cpu = await runScenario({ SPYDE_GPU_IMAGE: '0' }, 'cpu')

  try {
    expect(gpu.gpuActive, 'GPU run: WebGPU image path must be active').toBe(true)
    expect(cpu.gpuActive, 'CPU run: WebGPU must be OFF (SPYDE_GPU_IMAGE=0)').toBe(false)
    // Identical interactions must land on identical view state AT EVERY
    // viewpoint — if these diverge the bug is in the interaction/echo path,
    // not the render (and the pixel comparison below would be meaningless).
    for (const key of ['v1', 'v2', 'v3'] as const) {
      const g = gpu.views[key], c = cpu.views[key]
      expect(Math.abs(g.zoom - c.zoom), `view ${key} zoom: gpu=${g.zoom} cpu=${c.zoom}`)
        .toBeLessThan(1e-9)
      expect(Math.abs(g.cx - c.cx), `view ${key} center_x: gpu=${g.cx} cpu=${c.cx}`)
        .toBeLessThan(1e-6)
      expect(Math.abs(g.cy - c.cy), `view ${key} center_y: gpu=${g.cy} cpu=${c.cy}`)
        .toBeLessThan(1e-6)
    }

    for (const key of ['v1', 'v2', 'v3'] as const) {
      const d: any = await diffPngs(cpu.page, gpu.shots[key], cpu.shots[key])
      console.log(`parity ${key}:`, JSON.stringify(d))
      expect(d.err, `parity ${key}: ${d.err}`).toBeUndefined()
      // Tolerance: nearest-sampling rounding between the shader and Canvas2D
      // drawImage; a coordinate bug (mirror/offset) moves whole regions and
      // produces fracOver > 0.3.
      expect(d.fracOver,
        `parity ${key}: GPU render diverged from Canvas2D reference (${JSON.stringify(d)})`,
      ).toBeLessThan(0.02)
    }
  } finally {
    await cpu.app.close()
  }
})

test('GPU zoom/pan is stable: pan follows cursor, no blank flashes, fresh frames while zoomed', async () => {
  test.setTimeout(600_000)
  const ctx = await openMovie({})
  const { app, page, sig, assertNoJsErrors } = ctx as any
  try {
    const gpuRec = await sig.fr.evaluate(
      (pid: string) => (globalThis as any).__apl_gpu2d[pid], sig.panelId)
    expect(gpuRec.active, 'WebGPU image path must be active').toBe(true)

    // ── Pan direction: drag DOWN must move image content DOWN ──────────────
    await zoomTo(sig.fr, sig.panelId, 3.0, 0.5, 0.5)
    await waitDetailCovers(sig.fr, sig.panelId)
    const before = await snapSignal(sig.iframeEl, 'stable-before-pan')
    await dragPan(page, sig.iframeEl, 0, 45)
    await page.waitForTimeout(250)
    const after = await snapSignal(sig.iframeEl, 'stable-after-pan')
    const dir: any = await page.evaluate(async ({ b64a, b64b, shift }: any) => {
      const load = (b64: string) => new Promise<HTMLImageElement>((res, rej) => {
        const i = new Image(); i.onload = () => res(i); i.onerror = rej
        i.src = 'data:image/png;base64,' + b64
      })
      const ia = await load(b64a), ib = await load(b64b)
      const w = ia.width, h = ia.height
      const cv = document.createElement('canvas')
      cv.width = w; cv.height = h
      const ctx2 = cv.getContext('2d')!
      const gray = (img: HTMLImageElement) => {
        ctx2.clearRect(0, 0, w, h); ctx2.drawImage(img, 0, 0)
        const d = ctx2.getImageData(0, 0, w, h).data
        const g = new Float32Array(w * h)
        for (let i = 0; i < w * h; i++)
          g[i] = 0.3 * d[i * 4] + 0.59 * d[i * 4 + 1] + 0.11 * d[i * 4 + 2]
        return g
      }
      const ga = gray(ia), gb = gray(ib)
      // Compare central band: after ≈ before shifted DOWN by `shift` px.
      const y0 = 80, y1 = h - 80, x0 = 60, x1 = w - 60
      let sCorrect = 0, sInverted = 0, n = 0
      for (let y = y0; y < y1; y++)
        for (let x = x0; x < x1; x += 2) {
          sCorrect += Math.abs(gb[(y + shift) * w + x] - ga[y * w + x])
          sInverted += Math.abs(gb[(y - shift) * w + x] - ga[y * w + x])
          n++
        }
      return { correct: +(sCorrect / n).toFixed(2), inverted: +(sInverted / n).toFixed(2) }
    }, { b64a: before.toString('base64'), b64b: after.toString('base64'), shift: 45 })
    console.log('pan direction:', JSON.stringify(dir))
    expect(dir.correct,
      `pan-y direction inverted on GPU path (down-shift diff ${dir.correct} ` +
      `should beat up-shift diff ${dir.inverted})`,
    ).toBeLessThan(dir.inverted)

    // ── No blank flashes during a continuous pan ────────────────────────────
    const box = await sig.iframeEl.boundingBox()
    const cx0 = box.x + box.width / 2, cy0 = box.y + box.height / 2
    await page.mouse.move(cx0, cy0)
    await page.mouse.down()
    for (let i = 1; i <= 5; i++) {
      await page.mouse.move(cx0 - i * 12, cy0 - i * 9, { steps: 2 })
      const shot = await snapSignal(sig.iframeEl, `stable-panflash-${i}`)
      const st: any = await pngStats(page, shot)
      expect(st.mean, `blank/black frame mid-pan (step ${i}): ${JSON.stringify(st)}`)
        .toBeGreaterThan(20)
      expect(st.std, `flat frame mid-pan (step ${i}): ${JSON.stringify(st)}`)
        .toBeGreaterThan(5)
    }
    await page.mouse.up()
    await waitDetailCovers(sig.fr, sig.panelId)

    // ── Scrub while zoomed: each frame must repaint the detail tile ────────
    // (regression: the detail dedup key froze the zoomed view on frame 1).
    let prev = await snapSignal(sig.iframeEl, 'stable-scrub-t2')
    let changes = 0
    for (const t of [3, 4]) {
      await backendAction(page, 'test_nav_drag', { targets: [[t, 0]] })
      await page.waitForTimeout(2500)
      const cur = await snapSignal(sig.iframeEl, `stable-scrub-t${t}`)
      const st: any = await pngStats(page, cur)
      expect(st.mean, `blank frame after scrub to t=${t}`).toBeGreaterThan(20)
      const d: any = await diffPngs(page, prev, cur, 12)
      console.log(`scrub t=${t}: diff`, JSON.stringify(d))
      if (!d.err && d.fracOver > 0.005) changes++
      prev = cur
    }
    expect(changes,
      'zoomed-in display FROZE during scrub (detail tile not re-uploaded per frame)',
    ).toBeGreaterThanOrEqual(2)

    assertNoJsErrors()
  } finally {
    await app.close()
  }
})
