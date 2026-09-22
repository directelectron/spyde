/**
 * multiangle_loader_dialog.spec.ts — the Multi-Angle 4D STEM loader DIALOG.
 *
 * Renderer-only: the dialog renders from the backend's `maped_state` snapshot,
 * so injecting that message is the whole input surface, and `spyde:action` is
 * the whole output surface. That makes every tableau, lock, edit and run
 * assertable without a single member file — and lets this spec cover the states
 * a real acquisition only reaches after minutes of alignment (a solved fit with
 * an ambiguous half-pixel residual, four corner sums per member, a file that
 * failed to open).
 *
 * The previews are REAL PNGs, encoded here rather than stubbed with a blank
 * pixel: the whole point of the tableau is that it shows pictures, and a spec
 * that feeds it nothing to draw cannot tell a working thumbnail from a grey
 * square. The screenshots at the bottom are the actual check (CLAUDE.md).
 *
 * What it CANNOT reach, and no headless test can: a real OS file drop. A File
 * built in the renderer has no path on disk, so `webUtils.getPathForFile`
 * (Electron 44 removed `File.path`) has nothing to resolve — which is exactly
 * the case asserted below, that a pathless drop SAYS so instead of silently
 * doing nothing. That a genuine drag from Finder/Explorer yields real paths has
 * to be checked by hand in the running ctx.app.
 */
import { test, expect, Page } from '@playwright/test'
import { join } from 'path'
import { deflateSync } from 'zlib'

const { launchApp } = require('./_harness.cjs')

let ctx: any
let page: Page

test.beforeAll(async () => {
  ctx = await launchApp({ dask: false })
  page = ctx.page
})

test.afterAll(async () => { await ctx?.app?.close() })

/** Every spec starts from a closed dialog and an empty action log. The action
 *  channel's own listener is removed, so nothing here reaches Python — the
 *  dialog's contract is what it SENDS, and the backend half is tested in
 *  Python. */
test.beforeEach(async () => {
  await page.reload()
  await page.waitForSelector('[data-testid="mdi-area"]')
  await ctx.app.evaluate(({ ipcMain }) => {
    ;(globalThis as any).__sent = []
    ipcMain.removeAllListeners('spyde:action')
    ipcMain.on('spyde:action', (_e, action, payload) => {
      ;(globalThis as any).__sent.push({ action, payload })
    })
  })
})

const sentActions = (): Promise<{ action: string; payload: any }[]> =>
  ctx.app.evaluate(() => (globalThis as any).__sent)

const sentNamed = async (name: string) =>
  (await sentActions()).filter((a) => a.action === name)

async function inject(msg: Record<string, unknown>) {
  await page.evaluate((m) => { (window as any)._spyde_test_inject?.(m) }, msg)
}

async function openLoader() {
  await page.getByTestId('menu-file').click()
  await page.getByTestId('menu-load-multiangle').click()
  await expect(page.getByTestId('multiangle-loader')).toBeVisible()
}

/** Stub the native picker in the MAIN process: a contextBridge object is
 *  immutable, so reassigning window.electron.pickFiles silently no-ops (and a
 *  real native dialog would block the run). */
async function stubPicker(paths: string[]) {
  await ctx.app.evaluate(({ ipcMain }, p) => {
    ipcMain.removeHandler('spyde:pick-files')
    ipcMain.handle('spyde:pick-files', async () => p)
  }, paths)
}

/** Drop a File with no path on disk — the shape of the one failure a drop
 *  handler can have silently. */
async function dropPathlessFile(testid: string) {
  await page.getByTestId(testid).evaluate((el: HTMLElement) => {
    const dt = new DataTransfer()
    dt.items.add(new File(['x'], 'angle00.mrc', { type: 'application/octet-stream' }))
    el.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }))
  })
}

/** An internal drag: one DataTransfer carried from a source element's
 *  `dragstart` to a target's `drop`, which is how the dialog re-angles a member
 *  without anyone typing a number. */
async function dragOnto(fromTestid: string, toTestid: string) {
  await page.evaluate(([from, to]) => {
    const source = document.querySelector(`[data-testid="${from}"]`) as HTMLElement
    const target = document.querySelector(`[data-testid="${to}"]`) as HTMLElement
    const dt = new DataTransfer()
    const fire = (el: HTMLElement, type: string) => el.dispatchEvent(
      new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }))
    fire(source, 'dragstart')
    fire(target, 'dragover')
    fire(target, 'drop')
  }, [fromTestid, toTestid] as const)
}

// ── Real PNG previews, so the tableau has something to actually draw ────────

const CRC_TABLE = (() => {
  const table = new Int32Array(256)
  for (let n = 0; n < 256; n += 1) {
    let c = n
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c
  }
  return table
})()

function crc32(buf: Buffer): number {
  let crc = -1
  for (let i = 0; i < buf.length; i += 1) {
    crc = CRC_TABLE[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8)
  }
  return (crc ^ -1) >>> 0
}

function pngChunk(type: string, data: Buffer): Buffer {
  const length = Buffer.alloc(4)
  length.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const checksum = Buffer.alloc(4)
  checksum.writeUInt32BE(crc32(body))
  return Buffer.concat([length, body, checksum])
}

/** A tiny truecolour PNG as a data URI — the shape the backend sends. */
function pngDataUri(pixel: (x: number, y: number) => [number, number, number], size: number): string {
  const raw = Buffer.alloc(size * (size * 3 + 1))
  let p = 0
  for (let y = 0; y < size; y += 1) {
    raw[p] = 0; p += 1                     // filter byte: none
    for (let x = 0; x < size; x += 1) {
      const [r, g, b] = pixel(x, y)
      raw[p] = r; raw[p + 1] = g; raw[p + 2] = b
      p += 3
    }
  }
  const header = Buffer.alloc(13)
  header.writeUInt32BE(size, 0)
  header.writeUInt32BE(size, 4)
  header[8] = 8    // bit depth
  header[9] = 2    // truecolour
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', header),
    pngChunk('IDAT', deflateSync(raw)),
    pngChunk('IEND', Buffer.alloc(0)),
  ])
  return `data:image/png;base64,${png.toString('base64')}`
}

const PALETTE: [number, number, number][] = [
  [243, 139, 168], [250, 179, 135], [249, 226, 175], [166, 227, 161],
  [148, 226, 213], [137, 180, 250], [203, 166, 247], [245, 194, 231],
  [116, 199, 236], [180, 190, 254], [137, 220, 235],
]

/** A member's virtual image: its own hue, a diagonal ramp, and a bright corner
 *  block — asymmetric, so a mirrored or stale thumbnail is visible. */
function virtualImage(index: number, size = 28): string {
  const [r, g, b] = PALETTE[index % PALETTE.length]
  return pngDataUri((x, y) => {
    if (x >= size - 9 && y < 9) return [255, 255, 255]
    const ramp = 0.3 + (0.7 * (x + y)) / (2 * size)
    return [Math.round(r * ramp), Math.round(g * ramp), Math.round(b * ramp)]
  }, size)
}

/** A scan corner's summed diffraction pattern: a dark field with the direct
 *  beam a little way off centre, in a different direction per corner — which
 *  is exactly the tilt across the scan the corner method is measuring. */
function cornerSum(corner: number, size = 30): string {
  const cx = size / 2 + (corner % 2 === 0 ? -4 : 4)
  const cy = size / 2 + (corner < 2 ? -4 : 4)
  return pngDataUri((x, y) => {
    const distance = Math.hypot(x - cx, y - cy)
    const beam = Math.max(0, 1 - distance / 5)
    const halo = Math.max(0, 1 - Math.abs(distance - 10) / 3) * 0.35
    const value = Math.min(1, beam + halo)
    return [Math.round(24 + 231 * value), Math.round(26 + 200 * value), Math.round(38 + 160 * value)]
  }, size)
}

// ── A realistic acquisition: ten members, two shells, one that failed ───────

const member = (index: number, tilt: number, azimuth: number) => ({
  index,
  path: `/data/angle${String(index).padStart(2, '0')}.mrc`,
  name: `angle${String(index).padStart(2, '0')}.mrc`,
  scan_shape: [32, 32],
  detector_shape: [256, 256],
  dtype: 'uint16',
  size_bytes: 32 * 32 * 256 * 256 * 2,
  tilt,
  azimuth,
  shell: tilt === 1.0 ? 0 : 1,
  error: null,
  preview: virtualImage(index),
  virtual_images: ['Bright field', 'HAADF'],
})

const LOADED = {
  type: 'maped_state',
  members: [
    ...[0, 60, 120, 180, 240, 300].map((az, i) => member(i, 1.0, az)),
    ...[0, 90, 180, 270].map((az, i) => member(i + 6, 0.5, az)),
  ],
  reference: 0,
  shells: [
    { shell: 0, tilt: 1.0, members: [0, 1, 2, 3, 4, 5] },
    { shell: 1, tilt: 0.5, members: [6, 7, 8, 9] },
  ],
  scan_shape: null,
  virtual_image: 'Bright field',
  available_virtual_images: ['Bright field', 'HAADF'],
  real: { solved: false, offsets: null, residuals: null, max_residual: null },
  reciprocal: { solved: false, offsets: null, residuals: null, max_residual: null },
  busy: false,
  message: '10 datasets loaded',
  can_commit: false,
}

/** A member the backend added but nobody has angled yet — it belongs to no
 *  ring, so the picture has to keep it somewhere visible. */
const UNANGLED = {
  index: 10,
  path: '/data/angle10.mrc',
  name: 'angle10.mrc',
  scan_shape: [32, 32], detector_shape: [256, 256],
  dtype: 'uint16', size_bytes: 1024 * 256 * 256 * 2,
  tilt: null, azimuth: null, shell: null, error: null,
  preview: virtualImage(10),
  virtual_images: ['Bright field', 'HAADF'],
}

const FAILED = {
  index: 11,
  path: '/data/angle11.mrc',
  name: 'angle11.mrc',
  scan_shape: null, detector_shape: null, dtype: null, size_bytes: null,
  tilt: null, azimuth: null, shell: null,
  error: 'not a 4-D dataset (shape (32, 32))',
  preview: null,
  virtual_images: [],
}

/** The same acquisition with a real-space solve on it. The last member's
 *  residual is 0.47 px — the case the UI must not round into a confident
 *  integer without saying so. */
const REAL_SOLVED = {
  ...LOADED,
  real: {
    solved: true,
    offsets: [[0, 0], [1, -2], [2, -3], [1, 4], [-1, 3], [-2, 1],
              [0, 1], [1, 1], [-1, 0], [3, -4]],
    residuals: [0.0, 0.08, 0.11, 0.05, 0.13, 0.09, 0.04, 0.07, 0.12, 0.47],
    max_residual: 0.47,
  },
  message: 'Real-space alignment solved',
}

/** Every member's four scan-corner sums, at the extents they were summed over. */
const CORNERS = Object.fromEntries(LOADED.members.map((m) => [
  String(m.index),
  { previews: [0, 1, 2, 3].map((corner) => cornerSum(corner)), extents: [4, 4, 4, 4] },
]))

const BOTH_SOLVED = {
  ...REAL_SOLVED,
  reciprocal: {
    solved: true,
    offsets: [[0, 0], [0, 1], [-1, 0], [1, 1], [0, -1], [1, 0],
              [0, 0], [-1, 1], [0, 1], [1, -1]],
    residuals: [0.0, 0.02, 0.03, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02, 0.05],
    max_residual: 0.05,
    corners: CORNERS,
  },
  can_commit: true,
  message: 'Ready to open',
}

/** Bare MRCs: each file says how many frames and how big the detector is, but
 *  not how the frames were laid out, so every member probes as a 3-D stack. */
const NO_SCAN_GRID = {
  type: 'maped_state',
  members: [0, 1, 2].map((index) => ({
    index,
    path: `/data/angle0${index}.mrc`,
    name: `angle0${index}.mrc`,
    scan_shape: null,
    detector_shape: [256, 256],
    dtype: 'uint16',
    size_bytes: 1024 * 256 * 256 * 2,
    tilt: null, azimuth: null, shell: null,
    error: 'no scan grid: 1024 frames of 256×256, and nothing says how they were scanned',
    preview: null,
    virtual_images: [],
  })),
  reference: null,
  shells: [],
  scan_shape: null,
  virtual_image: null,
  available_virtual_images: [],
  real: { solved: false, offsets: null, residuals: null, max_residual: null },
  reciprocal: { solved: false, offsets: null, residuals: null, max_residual: null },
  busy: false,
  message: '3 datasets need a scan grid',
  can_commit: false,
}

/** The same three files re-probed once a 32 × 32 grid was supplied. */
const SCAN_GRID_SET = {
  ...NO_SCAN_GRID,
  members: [0, 1, 2].map((index) => ({
    ...NO_SCAN_GRID.members[index],
    scan_shape: [32, 32],
    tilt: 1.0,
    azimuth: index * 120,
    shell: 0,
    error: null,
    preview: virtualImage(index),
  })),
  reference: 0,
  shells: [{ shell: 0, tilt: 1.0, members: [0, 1, 2] }],
  scan_shape: [32, 32],
  message: '3 members re-read as 32 × 32',
}

// ── Opening, and what an EMPTY loader offers ────────────────────────────────

test('File → Load Multi-Angle 4D STEM… opens the loader and announces itself', async () => {
  await openLoader()
  // The loader asks the backend for the acquisition exactly once, despite
  // StrictMode's mount → cleanup → remount.
  await expect.poll(() => sentNamed('maped_open_loader').then((a) => a.length))
    .toBe(1)
  await expect(page.getByTestId('maped-summary')).toContainText('No datasets')
  // Nothing to draw, so the picture says how to start it rather than being a
  // blank square.
  await expect(page.getByTestId('maped-tableau-load-empty')).toContainText('Add a ring')
})

test('the two alignment tabs are locked with nothing loaded', async () => {
  await openLoader()
  await expect(page.getByTestId('maped-tab-load')).toBeVisible()
  await expect(page.getByTestId('maped-tab-real')).toBeDisabled()
  await expect(page.getByTestId('maped-tab-reciprocal')).toBeDisabled()
  // …and the acquisition cannot be opened.
  await expect(page.getByTestId('maped-open')).toBeDisabled()
})

// ── The ring scaffolding: the renderer's own state ─────────────────────────

test('a ring lays out empty spots, and removing it takes them away', async () => {
  await openLoader()
  await page.getByTestId('maped-ring-tilt').fill('1.0')
  await page.getByTestId('maped-ring-count').fill('6')
  await page.getByTestId('maped-ring-add').click()

  // Six spots at 0/60/…/300, and NOTHING said to the backend: empty slots are
  // scaffolding, not acquisition state.
  await expect(page.getByTestId('maped-tableau-load')).toBeVisible()
  for (const azimuth of [0, 60, 120, 180, 240, 300]) {
    await expect(page.getByTestId(`maped-slot-1.00-${azimuth}`)).toBeVisible()
  }
  expect(await sentActions()).toEqual([{ action: 'maped_open_loader', payload: {} }])

  await expect(page.getByTestId('maped-ring-chip-spec-1')).toContainText('1° × 6')
  await page.getByTestId('maped-ring-remove-1').click()
  await expect(page.getByTestId('maped-slot-1.00-60')).toHaveCount(0)
  await expect(page.getByTestId('maped-tableau-load-empty')).toBeVisible()
})

test('clicking an empty spot picks a file and gives it that spot’s angles', async () => {
  await stubPicker(['/data/new-angle.mrc'])
  await openLoader()
  await page.getByTestId('maped-ring-count').fill('4')
  await page.getByTestId('maped-ring-add').click()

  await page.getByTestId('maped-slot-1.00-90').click()
  await expect.poll(() => sentNamed('maped_add_files')).toEqual([
    { action: 'maped_add_files', payload: { paths: ['/data/new-angle.mrc'] } },
  ])
  // The index is the backend's to issue, so the angles wait for the snapshot
  // that carries the new member — and then land on it.
  expect(await sentNamed('maped_set_member')).toHaveLength(0)
  await inject({
    ...LOADED,
    members: [{ ...UNANGLED, index: 7, path: '/data/new-angle.mrc', name: 'new-angle.mrc' }],
    shells: [], reference: null,
  })
  await expect.poll(() => sentNamed('maped_set_member')).toEqual([
    { action: 'maped_set_member', payload: { index: 7, tilt: 1, azimuth: 90 } },
  ])
})

test('a drop on a spot that carries no usable path says so rather than no-op', async () => {
  await openLoader()
  await page.getByTestId('maped-ring-add').click()
  // A File built in the renderer has no path on disk, so the preload's
  // webUtils.getPathForFile resolves nothing — the same shape as the silent
  // failure this note exists to prevent.
  await dropPathlessFile('maped-slot-1.00-120')
  await expect(page.getByTestId('maped-drop-note')).toContainText('Add datasets…')
  expect(await sentNamed('maped_add_files')).toHaveLength(0)
})

test('the same is true of the strip that adds files without an angle', async () => {
  await openLoader()
  await dropPathlessFile('maped-dropzone')
  await expect(page.getByTestId('maped-drop-note')).toBeVisible()
  expect(await sentNamed('maped_add_files')).toHaveLength(0)
})

test('the pickers feed maped_add_files', async () => {
  await stubPicker(['/data/a.mrc', '/data/b.mrc'])
  await openLoader()
  await page.getByTestId('maped-add-files').click()
  await expect.poll(() => sentNamed('maped_add_files')).toEqual([
    { action: 'maped_add_files', payload: { paths: ['/data/a.mrc', '/data/b.mrc'] } },
  ])
})

// ── A loaded acquisition, on the ring ──────────────────────────────────────

test('the header states what the acquisition IS', async () => {
  await openLoader()
  await inject(LOADED)
  await expect(page.getByTestId('maped-summary'))
    .toHaveText('10 angles · 2 shells · 1°, 0.5°')
})

test('members land on rings taken from their own angles, each drawing its preview', async () => {
  await openLoader()
  await inject(LOADED)
  // No scaffolding was laid out, so the rings come from the data — one per
  // tilt, listed as such, and every member is on one of them.
  await expect(page.getByTestId('maped-ring-chip-data-0.500')).toContainText('0.5° × 4')
  await expect(page.getByTestId('maped-ring-chip-data-1.000')).toContainText('1° × 6')
  await expect(page.getByTestId('maped-ring-chip-data-1.000')).toContainText('from data')

  const tableau = page.getByTestId('maped-tableau-load')
  for (let index = 0; index < 10; index += 1) {
    await expect(tableau.getByTestId(`maped-member-${index}`)).toBeVisible()
  }
  // The preview is an actual DECODED image, not an <img> pointing at nothing —
  // which is the failure a visibility assertion cannot see.
  await expect.poll(() => tableau.getByTestId('maped-member-3').locator('img')
    .evaluate((img: HTMLImageElement) => (img.complete ? img.naturalWidth : 0)))
    .toBe(28)
  // The caption drops the extension every member shares; the whole name is on
  // the tile's tooltip, and in the enlarged view.
  await expect(tableau.getByTestId('maped-member-3')).toContainText('angle03')
  await expect(tableau.getByTestId('maped-member-3'))
    .toHaveAttribute('title', /angle03\.mrc/)
})

test('a scaffold ring keeps its empty spots and draws members at their TRUE azimuth', async () => {
  await openLoader()
  await page.getByTestId('maped-ring-count').fill('8')
  await page.getByTestId('maped-ring-add').click()
  await inject(LOADED)

  // Six members at 0/60/…/300 nearest six of the eight spots; the two spots
  // nobody acquired stay open, and the members keep their own angles rather
  // than snapping to the 45° lattice they were laid against.
  await expect(page.getByTestId('maped-slot-1.00-90')).toBeVisible()
  await expect(page.getByTestId('maped-slot-1.00-270')).toBeVisible()
  await expect(page.getByTestId('maped-member-1'))
    .toHaveAttribute('title', /60° azimuth/)
  await expect(page.getByTestId('maped-ring-chip-spec-1')).toContainText('1° × 8')
})

test('a member with no angle waits in the tray, and dragging it onto a spot angles it', async () => {
  await openLoader()
  // Eight spots for the six members already at 1°, so two stay open.
  await page.getByTestId('maped-ring-count').fill('8')
  await page.getByTestId('maped-ring-add').click()
  await inject({ ...LOADED, members: [...LOADED.members, UNANGLED] })

  await expect(page.getByTestId('maped-unplaced')).toContainText('1 dataset with no angle')
  await expect(page.getByTestId('maped-unplaced-10')).toBeVisible()

  await dragOnto('maped-unplaced-10', 'maped-slot-1.00-270')
  await expect.poll(() => sentNamed('maped_set_member')).toEqual([
    { action: 'maped_set_member', payload: { index: 10, tilt: 1, azimuth: 270 } },
  ])
})

test('a member that failed to open is VISIBLE, not missing', async () => {
  await openLoader()
  await inject({ ...LOADED, members: [...LOADED.members, FAILED] })
  // It has no angles, so it cannot be on a ring — it lands in the tray, and its
  // reason is spelled out in full where nothing truncates it.
  await expect(page.getByTestId('maped-unplaced-11')).toBeVisible()
  await expect(page.getByTestId('maped-member-error-11'))
    .toContainText('not a 4-D dataset')
  await expect(page.getByTestId('maped-error-count')).toContainText('1 failed to open')
})

test('a failed member that DOES have angles keeps its spot, marked', async () => {
  await openLoader()
  await inject({
    ...LOADED,
    members: [...LOADED.members,
      { ...FAILED, tilt: 0.5, azimuth: 45, shell: 1 }],
  })
  await expect(page.getByTestId('maped-tableau-load').getByTestId('maped-member-11'))
    .toBeVisible()
  await expect(page.getByTestId('maped-slot-error-11')).toBeVisible()
  // …and it is still readable in words, not only as a red tile.
  await expect(page.getByTestId('maped-member-error-11')).toContainText('not a 4-D dataset')
})

// ── The enlarged panel: where a 58-pixel spot keeps its controls ───────────

test('double-clicking a spot enlarges it, and its angles, reference and removal live there', async () => {
  await openLoader()
  await inject(LOADED)
  await page.getByTestId('maped-member-7').dblclick()
  await expect(page.getByTestId('maped-zoom-caption')).toHaveText('angle07.mrc')
  await expect(page.getByTestId('maped-zoom-image')).toBeVisible()
  await expect(page.getByTestId('maped-zoom')).toContainText('256×256 det')

  await page.getByTestId('maped-tilt-7').fill('0.75')
  await expect.poll(() => sentNamed('maped_set_member')).toEqual([
    { action: 'maped_set_member', payload: { index: 7, tilt: 0.75 } },
  ])

  // `.click()`, not `.check()`: the radio is controlled by the snapshot, so it
  // does NOT move on its own — the dot follows only once the backend echoes a
  // state with the new reference, which is the no-local-copy rule made visible.
  await page.getByTestId('maped-reference-7').click()
  await expect.poll(() => sentNamed('maped_set_reference')).toEqual([
    { action: 'maped_set_reference', payload: { index: 7 } },
  ])
  await expect(page.getByTestId('maped-reference-7')).not.toBeChecked()
  await inject({ ...LOADED, reference: 7 })
  await expect(page.getByTestId('maped-reference-7')).toBeChecked()

  await page.getByTestId('maped-remove-7').click()
  await expect.poll(() => sentNamed('maped_remove_member')).toEqual([
    { action: 'maped_remove_member', payload: { index: 7 } },
  ])
  // Removing closes the panel it was removed from.
  await expect(page.getByTestId('maped-zoom')).toHaveCount(0)
})

test('the enlarged panel closes on Escape and on a click', async () => {
  await openLoader()
  await inject(LOADED)
  await page.getByTestId('maped-member-2').dblclick()
  await expect(page.getByTestId('maped-zoom')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('maped-zoom')).toHaveCount(0)

  await page.getByTestId('maped-member-2').dblclick()
  await page.getByTestId('maped-zoom').click({ position: { x: 8, y: 8 } })
  await expect(page.getByTestId('maped-zoom')).toHaveCount(0)
})

// ── The scan grid (the MRC fallback) ───────────────────────────────────────

test('the scan-grid field stays out of the way when the files supply one', async () => {
  await openLoader()
  await expect(page.getByTestId('maped-scan-shape')).toHaveCount(0)
  await inject(LOADED)
  await expect(page.getByTestId('maped-scan-shape')).toHaveCount(0)
})

test('bare MRCs surface the scan-grid field, and setting it re-probes them', async () => {
  await openLoader()
  await inject(NO_SCAN_GRID)
  await expect(page.getByTestId('maped-scan-shape')).toBeVisible()
  await expect(page.getByTestId('maped-member-error-0'))
    .toContainText('no scan grid')
  // Three files, but nothing that opened — an enabled alignment tab here would
  // only move the failure one click further in.
  await expect(page.getByTestId('maped-tab-real')).toBeDisabled()

  // One box alone is a half-typed pair, not a grid: nothing is sent until both
  // are valid — and tabbing across must not discard the first.
  await page.getByTestId('maped-scan-x').fill('32')
  expect(await sentNamed('maped_set_scan_shape')).toHaveLength(0)
  await page.getByTestId('maped-scan-y').fill('32')
  expect(await page.getByTestId('maped-scan-x').inputValue()).toBe('32')
  await expect.poll(() => sentNamed('maped_set_scan_shape')).toEqual([
    { action: 'maped_set_scan_shape', payload: { scan_shape: [32, 32] } },
  ])

  // The re-probe lands: the three files become members on a 1° ring, and the
  // field says what is now being applied.
  await inject(SCAN_GRID_SET)
  await expect(page.getByTestId('maped-scan-shape')).toContainText('32 × 32')
  await expect(page.getByTestId('maped-tableau-load').getByTestId('maped-member-0'))
    .toBeVisible()
  await expect(page.getByTestId('maped-summary')).toContainText('3 angles · 1 shell')
  await expect(page.getByTestId('maped-error-count')).toHaveCount(0)
  await expect(page.getByTestId('maped-tab-real')).toBeEnabled()
})

test('clearing both boxes sends null, so a wrong grid is undoable', async () => {
  await openLoader()
  await inject(SCAN_GRID_SET)
  // Still offered once a grid is forced — that is the only way back.
  await expect(page.getByTestId('maped-scan-shape')).toBeVisible()
  expect(await page.getByTestId('maped-scan-x').inputValue()).toBe('32')

  await page.getByTestId('maped-scan-x').fill('')
  expect(await sentNamed('maped_set_scan_shape')).toHaveLength(0)
  await page.getByTestId('maped-scan-y').fill('')
  await expect.poll(() => sentNamed('maped_set_scan_shape')).toEqual([
    { action: 'maped_set_scan_shape', payload: { scan_shape: null } },
  ])
})

test('the scan-grid ⓘ explains why an MRC needs one', async () => {
  await openLoader()
  await inject(NO_SCAN_GRID)
  await page.getByTestId('maped-info-scan-shape').click()
  await expect(page.getByTestId('maped-info-scan-shape-text'))
    .toContainText('_info.txt')
})

// ── Align real space: the virtual-image tableau ────────────────────────────

test('real space unlocks with members; reciprocal stays locked until it is solved', async () => {
  await openLoader()
  await inject(LOADED)
  await expect(page.getByTestId('maped-tab-real')).toBeEnabled()
  await expect(page.getByTestId('maped-tab-reciprocal')).toBeDisabled()

  await inject(REAL_SOLVED)
  await expect(page.getByTestId('maped-tab-reciprocal')).toBeEnabled()
})

test('the virtual image is chosen once, for every member at once', async () => {
  await openLoader()
  await inject(LOADED)
  await page.getByTestId('maped-tab-real').click()

  // The tableau is the same ring, showing each member's chosen virtual image.
  const tableau = page.getByTestId('maped-tableau-real')
  await expect(tableau.getByTestId('maped-member-5')).toBeVisible()
  await expect(tableau.getByTestId('maped-member-5').locator('img')).toBeVisible()

  await expect(page.getByTestId('maped-virtual-image'))
    .toHaveAttribute('data-value', 'Bright field')
  await page.getByTestId('maped-virtual-image').click()
  await page.getByTestId('maped-virtual-image-opt-HAADF').click()
  await expect.poll(() => sentNamed('maped_set_virtual_image')).toEqual([
    { action: 'maped_set_virtual_image', payload: { name: 'HAADF' } },
  ])

  // "Compute" is the absence of a name, not a name of its own.
  await inject({ ...LOADED, virtual_image: 'HAADF' })
  await page.getByTestId('maped-virtual-image').click()
  await page.getByTestId('maped-virtual-image-opt-__compute__').click()
  await expect.poll(() => sentNamed('maped_set_virtual_image').then((a) => a[1])).toEqual(
    { action: 'maped_set_virtual_image', payload: { name: null } })
})

test('a member that does not carry the chosen virtual image is marked', async () => {
  await openLoader()
  await inject({
    ...LOADED,
    virtual_image: 'HAADF',
    members: LOADED.members.map((m) => m.index === 4
      ? { ...m, virtual_images: ['Bright field'] } : m),
  })
  await page.getByTestId('maped-tab-real').click()
  await expect(page.getByTestId('maped-slot-missing-4')).toBeVisible()
  await expect(page.getByTestId('maped-slot-missing-5')).toHaveCount(0)
})

test('a real-space run reports its offsets and flags a half-pixel residual', async () => {
  await openLoader()
  await inject(LOADED)
  await page.getByTestId('maped-tab-real').click()
  await expect(page.getByTestId('maped-real-empty')).toContainText('Not solved')

  await page.getByTestId('maped-run-real').click()
  await expect.poll(() => sentNamed('maped_align_real')).toEqual([
    { action: 'maped_align_real', payload: { params: { max_shift: 32 } } },
  ])

  // Busy is the backend's to declare, and it must reach the Run button.
  await inject({ ...LOADED, busy: true, message: 'Aligning 10 members…' })
  await expect(page.getByTestId('maped-run-real')).toBeDisabled()
  await expect(page.getByTestId('maped-status')).toContainText('Aligning 10 members')

  await inject(REAL_SOLVED)
  await expect(page.getByTestId('maped-real-residual')).toContainText('0.470 px')
  await expect(page.getByTestId('maped-real-residual-warning')).toContainText('Near half a pixel')
  await expect(page.getByTestId('maped-real-offset-1')).toContainText('(1, -2)')
  await expect(page.getByTestId('maped-real-offset-9')).toContainText('(3, -4)')
  await expect(page.getByTestId('maped-real-offset-9')).toContainText('0.470 px')
})

test('max shift defaults to 32 px, is editable, and explains itself behind ⓘ', async () => {
  await openLoader()
  await inject(LOADED)
  await page.getByTestId('maped-tab-real').click()
  expect(await page.getByTestId('maped-max-shift').inputValue()).toBe('32')

  // The guard is against a periodic lattice, and that has to be findable
  // without putting a paragraph on the face of the tab.
  await page.getByTestId('maped-info-max-shift').click()
  await expect(page.getByTestId('maped-info-max-shift-text'))
    .toContainText('lattice translation')
  await page.getByTestId('maped-info-max-shift-text').click()
  await expect(page.getByTestId('maped-info-max-shift-text')).toHaveCount(0)

  await page.getByTestId('maped-max-shift').fill('8')
  await page.getByTestId('maped-run-real').click()
  await expect.poll(() => sentNamed('maped_align_real')).toEqual([
    { action: 'maped_align_real', payload: { params: { max_shift: 8 } } },
  ])
})

// ── Align reciprocal space: the corner tableau ─────────────────────────────

test('the corner method is the default, and the other two are still there', async () => {
  await openLoader()
  await inject(REAL_SOLVED)
  await page.getByTestId('maped-tab-reciprocal').click()
  await expect(page.getByTestId('maped-reciprocal-method'))
    .toHaveAttribute('data-value', 'corners')
  await expect(page.getByTestId('maped-reciprocal-method')).toContainText('Corners (fast)')

  await page.getByTestId('maped-run-reciprocal').click()
  await expect.poll(() => sentNamed('maped_align_reciprocal')).toEqual([
    { action: 'maped_align_reciprocal', payload: { method: 'corners', params: {} } },
  ])

  // Dropdown, not a native <select> (Dropdown.tsx): click the trigger, then
  // the option.
  await page.getByTestId('maped-reciprocal-method').click()
  await page.getByTestId('maped-reciprocal-method-opt-correlate').click()
  await page.getByTestId('maped-run-reciprocal').click()
  await expect.poll(() => sentNamed('maped_align_reciprocal').then((a) => a[1])).toEqual(
    { action: 'maped_align_reciprocal', payload: { method: 'correlate', params: {} } })
})

test('each member opens into its four corner sums, each with its own extent', async () => {
  await openLoader()
  await inject(BOTH_SOLVED)
  await page.getByTestId('maped-tab-reciprocal').click()

  const card = page.getByTestId('maped-corners-2')
  await expect(card).toContainText('angle02.mrc')
  await expect(card).toContainText('1° · 120°')
  for (let corner = 0; corner < 4; corner += 1) {
    await expect(card.getByTestId(`maped-corner-2-${corner}`).locator('img')).toBeVisible()
    expect(await card.getByTestId(`maped-corner-extent-2-${corner}`).inputValue()).toBe('4')
  }

  await card.getByTestId('maped-corner-extent-2-3').fill('9')
  await expect.poll(() => sentNamed('maped_set_corner_extent')).toEqual([
    { action: 'maped_set_corner_extent', payload: { member: 2, corner: 3, extent: 9 } },
  ])
})

test('one extent can be applied to every member at once', async () => {
  await openLoader()
  await inject(BOTH_SOLVED)
  await page.getByTestId('maped-tab-reciprocal').click()
  // Every member shares 4, so the all-members box says 4 rather than guessing.
  expect(await page.getByTestId('maped-corner-extent-all-0').inputValue()).toBe('4')

  await page.getByTestId('maped-corner-extent-all-1').fill('6')
  await expect.poll(() => sentNamed('maped_set_corner_extent')).toEqual([
    { action: 'maped_set_corner_extent', payload: { member: null, corner: 1, extent: 6 } },
  ])

  // Once they disagree, the shared box goes blank instead of showing one
  // member's number as if it were everyone's. (Blurred first: a box being typed
  // in is authoritative over any snapshot, which is the point of the draft.)
  await page.getByTestId('maped-corner-extent-all-1').blur()
  await inject({
    ...BOTH_SOLVED,
    reciprocal: {
      ...BOTH_SOLVED.reciprocal,
      corners: { ...CORNERS, '3': { previews: [0, 1, 2, 3].map((corner) => cornerSum(corner)), extents: [4, 11, 4, 4] } },
    },
  })
  expect(await page.getByTestId('maped-corner-extent-all-1').inputValue()).toBe('')
})

test('the corner panels are there before anything has been computed', async () => {
  await openLoader()
  await inject(REAL_SOLVED)
  await page.getByTestId('maped-tab-reciprocal').click()
  // The extents are INPUTS, so a corner with no sum yet is still a control.
  await expect(page.getByTestId('maped-corner-0-0')).toBeVisible()
  await expect(page.getByTestId('maped-corner-0-0').locator('img')).toHaveCount(0)
  await page.getByTestId('maped-corner-extent-0-0').fill('3')
  await expect.poll(() => sentNamed('maped_set_corner_extent')).toEqual([
    { action: 'maped_set_corner_extent', payload: { member: 0, corner: 0, extent: 3 } },
  ])
})

test('a corner panel enlarges on a double-click, like every other panel', async () => {
  await openLoader()
  await inject(BOTH_SOLVED)
  await page.getByTestId('maped-tab-reciprocal').click()
  await page.getByTestId('maped-corner-5-2').dblclick()
  await expect(page.getByTestId('maped-zoom-caption'))
    .toHaveText('angle05.mrc — Bottom-left corner')
  await expect(page.getByTestId('maped-zoom-image')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('maped-zoom')).toHaveCount(0)
})

test('a reciprocal solve reports itself the same way a real one does', async () => {
  await openLoader()
  await inject(REAL_SOLVED)
  await page.getByTestId('maped-tab-reciprocal').click()
  await expect(page.getByTestId('maped-reciprocal-empty')).toContainText('Not solved')
  await inject(BOTH_SOLVED)
  await expect(page.getByTestId('maped-reciprocal-residual')).toContainText('0.050 px')
  await expect(page.getByTestId('maped-reciprocal-residual-warning')).toHaveCount(0)
})

// ── Opening and closing ────────────────────────────────────────────────────

test('Open is gated on can_commit, and commits without a trailing close', async () => {
  await openLoader()
  await inject(REAL_SOLVED)
  await expect(page.getByTestId('maped-open')).toBeDisabled()

  await inject(BOTH_SOLVED)
  await expect(page.getByTestId('maped-open')).toBeEnabled()
  await page.getByTestId('maped-open').click()

  await expect.poll(() => sentNamed('maped_commit')).toEqual([
    { action: 'maped_commit', payload: {} },
  ])
  await expect(page.getByTestId('multiangle-loader')).toHaveCount(0)
  // A close right behind the commit could tear down the session the commit is
  // still working in.
  expect(await sentNamed('maped_close_loader')).toHaveLength(0)
})

test('cancelling closes the loader session', async () => {
  await openLoader()
  await inject(LOADED)
  await page.getByTestId('maped-cancel').click()
  await expect(page.getByTestId('multiangle-loader')).toHaveCount(0)
  await expect.poll(() => sentNamed('maped_close_loader').then((a) => a.length)).toBe(1)
})

// ── Pixels ─────────────────────────────────────────────────────────────────

/**
 * A green assertion above does not prove the dialog DREW anything — ten members
 * can be present in the DOM and every one of them a grey square, which is the
 * exact failure this redesign exists to avoid. Shots land in
 * electron/multiangle_loader_shots/ (gitignored) and are meant to be looked at.
 */
test('screenshots: every tableau of a full acquisition', async () => {
  const SHOTS = join(__dirname, '..', 'multiangle_loader_shots')
  const shot = (name: string) =>
    page.getByTestId('multiangle-loader').screenshot({ path: join(SHOTS, name) })

  await openLoader()
  await shot('01-empty.png')

  // Scaffolding first: an eight-angle ring with nothing in it yet.
  await page.getByTestId('maped-ring-count').fill('8')
  await page.getByTestId('maped-ring-add').click()
  await shot('02-ring-scaffold.png')

  // …then the members, six of eight spots filled, plus a second shell the data
  // brought with it and two files that are not on any ring.
  await inject({ ...LOADED, members: [...LOADED.members, UNANGLED, FAILED] })
  await shot('03-loaded.png')

  await page.getByTestId('maped-member-4').dblclick()
  await shot('04-zoom.png')
  await page.keyboard.press('Escape')

  await page.getByTestId('maped-tab-real').click()
  await inject(REAL_SOLVED)
  await shot('05-real.png')

  await page.getByTestId('maped-tab-reciprocal').click()
  await inject(BOTH_SOLVED)
  await shot('06-reciprocal.png')

  await page.getByTestId('maped-corner-1-0').dblclick()
  await shot('07-corner-zoom.png')
  await page.keyboard.press('Escape')

  // The MRC fallback: three files that cannot say how they were scanned.
  await page.getByTestId('maped-tab-load').click()
  await inject(NO_SCAN_GRID)
  await shot('08-scan-grid.png')
})
