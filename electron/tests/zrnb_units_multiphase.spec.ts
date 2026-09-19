/**
 * zrnb_units_multiphase.spec.ts — the ZrNb precipitate scan, end to end.
 *
 * This dataset is the reason the reciprocal-units work exists: it ships
 * calibrated in **nm⁻¹**, while diffsims, orix and pyxem's matcher all work in
 * Å⁻¹ and none of them checks. The library was built ten times too large and
 * the orientation map came out as noise, with nothing raised.
 *
 * So the spec proves two things on real pixels, which no headless test can:
 *   1. the detector axes are re-expressed in Å⁻¹ when the file opens, and the
 *      Plot Control dock's units control converts rather than relabels;
 *   2. a TWO-phase vector-orientation run (α-Zr + β-Nb, the phases in the
 *      paper) reaches phase + orientation + strain windows from one fit.
 *
 * Drive shape lifted from ipf_two_window_vom.spec.ts — the file picker is
 * mocked, here with a QUEUE so the wizard can take one .cif per phase.
 */
import { test, expect, _electron as electron, ElectronApplication, Page } from '@playwright/test'
import { join } from 'path'
import { mkdirSync, existsSync } from 'fs'
// Windows are addressed by their breadcrumb prefix (S- signal, N- navigator),
// not by a title: this dataset's windows are "S-ZrNbPercipitate".
const { sigWindow } = require('./_harness.cjs')

const SHOTS = join(__dirname, '..', 'zrnb_shots')
// Fetched from the Crystallography Open Database: 9008523 (α-Zr, P6₃/mmc,
// a=3.232 c=5.147) and 4000948 (β-Nb, Im-3m, a=3.301) — the structures the
// paper identifies.
const CIF_DIR = process.env.ZRNB_CIF_DIR || ''
const ZR_CIF = join(CIF_DIR, 'alpha_Zr_cod9008523.cif')
const NB_CIF = join(CIF_DIR, 'beta_Nb_cod4000948.cif')

let app: ElectronApplication
let page: Page

test.describe.configure({ mode: 'serial' })
test.setTimeout(600_000)

async function shot(name: string) {
  await page.screenshot({ path: join(SHOTS, `${name}.png`) })
}

/** Move a wizard slider. React tracks the input's value and swallows an event
 *  that does not change it, so the native setter has to be called directly —
 *  assigning `.value` leaves the tracker thinking nothing happened and the
 *  onChange never fires. */
async function setSlider(testid: string, value: string) {
  await page.getByTestId(testid).evaluate((el: HTMLInputElement, v: string) => {
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value')!.set!
    setter.call(el, v)
    el.dispatchEvent(new Event('input', { bubbles: true }))
  }, value)
}

// Opt-in: this drives the REAL ZrNb scan (a download through the Examples
// menu) against two .cif files the run has to supply, so it cannot be a CI
// default. Point ZRNB_CIF_DIR at a directory holding both and it runs.
test.skip(!CIF_DIR || !existsSync(ZR_CIF) || !existsSync(NB_CIF),
  'set ZRNB_CIF_DIR to a directory holding alpha_Zr_cod9008523.cif and beta_Nb_cod4000948.cif')

test.beforeAll(async () => {
  test.setTimeout(600_000)
  mkdirSync(SHOTS, { recursive: true })
  app = await electron.launch({
    args: [join(__dirname, '..', 'out', 'main', 'index.js')],
    env: { ...process.env, SPYDE_LOG_LEVEL: 'INFO' },
  })
  let daskReady = false
  app.process().stdout?.on('data', (d: Buffer) => {
    if (String(d).includes('Dask cluster ready')) daskReady = true
  })
  page = await app.firstWindow()
  await page.waitForLoadState('domcontentloaded')
  for (let i = 0; i < 120 && !daskReady; i++) await page.waitForTimeout(500)

  // One .cif per call, in order — the wizard's phase list is built by picking
  // repeatedly, so a picker that always returns the same file yields one phase.
  await app.evaluate(({ ipcMain }, cifs: string[]) => {
    const queue = [...cifs]
    ipcMain.removeHandler('spyde:pick-file')
    ipcMain.handle('spyde:pick-file', async () => queue.shift() ?? cifs[cifs.length - 1])
  }, [ZR_CIF, NB_CIF])

  // The real ZrNb scan from the Examples menu (already downloaded on this box).
  await page.evaluate(() => window.electron.action('load_example', { name: 'ZrNbPrecipitate' }))
  await page.waitForFunction(
    () => document.querySelectorAll('[data-testid="subwindow"]').length >= 2,
    { timeout: 180_000 },
  )
  await page.waitForTimeout(3000)
  await shot('01-loaded')
})

test.afterAll(async () => { await app?.close() })

test('the detector opens in Å⁻¹, not the file\'s nm⁻¹', async () => {
  // The dock's axes table is where a user sees what the detector is calibrated
  // in — the whole point is that it no longer disagrees with what the
  // crystallography does.
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-title').click().catch(() => {})
  await page.waitForTimeout(500)

  const rows = page.getByTestId('axes-table')
  await expect(rows).toBeVisible({ timeout: 30_000 })
  await shot('02-axes-dock')

  const table = await rows.innerText()
  console.log('[axes table]\n' + table)
  // The SCALE is the substantive check — a relabel would have left 5.1e-2
  // (nm⁻¹) sitting under an Å⁻¹ label, which is the failure mode the whole
  // change exists to prevent. 0.05128 nm⁻¹/px is 0.005128 Å⁻¹/px.
  expect(table).toContain('5.1e-3')
  expect(table).not.toContain('5.1e-2')
  expect(table).not.toContain('nm^-1')
  // The SCAN axes are in nm and must be untouched — this converts detector
  // axes, not every axis that happens to have a unit.
  expect(table).toContain('nm')

  // The units cell keeps the stored string in `title` and RENDERS it. KaTeX
  // emits Å as a letter plus a combining ring, and the exponent as its own
  // element, so neither the composed glyph nor "−1" is a contiguous substring
  // — flatten the MathML and compare what a reader actually sees.
  const cell = page.getByTestId('unit-latex').last()
  expect(await cell.getAttribute('title')).toBe('A^-1')
  const rendered = (await cell.innerHTML())
    .replace(/<annotation[^>]*>[\s\S]*?<\/annotation>/, '')
    .replace(/<[^>]+>/g, '')
  console.log('[units cell]', JSON.stringify(rendered))
  expect(rendered).toMatch(/A\s*˚\s*−\s*1|Å\s*−\s*1/)
})

test('the detector-units control converts, it does not relabel', async () => {
  const scaleCell = () => page.getByTestId('axis-2-scale').innerText()

  // Å⁻¹ → nm⁻¹ → mrad → Å⁻¹. Each step must MOVE the scale: a control that
  // only rewrote the label would leave 5.1e-3 under every one of them, which
  // is exactly the lie this replaces.
  expect(await scaleCell()).toBe('5.1e-3')

  const pick = async (unit: string) => {
    await page.getByTestId('detector-units').click()
    await page.getByTestId(`detector-units-opt-${unit}`).click()
    await page.waitForTimeout(1200)
  }

  // The cell rounds to 2 dp above 0.01 and goes exponential below, so these are
  // the RENDERED forms: 0.005128 Å⁻¹/px is 0.05128 nm⁻¹/px → "0.05".
  await pick('nm^-1')
  expect(await scaleCell()).toBe('0.05')
  await shot('02b-units-nm')

  // mrad is a SCATTERING ANGLE, so it needs the electron wavelength and this
  // file records no beam energy — the option says so rather than silently
  // doing nothing or, worse, relabelling.
  await page.getByTestId('detector-units').click()
  const mradLabel = await page.getByTestId('detector-units-opt-mrad').innerText()
  console.log('[mrad option]', mradLabel)
  expect(mradLabel).toContain('needs kV')
  await shot('02c-units-mrad-needs-energy')
  await page.getByTestId('detector-units-opt-mrad').click()
  await page.waitForTimeout(800)
  expect(await scaleCell()).toBe('0.05')          // refused; nothing moved

  // Supply it the way a user would — the Metadata panel's Acc. Volt. cell —
  // and the same option becomes reachable.
  await page.getByTestId('meta-Instrument Metadata-Acc. Volt.').click()
  const input = page.getByTestId('meta-Instrument Metadata-Acc. Volt.-input')
  await input.fill('200')
  await input.press('Enter')
  await page.waitForTimeout(1200)
  await shot('02d-beam-energy-set')

  await pick('mrad')
  // θ = k·λ at 200 kV (λ = 0.025079 Å): 0.005128 × 1000 × 0.025079 = 0.1286.
  expect(await scaleCell()).toBe('0.13')
  await shot('02e-units-mrad')

  await pick('px')
  expect(await scaleCell()).toBe('1.00')
  await shot('02f-units-px')

  // Round trip: nothing was lost on the way through pixels.
  await pick('A^-1')
  expect(await scaleCell()).toBe('5.1e-3')
  await shot('02g-units-back-to-angstrom')
})

test('two-phase vector orientation reaches phase + strain windows', async () => {
  const before = await page.getByTestId('subwindow').count()

  // ── Find Diffraction Vectors ──────────────────────────────────────────────
  const sig = sigWindow(page)
  await sig.getByTestId('subwindow-titlebar').hover()
  await sig.getByTestId('action-btn-Find Diffraction Vectors').click()
  await expect(page.getByTestId('find-vectors-wizard')).toBeVisible({ timeout: 30_000 })

  // These are NBED DISKS about 12 px in radius, ~28 px apart, not small spots:
  // disk cross-correlation at a matched radius finds one peak per disk (~24 a
  // pattern), where the default settings find roughly twice that by splitting
  // disks and picking up background.
  await page.getByTestId('fv-method').click()
  await page.getByTestId('fv-method-opt-nxcorr').click()
  await expect(page.getByTestId('fv-method')).toHaveAttribute('data-value', 'nxcorr')
  for (const [testid, value] of [['fv-radius', '12'], ['fv-mindist', '28'],
                                 ['fv-threshold', '0.3']] as const) {
    await setSlider(testid, value)
  }
  await page.waitForTimeout(1500)     // let the debounced live preview settle
  await shot('03-find-vectors-wizard')

  await page.getByTestId('fv-compute').click()
  await expect(page.getByTestId('status-text'))
    .toContainText('Found', { timeout: 300_000 })
  await page.waitForTimeout(2000)
  await shot('04-vectors-found')

  // ── Vector Orientation Mapping, two phases ────────────────────────────────
  // The SIGNAL vectors window ("S-… — Vectors"), not the navigator count map
  // that precedes it in the DOM — the vector actions live on the signal side.
  const vecWin = page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /^S-.*Vectors/ }),
  }).first()
  await vecWin.getByTestId('subwindow-titlebar').hover()
  await vecWin.getByTestId('action-btn-Vector Orientation Mapping').click()
  await expect(page.getByTestId('vector-orientation-wizard')).toBeVisible({ timeout: 30_000 })

  // Two phases, each given a structure from the (queued) file picker.
  await page.getByTestId('vom-add-phase').click()
  await expect(page.getByTestId('periodic-table')).toBeVisible()
  await expect(page.getByTestId('phase-row-0')).toBeVisible()
  await page.getByTestId('phase-0-cif').click()
  await expect(page.getByTestId('phase-0-structure')).toContainText('alpha_Zr')
  await page.getByTestId('ptable-add-phase').click()
  await page.getByTestId('phase-1-cif').click()
  await expect(page.getByTestId('phase-1-structure')).toContainText('beta_Nb')
  await page.getByTestId('ptable-done').click()
  const list = await page.getByTestId('vom-cif-list').innerText()
  console.log('[phase list]\n' + list)
  expect(list).toContain('Zr')
  expect(list).toContain('Nb')
  await shot('05-two-phases-loaded')

  await page.getByTestId('vom-tab-Library').click()
  await page.getByTestId('vom-generate').click()
  // "library ready" names the phases, but it is TRANSIENT — generating the
  // library rolls straight on into the whole-field fit, so polling for it is a
  // race. Wait for the settled end of that sequence instead; reaching it at all
  // means the two-phase library built and fitted.
  await expect(page.getByTestId('status-text'))
    .toContainText(/live IPF ready|library ready/, { timeout: 420_000 })
  await page.waitForTimeout(3000)
  await shot('06-library-ready')

  // The live IPF window is what Generate opens, and it is the orientation half
  // of "structure and strain in one go".
  const titlesAfterLibrary = await page.getByTestId('subwindow-title').allInnerTexts()
  console.log('[windows after library]\n' + titlesAfterLibrary.join('\n'))
  expect(titlesAfterLibrary.join(' ')).toMatch(/IPF/)

  // ── Compute Maps ──────────────────────────────────────────────────────────
  await page.getByTestId('vom-tab-Run').click()
  await page.getByTestId('vom-compute').click()
  await expect(page.getByTestId('status-text'))
    .toContainText('complete', { timeout: 300_000 })
  await page.waitForTimeout(4000)
  await shot('07-maps-computed')

  const titles = await page.getByTestId('subwindow-title').allInnerTexts()
  console.log('[windows]\n' + titles.join('\n'))
  expect(await page.getByTestId('subwindow').count()).toBeGreaterThan(before)
  expect(titles.join(' ')).toMatch(/Strain/)
  expect(titles.join(' ')).toMatch(/Phase/)

  // Tiled, so every result is visible at once rather than stacked — this is the
  // frame a person actually reads the analysis off.
  await page.getByTestId('tile-windows').click()
  await page.waitForTimeout(2500)
  await shot('08-tiled')

  // Each strain component, so the maps can be compared against the platelet.
  const strain = page.getByTestId('subwindow').filter({
    has: page.getByTestId('window-breadcrumb').filter({ hasText: /Strain/ }),
  }).first()
  for (const component of ['εyy', 'εxy']) {
    const chip = strain.getByText(component, { exact: true })
    if (await chip.count()) {
      await chip.first().click()
      await page.waitForTimeout(1200)
      await shot(`09-strain-${component === 'εyy' ? 'eyy' : 'exy'}`)
    }
  }
})
