/**
 * orientation.ts — guided walkthrough for dense Orientation Mapping: match a
 * template library against every diffraction pattern, get an IPF orientation
 * map, and see the best-fit template overlaid (green) on the live pattern.
 * Authored once; rendered in-app (tour) and on the docs site. See types.ts.
 *
 * The `drive` blocks let `guide_screenshots.spec.ts` walk this end-to-end and
 * capture a real screenshot per step (it uses the built-in test phase, so no CIF
 * dialog is needed for the screenshot run).
 */
import type { Guide } from './types'

export const orientationGuide: Guide = {
  id: 'orientation',
  title: 'Orientation Mapping',
  summary:
    'Match a simulated template library against a 4D-STEM scan to map crystal ' +
    'orientation, with the best-fit template overlaid live on the pattern.',
  // Load the small instant Orientation tutorial dataset (Si grains) on open —
  // no download.
  autoload: {
    action: 'backend', backend: 'tutorial_load', payload: { name: 'orientation' },
    waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1000,
  },
  info: {
    blurb:
      'Orientation mapping (template matching / ACOM) assigns a crystal ' +
      'orientation to every scan position. A **template library** is simulated ' +
      'from a known phase — one pattern per candidate orientation, sampled over ' +
      'the fundamental zone — and each measured pattern is correlated against ' +
      'the whole library; the best-correlating template wins. Adding a second ' +
      'phase to the library turns the same machinery into **phase mapping**: ' +
      'whichever phase’s templates match best is the phase assigned there.\n\n' +
      'The result is usually shown as an **IPF map**, colouring each position ' +
      'by which crystal direction points along a chosen sample axis, with the ' +
      'correlation score as a confidence map beside it. Two things dominate ' +
      'quality: how finely the library samples orientation space, and how well ' +
      'the pattern centre and camera length are calibrated.\n\n' +
      '> 💡 SpyDE builds the library and matches with **pyxem**, and renders ' +
      'the IPF colouring with **orix**. For EBSD rather than 4D-STEM, ' +
      '**kikuchipy** solves the same problem from Kikuchi patterns.',
    links: [
      {
        label: 'pyxem — Single-phase orientation mapping',
        url: 'https://pyxem.org/v0.21.0/examples/orientation_mapping/single_phase_orientation.html',
        note: 'The reference workflow: simulate a library, match it, read the orientation map.',
      },
      {
        label: 'pyxem — Orientation mapping gallery',
        url: 'https://pyxem.org/v0.21.0/examples/orientation_mapping/index.html',
        note: 'Also covers multi-phase indexing and the on-zone case.',
      },
      {
        label: 'orix — Visualising orientations',
        url: 'https://orix.readthedocs.io/en/stable/examples/plotting/visualizing_orientations.html',
        note: 'What the IPF colouring means, plus axis-angle / Rodrigues / homochoric views of the same data.',
      },
      {
        label: 'orix — Inverse pole density function',
        url: 'https://orix.readthedocs.io/en/stable/examples/inverse_pole_figures/inverse_pole_density_function.html',
        note: 'The density (texture) view behind SpyDE’s IPF “PDF” toggle.',
      },
      {
        label: 'kikuchipy — Pattern matching (dictionary indexing)',
        url: 'https://kikuchipy.org/en/stable/tutorials/pattern_matching.html',
        note: 'The EBSD counterpart: dictionary indexing and orientation refinement.',
      },
    ],
  },
  steps: [
    {
      anchor: null,
      title: 'What you’ll do',
      body:
        'Orientation mapping compares each diffraction pattern against a library ' +
        'of **simulated templates** (one per candidate crystal orientation) and ' +
        'keeps the best match. The result is an **IPF map** colouring every scan ' +
        'position by its crystal orientation.\n\n' +
        '> 💡 A small tutorial scan (**Tutorial Data → Orientation Mapping**, ' +
        'Si grains) is loaded for you — no download needed.',
      placement: 'center',
    },
    {
      anchor: 'mdi-area',
      title: 'Start from a diffraction pattern',
      body:
        'The **signal** window shows the pattern under the navigator crosshair. ' +
        'Orientation Mapping lives on this window’s toolbar.',
      placement: 'center',
      image: 'om-windows.png',
      drive: {
        action: 'backend', backend: 'tutorial_load', payload: { name: 'orientation' },
        waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1500,
      },
      autoDrive: true,
    },
    {
      anchor: 'mdi-area',
      title: 'The IPF orientation map',
      body:
        'Run the match across the scan and an **IPF-Z** orientation map window ' +
        'opens, colouring each scan position by its crystal orientation. The fit ' +
        'also attaches a live overlay to the source pattern.',
      placement: 'center',
      image: 'om-ipf-map.png',
      // Drives the built-in test orientation (no CIF dialog) for the screenshot.
      // NOT autoDrive: matching the whole scan against the template library is
      // the slow stage and can run long — leave it manual, like find-vectors'
      // Compute step, so the tour never appears to hang.
      drive: {
        action: 'backend', backend: 'run_test_orientation',
        waitFor: { subwindows: 3 }, timeoutMs: 180_000,
      },
    },
    {
      anchor: 'mdi-area',
      title: 'The matched template, overlaid live',
      body:
        'The best-fit template’s spots are drawn in **green** on the diffraction ' +
        'pattern, so you can confirm the indexing visually as you move the ' +
        'navigator. The markers sit exactly on the measured Bragg peaks when the ' +
        'orientation is correct.',
      placement: 'center',
      image: 'om-template-overlay.png',
      // Crop to the signal window for a clean close-up. NOTE: the live green
      // overlay render is a known gap (see orientation_workflow.spec.ts fixme),
      // so we do NOT block on green pixels here — just capture the DP window.
      drive: { settleMs: 500, shotTarget: 'subwindow' },
    },
  ],
}
