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
  steps: [
    {
      anchor: null,
      title: 'What you’ll do',
      body:
        'Orientation mapping compares each diffraction pattern against a library ' +
        'of **simulated templates** (one per candidate crystal orientation) and ' +
        'keeps the best match. The result is an **IPF map** colouring every scan ' +
        'position by its crystal orientation.\n\n' +
        '> 💡 Open a 4D dataset first — **Examples → sped_ag** is a good real scan.',
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
        action: 'backend', backend: 'load_test_data_si_grains',
        waitFor: { subwindows: 2 }, timeoutMs: 120_000, settleMs: 1500,
      },
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
