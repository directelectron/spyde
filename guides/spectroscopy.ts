/**
 * spectroscopy.ts — guided walkthrough for 1D spectroscopy: navigate a map of
 * per-pixel spectra (EELS/EDS-like) and watch the spectrum change under the
 * crosshair. Authored once; rendered in-app (tour) and on the docs site. See
 * guides/types.ts.
 *
 * Unlike the 4D-STEM guides, `tutorial_spectroscopy` (hs.data.two_gaussians,
 * a 32x32-probe Signal1D) opens a navigator + a 1D SPECTRUM window, not a
 * diffraction pattern. There is no dedicated 1D peak/region-fitting toolbar
 * action in toolbars.yaml today (the only plot_dim:[1] entries are the
 * generic Reset/Zoom/Add Selector and the insitu-only Play/Fast Forward) — so
 * this guide stays focused on navigation + reading the spectrum rather than
 * pointing at a tool that doesn't exist yet.
 */
import type { Guide } from './types'

export const spectroscopyGuide: Guide = {
  id: 'spectroscopy',
  title: '1D Spectroscopy',
  summary:
    'Navigate a map of per-pixel spectra and watch the spectrum change live ' +
    'under the crosshair — the basic EELS/EDS spectrum-imaging workflow.',
  // Load the small instant Spectroscopy tutorial dataset on open — no
  // download. Opens a navigator + a 1D spectrum window (not a 2D pattern).
  autoload: {
    action: 'backend', backend: 'tutorial_load', payload: { name: 'spectroscopy' },
    waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1000,
  },
  info: {
    blurb:
      'A **spectrum image** stores a full spectrum — EELS or EDS — at every ' +
      'position of a scan. The data has the same navigator/signal shape as ' +
      '4D-STEM, only the signal is one-dimensional: navigating the map plays ' +
      'the spectrum back position by position, and integrating a real-space ' +
      'region averages spectra to trade spatial resolution for signal-to-noise.\n\n' +
      'The usual analysis is quantitative rather than visual: subtract a ' +
      'background (a power law before an EELS edge, a bremsstrahlung model ' +
      'under EDS lines), fit a model of components to the remaining signal, and ' +
      'map a fitted parameter — an edge intensity, a peak position, a ' +
      'composition — back over the scan.\n\n' +
      '> 💡 SpyDE reads and displays this data through **HyperSpy**; the ' +
      'quantitative EELS/EDS methods live in **eXSpy**, HyperSpy’s ' +
      'spectroscopy extension.',
    links: [
      {
        label: 'eXSpy — EELS user guide',
        url: 'https://hyperspy.org/exspy/user_guide/eels.html',
        note: 'Thickness, zero-loss alignment, deconvolution, Kramers-Kronig analysis and EELS curve fitting.',
      },
      {
        label: 'eXSpy — EDS user guide',
        url: 'https://hyperspy.org/exspy/user_guide/eds.html',
        note: 'Background subtraction, line fitting and quantification for energy-dispersive X-ray data.',
      },
      {
        label: 'eXSpy — EELS curve fitting example',
        url: 'https://hyperspy.org/exspy/auto_examples/model_fitting/EELS_curve_fitting.html',
        note: 'A complete worked example: load, set microscope parameters, build a model, fit, plot.',
      },
      {
        label: 'HyperSpy — Signal1D tools',
        url: 'https://hyperspy.org/hyperspy-doc/current/user_guide/signal1d.html',
        note: 'Background removal, smoothing, peak finding and spectrum alignment.',
      },
      {
        label: 'HyperSpy — Model fitting',
        url: 'https://hyperspy.org/hyperspy-doc/current/user_guide/model/index.html',
        note: 'Components, fitting strategies and fitting a model across a whole spectrum image.',
      },
    ],
  },
  steps: [
    {
      anchor: null,
      title: 'What you’ll do',
      body:
        'Spectroscopy data (EELS, EDS) pairs a **spectrum** — intensity per ' +
        'energy channel — with every position in a scan. SpyDE shows the same ' +
        'navigator + linked-signal layout as imaging data, except the signal ' +
        'window is a **1D spectrum plot** instead of a 2D pattern.\n\n' +
        '> 💡 A small tutorial map (**Tutorial Data → Spectroscopy**, two ' +
        'Gaussian peaks whose position/width vary per pixel) is loaded for ' +
        'you — no download needed.',
      placement: 'center',
    },
    {
      anchor: 'mdi-area',
      title: 'Navigator + spectrum window',
      body:
        'The **navigator** (left) shows the 32×32 scan grid; the **signal** ' +
        'window (right) plots the spectrum — intensity vs. channel — at the ' +
        'crosshair position.',
      placement: 'center',
      image: 'spectroscopy-windows.png',
      drive: {
        action: 'backend', backend: 'tutorial_load', payload: { name: 'spectroscopy' },
        waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1500,
      },
      autoDrive: true,
    },
    {
      anchor: 'mdi-area',
      title: 'Move the crosshair, watch the spectrum change',
      body:
        'Drag the crosshair across the navigator — the two peaks in the ' +
        'spectrum window shift and change height as you cross the map, since ' +
        'each pixel carries its own peak position and width.',
      placement: 'center',
      image: 'spectroscopy-crosshair.png',
      // Manual: dragging to a meaningful new position and watching the
      // spectrum respond is the thing to discover by doing, same as the
      // Welcome guide's crosshair step.
    },
    {
      anchor: 'subwindow-titlebar',
      title: 'The plot toolbar',
      body:
        'Hover the spectrum window to reveal its floating toolbar — **Zoom**, ' +
        '**Reset**, and **Add Selector** (to place an integration region) work ' +
        'the same way here as on any 2D plot.',
      placement: 'top',
      image: 'spectroscopy-toolbar.png',
      drive: { action: 'hover', testid: 'subwindow-titlebar' },
      autoDrive: true,
    },
    {
      anchor: 'plot-control-dock',
      title: 'Reading the axes',
      body:
        'The Plot Control dock shows the spectrum’s channel axis and ' +
        'intensity scale for the active window — the same dock used for every ' +
        'plot in SpyDE.',
      placement: 'left',
      image: 'spectroscopy-dock.png',
    },
  ],
}
