/**
 * strain.ts — guided walkthrough for Strain Mapping: measure lattice
 * distortion from diffraction-disk positions relative to a reference region,
 * producing εxx/εyy/εxy/ω component maps. Authored once; rendered in-app
 * (tour) and on the docs site. See guides/types.ts.
 *
 * Strain Mapping (`spyde.actions.strain_action.strain_mapping`, toolbars.yaml)
 * is `requires_vectors: True` / `signal_types: [spyde_diffraction_vectors_image]`
 * — it only appears on a Find-Vectors RESULT window's toolbar, and self-waits
 * (`wait_for_vectors`) if vectors haven't finished attaching yet. So this guide
 * runs Find Vectors first, exactly like find-vectors.ts, then points at the
 * Strain Mapping button that appears on that result window.
 */
import type { Guide } from './types'

export const strainGuide: Guide = {
  id: 'strain',
  title: 'Strain Mapping',
  summary:
    'Measure lattice distortion from diffraction-disk positions relative to a ' +
    'reference region, and view it as εxx/εyy/εxy/rotation component maps.',
  // Load the small instant Strain tutorial dataset (downsized simulated_strain,
  // a strained precipitate) on open — no download.
  autoload: {
    action: 'backend', backend: 'tutorial_load', payload: { name: 'strain' },
    waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1000,
  },
  steps: [
    {
      anchor: null,
      title: 'What you’ll do',
      body:
        'Strain mapping measures how far each diffraction pattern’s Bragg ' +
        'disks have shifted from an **unstrained reference region**, and fits ' +
        'that shift to a local lattice distortion at every scan position.\n\n' +
        '> 💡 A small tutorial scan (**Tutorial Data → Strain Mapping**, a ' +
        'strained precipitate) is loaded for you — no download needed.',
      placement: 'center',
    },
    {
      anchor: 'mdi-area',
      title: 'Start from a diffraction pattern',
      body:
        'Strain mapping is computed **from diffraction vectors** — the Bragg ' +
        'peaks found in each pattern — so we first run Find Diffraction ' +
        'Vectors, the same as the Finding Diffraction Vectors walkthrough.',
      placement: 'center',
      image: 'strain-windows.png',
      drive: {
        action: 'backend', backend: 'tutorial_load', payload: { name: 'strain' },
        waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1500,
      },
      autoDrive: true,
    },
    {
      anchor: 'floating-toolbar',
      title: 'The plot toolbar',
      body:
        'Hover the diffraction-pattern window to reveal its floating toolbar, ' +
        'where **Find Diffraction Vectors** lives.',
      placement: 'top',
      image: 'strain-floating-toolbar.png',
      drive: { action: 'hover', testid: 'subwindow-titlebar' },
      autoDrive: true,
    },
    {
      anchor: 'action-btn-Find Diffraction Vectors',
      title: 'Find the diffraction vectors first',
      body:
        'Click **Find Diffraction Vectors** to open its wizard, tune the ' +
        'detection on the live preview, then **Compute** across the whole ' +
        'scan — same as the Finding Diffraction Vectors walkthrough.\n\n' +
        '> 💡 This is the slow step (it runs on every scan position) — give ' +
        'it a minute on a real scan.',
      placement: 'top',
      image: 'strain-find-vectors-button.png',
      drive: {
        action: 'click', testid: 'action-btn-Find Diffraction Vectors',
        waitFor: { visible: 'find-vectors-wizard' },
      },
      autoDrive: true,
    },
    {
      anchor: 'fv-compute',
      title: 'Compute the vectors',
      body:
        'Click **Compute** to detect peaks across the whole scan. Once it ' +
        'finishes, the result window’s toolbar gains a **Strain Mapping** ' +
        'button — it only appears once vectors exist.',
      placement: 'top',
      image: 'strain-compute-vectors.png',
      // Heavy full-scan compute — never autoDrive (mirrors find-vectors.ts's
      // Compute step). Left manual/centered-style so the tour never appears
      // to hang.
    },
    {
      anchor: 'action-btn-Strain Mapping',
      title: 'Open Strain Mapping',
      body:
        'Click **Strain Mapping** on the vectors result window. It opens a ' +
        'strain-map window plus a dedicated **cyan reference crosshair** — ' +
        'drag it to an unstrained region of the scan and the whole field ' +
        'recomputes live.',
      placement: 'top',
      image: 'strain-open.png',
      // The button only exists once Find Vectors has finished (previous step),
      // and the resulting extra windows (reference crosshair + strain map)
      // make an exact subwindow count fragile to predict here — left manual,
      // same reasoning as the compute step above.
    },
    {
      anchor: null,
      title: 'Reading the component maps',
      body:
        'Toggle between **εxx**, **εyy**, **εxy** (shear), and **ω** ' +
        '(rotation) to see each strain component. Double-click a spot in the ' +
        'reference window to include/exclude it from the fit, and use ' +
        '**Submit** to freeze the current field as a new result.',
      placement: 'center',
    },
  ],
}
