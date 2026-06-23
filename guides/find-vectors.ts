/**
 * find-vectors.ts — guided walkthrough for finding diffraction vectors (peaks)
 * in a 4D-STEM dataset. Authored once; rendered in-app (coachmark tour) and on
 * the docs website. See guides/types.ts.
 */
import type { Guide } from './types'

export const findVectorsGuide: Guide = {
  id: 'find-vectors',
  title: 'Finding Diffraction Vectors',
  summary:
    'Detect Bragg peaks across a 4D-STEM scan and overlay the found vectors on ' +
    'the live diffraction pattern.',
  steps: [
    {
      anchor: null,
      title: 'What you’ll do',
      body:
        'Diffraction-vector finding locates the Bragg disks in **every** ' +
        'diffraction pattern of a 4D-STEM scan. The result is a sparse set of ' +
        'peaks per scan position — the input to virtual imaging, strain, and ' +
        'orientation mapping.\n\n' +
        '> 💡 You’ll need a 4D dataset open. Use **Examples → sped_ag** for a ' +
        'real scan to follow along.',
      placement: 'center',
    },
    {
      anchor: 'mdi-area',
      title: 'The two linked windows',
      body:
        'Opening a 4D dataset gives you a **navigator** (the scan grid) and a ' +
        '**signal** window (the diffraction pattern at the crosshair). Moving ' +
        'the crosshair on the navigator updates the pattern live.',
      placement: 'center',
      image: 'mdi-two-windows.png',
      // Screenshot setup: load the dataset and wait for both windows.
      drive: {
        action: 'backend', backend: 'load_test_data_si_grains',
        waitFor: { subwindows: 2 }, timeoutMs: 120_000, settleMs: 1500,
      },
    },
    {
      anchor: 'floating-toolbar',
      title: 'The plot toolbar',
      body:
        'Hover the diffraction-pattern window to reveal its floating toolbar. ' +
        'Tools that act on the signal — FFT, Center Zero Beam, Find Vectors — ' +
        'live here.',
      placement: 'top',
      image: 'floating-toolbar.png',
      // Reveal the toolbar by hovering the signal window's titlebar.
      drive: { action: 'hover', testid: 'subwindow-titlebar' },
    },
    {
      anchor: 'action-btn-Find Diffraction Vectors',
      title: 'Open Find Diffraction Vectors',
      body:
        'Click the peak-finding tool to open its **wizard**. It opens with a ' +
        'live preview running on the pattern under the crosshair, so you can ' +
        'tune parameters and see the detected peaks immediately.',
      placement: 'top',
      image: 'find-vectors-button.png',
      drive: {
        action: 'click', testid: 'action-btn-Find Diffraction Vectors',
        waitFor: { visible: 'find-vectors-wizard' },
      },
    },
    {
      anchor: 'find-vectors-wizard',
      title: 'Tune the detection',
      body:
        'Adjust **σ** (Gaussian blur before detection) and the **threshold** ' +
        '(minimum peak strength). Red markers update live on the pattern as you ' +
        'drag the sliders.\n\n' +
        '> 💡 Start with a high threshold and lower it until real disks are ' +
        'marked but noise is not.',
      placement: 'left',
      image: 'find-vectors-wizard.png',
      // Wait for the live preview to actually mark peaks (red), then shoot.
      drive: { waitFor: { pixels: 'red' }, timeoutMs: 30_000 },
    },
    {
      anchor: 'fv-compute',
      title: 'Compute across the whole scan',
      body:
        'Happy with the preview? Click **Compute** to run detection on every ' +
        'scan position. Progress streams in the status bar; the found vectors ' +
        'are then overlaid on the live pattern and become a new node in the ' +
        'signal tree.',
      placement: 'top',
      image: 'find-vectors-compute.png',
      // Capture the wizard at the moment Compute is pressed (status bar shows
      // progress). We do NOT wait for the full-scan compute to finish here — it
      // is the slow stage and can run long; the result window is asserted by
      // find_vectors_workflow.spec.ts instead.
      drive: { action: 'click', testid: 'fv-compute', settleMs: 1500 },
    },
    {
      anchor: 'status-text',
      title: 'Done',
      body:
        'When the status bar reports completion, the diffraction vectors are ' +
        'ready. From here you can run **Vector Virtual Imaging** or **Vector ' +
        'Orientation Mapping** on them.',
      placement: 'top',
      image: 'find-vectors-done.png',
    },
  ],
}
