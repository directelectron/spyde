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
    },
    {
      anchor: 'status-text',
      title: 'Done',
      body:
        'When the status bar reports completion, the diffraction vectors are ' +
        'ready. From here you can run **Vector Virtual Imaging** or **Vector ' +
        'Orientation Mapping** on them.',
      placement: 'top',
    },
  ],
}
