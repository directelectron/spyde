/**
 * virtual-imaging.ts — guided walkthrough for Virtual Imaging: place a detector
 * over the diffraction pattern and form a real-space image from the signal it
 * integrates at every scan position. Authored once; rendered in-app (tour) and
 * on the docs site. See types.ts.
 */
import type { Guide } from './types'

export const virtualImagingGuide: Guide = {
  id: 'virtual-imaging',
  title: 'Virtual Imaging',
  summary:
    'Place a virtual detector over the diffraction pattern and form a real-space ' +
    'image from what it integrates at every scan position.',
  // Load the small instant Navigation & VI tutorial dataset on open (no
  // download) — a 10×10 scan with clear real-space contrast for a virtual image.
  autoload: {
    action: 'backend', backend: 'tutorial_load', payload: { name: 'navigation' },
    waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1000,
  },
  steps: [
    {
      anchor: null,
      title: 'What you’ll do',
      body:
        'A **virtual image** integrates the diffraction intensity inside a chosen ' +
        'detector region at every scan position, forming a real-space map. Move ' +
        'or resize the detector and the image updates live.\n\n' +
        '> 💡 A small tutorial scan (**Tutorial Data → Navigation & Virtual ' +
        'Imaging**) is loaded for you — no download needed.',
      placement: 'center',
    },
    {
      anchor: 'mdi-area',
      title: 'Start from a diffraction pattern',
      body:
        'The **signal** window shows the pattern under the navigator crosshair. ' +
        'Virtual Imaging lives on this window’s toolbar.',
      placement: 'center',
      image: 'vi-windows.png',
      drive: {
        action: 'backend', backend: 'tutorial_load', payload: { name: 'navigation' },
        waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1500,
      },
      autoDrive: true,
    },
    {
      anchor: 'sub-toolbar',
      title: 'Open the Virtual Imaging tools',
      body:
        'Click **Virtual Imaging** on the toolbar. A sub-toolbar appears where ' +
        'you add and manage detector regions.',
      placement: 'top',
      image: 'vi-subtoolbar.png',
      drive: {
        action: 'click', testid: 'action-btn-Virtual Imaging',
        waitFor: { visible: 'sub-toolbar' },
      },
      autoDrive: true,
    },
    {
      anchor: 'mdi-area',
      title: 'Add a detector → a virtual image',
      body:
        'Add a detector region and a **virtual image** window opens, filled from ' +
        'the intensity it integrates across the scan. Drag or resize the detector ' +
        'on the pattern to update the image live.\n\n' +
        '> 💡 Try it below — drag the green detector over a diffraction spot and ' +
        'watch the scan map light up wherever that spot appears.',
      placement: 'center',
      image: 'vi-output.png',
      drive: {
        action: 'click', testid: 'subaction-add_virtual_image',
        waitFor: { visible: 'vi-icon-Virtual Image 1 (red)' }, timeoutMs: 60_000,
        settleMs: 1500,
      },
      autoDrive: true,
      // Interactive web embed: the vectors explorer's live detector→virtual-image
      // scan, running entirely in the browser. Falls back to the screenshot.
      embed: 'vectors-explorer.html',
    },
  ],
}
