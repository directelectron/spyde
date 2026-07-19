/**
 * welcome.ts — the first-run "First Steps" walkthrough: what SpyDE is, the
 * navigator/signal window pair, the linked crosshair, hovering for a floating
 * toolbar, and the Plot Control dock. Authored once; rendered in-app (tour)
 * and on the docs site. See guides/types.ts.
 *
 * Deliberately short (5-6 steps) — this is the very first thing a new user
 * sees (Phase 4 auto-opens it on first run), so it teaches orientation, not a
 * workflow. Point elsewhere (Virtual Imaging / Find Vectors) for the next step.
 */
import type { Guide } from './types'

export const welcomeGuide: Guide = {
  id: 'welcome',
  title: 'First Steps',
  summary:
    'A quick orientation to SpyDE: the navigator and signal windows, the ' +
    'linked crosshair, per-window toolbars, and the Plot Control dock.',
  // Load the small instant Navigation & VI tutorial dataset on open — no
  // download — so the first thing a new user sees is real data, not a blank
  // canvas.
  autoload: {
    action: 'backend', backend: 'tutorial_load', payload: { name: 'navigation' },
    waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1000,
  },
  steps: [
    {
      anchor: null,
      title: 'Welcome to SpyDE',
      body:
        'SpyDE visualizes and analyzes electron microscopy data — TEM, STEM, ' +
        'Cryo EM, 4D-STEM, EELS. You work with **windows**: a navigator shows ' +
        'the scan, a signal window shows the pattern or spectrum at the ' +
        'crosshair, and toolbars on each window run analyses.\n\n' +
        '> 💡 A small tutorial scan (**Tutorial Data → Navigation & Virtual ' +
        'Imaging**) is loaded for you — no download needed.',
      placement: 'center',
    },
    {
      anchor: 'mdi-area',
      title: 'Two linked windows',
      body:
        'The **navigator** (left) shows the scan grid with a crosshair; the ' +
        '**signal** window (right) shows the diffraction pattern at that ' +
        'crosshair position. Every dataset you open works this way.',
      placement: 'center',
      image: 'welcome-windows.png',
      drive: {
        action: 'backend', backend: 'tutorial_load', payload: { name: 'navigation' },
        waitFor: { subwindows: 2 }, timeoutMs: 60_000, settleMs: 1500,
      },
      autoDrive: true,
    },
    {
      anchor: 'mdi-area',
      title: 'Move the crosshair',
      body:
        'Drag the crosshair on the navigator — the signal window updates ' +
        'live to show the pattern at the new scan position. Try it now.',
      placement: 'center',
      image: 'welcome-crosshair.png',
      // Manual: dragging the crosshair to a meaningful new position is exactly
      // the thing we want the user to discover by doing, not something a
      // scripted click communicates well.
    },
    {
      anchor: 'subwindow-titlebar',
      title: 'Hover a window for its toolbar',
      body:
        'Hover any window to reveal its **floating toolbar** — the tools that ' +
        'act on that window (Find Vectors, Virtual Imaging, FFT, and more all ' +
        'live here, depending on the data).',
      placement: 'top',
      image: 'welcome-toolbar.png',
      drive: { action: 'hover', testid: 'subwindow-titlebar' },
      autoDrive: true,
    },
    {
      anchor: 'plot-control-dock',
      title: 'The Plot Control dock',
      body:
        'The dock on the right shows the **contrast histogram**, axes, ' +
        'signal-tree, and metadata for whichever window is active — your ' +
        'control panel for the current plot.',
      placement: 'left',
      image: 'welcome-dock.png',
    },
    {
      anchor: null,
      title: 'Where to go next',
      body:
        'Ready for a real workflow? Open **Help → Virtual Imaging** or ' +
        '**Help → Finding Diffraction Vectors** for a guided walkthrough on ' +
        'its own tutorial dataset.',
      placement: 'center',
    },
  ],
}
