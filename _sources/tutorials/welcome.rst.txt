..
   GENERATED FILE — do not edit by hand.
   Source: guides/welcome.ts (the same walkthrough the in-app
   guided tour renders). Regenerate with:
       node scripts/gen_guide_docs.mjs

.. _tutorial-welcome:

First Steps
===========

A quick orientation to SpyDE: the navigator and signal windows, the linked
crosshair, per-window toolbars, and the Plot Control dock.

.. admonition:: Follow along in the app
   :class: note

   Every step below is also a live walkthrough inside SpyDE:
   **Help → First Steps → Guided tour**. The tour loads the same small
   tutorial dataset for you (no download), highlights each control as you
   go, and closes the example data again when you exit.

Steps
-----

1. Welcome to SpyDE
~~~~~~~~~~~~~~~~~~~

SpyDE visualizes and analyzes electron microscopy data — TEM, STEM, Cryo EM,
4D-STEM, EELS. You work with **windows**: a navigator shows the scan, a signal
window shows the pattern or spectrum at the crosshair, and toolbars on each
window run analyses.

.. tip::

   A small tutorial scan (**Tutorial Data → Navigation & Virtual Imaging**) is
   loaded for you — no download needed.

2. Two linked windows
~~~~~~~~~~~~~~~~~~~~~

The **navigator** (left) shows the scan grid with a crosshair; the **signal**
window (right) shows the diffraction pattern at that crosshair position. Every
dataset you open works this way.

3. Move the crosshair
~~~~~~~~~~~~~~~~~~~~~

Drag the crosshair on the navigator — the signal window updates live to show
the pattern at the new scan position. Try it now.

4. Hover a window for its toolbar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hover any window to reveal its **floating toolbar** — the tools that act on
that window (Find Vectors, Virtual Imaging, FFT, and more all live here,
depending on the data).

5. The Plot Control dock
~~~~~~~~~~~~~~~~~~~~~~~~

The dock on the right shows the **contrast histogram**, axes, signal-tree, and
metadata for whichever window is active — your control panel for the current
plot.

6. Where to go next
~~~~~~~~~~~~~~~~~~~

Ready for a real workflow? Open **Help → Virtual Imaging** or **Help → Finding
Diffraction Vectors** for a guided walkthrough on its own tutorial dataset.

More information
----------------

SpyDE is a desktop front end for the Python electron-microscopy stack:
**HyperSpy** for the multidimensional signal model and lazy/out-of-core
loading, **pyxem** for the 4D-STEM methods, **orix** for crystal orientations.
Everything you do in the interface is a call into those libraries, so an
analysis you build here has a direct equivalent in a notebook — and vice
versa.

The navigator/signal window pair is HyperSpy’s own idea of navigation and
signal axes made interactive, which is why the same layout appears whether the
signal is a diffraction pattern, an image or a spectrum.

.. tip::

   Pick a technique from the Help menu for a walkthrough of a specific
   workflow.

Further reading
~~~~~~~~~~~~~~~

SpyDE wraps pyxem, HyperSpy, eXSpy, kikuchipy and orix; those projects
document the underlying methods in far more depth than a walkthrough can.

* `HyperSpy — Data visualisation <https://hyperspy.org/hyperspy-doc/current/user_guide/visualisation.html>`_

  Navigation and signal axes, customising the navigator, plotting several
  signals together.

* `HyperSpy — User guide <https://hyperspy.org/hyperspy-doc/current/user_guide/index.html>`_

  The library SpyDE is built on: signals, axes, regions of interest, lazy
  big-data handling.

* `pyxem — Example gallery <https://pyxem.org/v0.21.0/examples/index.html>`_

  Worked 4D-STEM examples for every technique SpyDE exposes.


