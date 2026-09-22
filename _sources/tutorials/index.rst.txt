..
   GENERATED FILE — do not edit by hand.
   Regenerate with: node scripts/gen_guide_docs.mjs

.. _tutorials-index:

Tutorials
=========

Click-by-click walkthroughs of a complete technique in the SpyDE interface.
Each one is generated from the same source as the guided tour built into the
app, so what you read here is exactly what the app shows you — open one in
SpyDE from **Help → <technique> → Guided tour**, or read it through here
first.

Every tutorial is self-contained: the in-app tour loads its own small example
dataset (no download, a couple of seconds), and closes it again when you
finish. Each one ends with **More information** — background on the technique
and links to the upstream documentation.

.. toctree::
   :maxdepth: 1

   welcome
   find-vectors
   virtual-imaging
   orientation
   strain
   spectroscopy

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Tutorial
     - What it covers
   * - :ref:`First Steps <tutorial-welcome>`
     - A quick orientation to SpyDE: the navigator and signal windows, the linked crosshair, per-window toolbars, and the Plot Control dock.
   * - :ref:`Finding Diffraction Vectors <tutorial-find-vectors>`
     - Detect Bragg peaks across a 4D-STEM scan and overlay the found vectors on the live diffraction pattern.
   * - :ref:`Virtual Imaging <tutorial-virtual-imaging>`
     - Place a virtual detector over the diffraction pattern and form a real-space image from what it integrates at every scan position.
   * - :ref:`Orientation Mapping <tutorial-orientation>`
     - Match a simulated template library against a 4D-STEM scan to map crystal orientation, with the best-fit template overlaid live on the pattern.
   * - :ref:`Strain Mapping <tutorial-strain>`
     - Measure lattice distortion from diffraction-disk positions relative to a reference region, and view it as εxx/εyy/εxy/rotation component maps.
   * - :ref:`1D Spectroscopy <tutorial-spectroscopy>`
     - Navigate a map of per-pixel spectra and watch the spectrum change live under the crosshair — the basic EELS/EDS spectrum-imaging workflow.

