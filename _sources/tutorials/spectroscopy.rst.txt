..
   GENERATED FILE — do not edit by hand.
   Source: guides/spectroscopy.ts (the same walkthrough the in-app
   guided tour renders). Regenerate with:
       node scripts/gen_guide_docs.mjs

.. _tutorial-spectroscopy:

1D Spectroscopy
===============

Navigate a map of per-pixel spectra and watch the spectrum change live under
the crosshair — the basic EELS/EDS spectrum-imaging workflow.

.. admonition:: Follow along in the app
   :class: note

   Every step below is also a live walkthrough inside SpyDE:
   **Help → 1D Spectroscopy → Guided tour**. The tour loads the same small
   tutorial dataset for you (no download), highlights each control as you
   go, and closes the example data again when you exit.

Steps
-----

1. What you’ll do
~~~~~~~~~~~~~~~~~

Spectroscopy data (EELS, EDS) pairs a **spectrum** — intensity per energy
channel — with every position in a scan. SpyDE shows the same navigator +
linked-signal layout as imaging data, except the signal window is a **1D
spectrum plot** instead of a 2D pattern.

.. tip::

   A small tutorial map (**Tutorial Data → Spectroscopy**, two Gaussian peaks
   whose position/width vary per pixel) is loaded for you — no download
   needed.

2. Navigator + spectrum window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **navigator** (left) shows the 32×32 scan grid; the **signal** window
(right) plots the spectrum — intensity vs. channel — at the crosshair
position.

3. Move the crosshair, watch the spectrum change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drag the crosshair across the navigator — the two peaks in the spectrum window
shift and change height as you cross the map, since each pixel carries its own
peak position and width.

4. The plot toolbar
~~~~~~~~~~~~~~~~~~~

Hover the spectrum window to reveal its floating toolbar — **Zoom**,
**Reset**, and **Add Selector** (to place an integration region) work the same
way here as on any 2D plot.

5. Reading the axes
~~~~~~~~~~~~~~~~~~~

The Plot Control dock shows the spectrum’s channel axis and intensity scale
for the active window — the same dock used for every plot in SpyDE.

More information
----------------

A **spectrum image** stores a full spectrum — EELS or EDS — at every position
of a scan. The data has the same navigator/signal shape as 4D-STEM, only the
signal is one-dimensional: navigating the map plays the spectrum back position
by position, and integrating a real-space region averages spectra to trade
spatial resolution for signal-to-noise.

The usual analysis is quantitative rather than visual: subtract a background
(a power law before an EELS edge, a bremsstrahlung model under EDS lines), fit
a model of components to the remaining signal, and map a fitted parameter — an
edge intensity, a peak position, a composition — back over the scan.

.. tip::

   SpyDE reads and displays this data through **HyperSpy**; the quantitative
   EELS/EDS methods live in **eXSpy**, HyperSpy’s spectroscopy extension.

Further reading
~~~~~~~~~~~~~~~

SpyDE wraps pyxem, HyperSpy, eXSpy, kikuchipy and orix; those projects
document the underlying methods in far more depth than a walkthrough can.

* `eXSpy — EELS user guide <https://hyperspy.org/exspy/user_guide/eels.html>`_

  Thickness, zero-loss alignment, deconvolution, Kramers-Kronig analysis and
  EELS curve fitting.

* `eXSpy — EDS user guide <https://hyperspy.org/exspy/user_guide/eds.html>`_

  Background subtraction, line fitting and quantification for
  energy-dispersive X-ray data.

* `eXSpy — EELS curve fitting example <https://hyperspy.org/exspy/auto_examples/model_fitting/EELS_curve_fitting.html>`_

  A complete worked example: load, set microscope parameters, build a model,
  fit, plot.

* `HyperSpy — Signal1D tools <https://hyperspy.org/hyperspy-doc/current/user_guide/signal1d.html>`_

  Background removal, smoothing, peak finding and spectrum alignment.

* `HyperSpy — Model fitting <https://hyperspy.org/hyperspy-doc/current/user_guide/model/index.html>`_

  Components, fitting strategies and fitting a model across a whole spectrum
  image.


