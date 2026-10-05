..
   GENERATED FILE — do not edit by hand.
   Source: guides/strain.ts (the same walkthrough the in-app
   guided tour renders). Regenerate with:
       node scripts/gen_guide_docs.mjs

.. _tutorial-strain:

Strain Mapping
==============

Measure lattice distortion from diffraction-disk positions relative to a
reference region, and view it as εxx/εyy/εxy/rotation component maps.

.. admonition:: Follow along in the app
   :class: note

   Every step below is also a live walkthrough inside SpyDE:
   **Help → Strain Mapping → Guided tour**. The tour loads the same small
   tutorial dataset for you (no download), highlights each control as you
   go, and closes the example data again when you exit.

Steps
-----

1. What you’ll do
~~~~~~~~~~~~~~~~~

Strain mapping measures how far each diffraction pattern’s Bragg disks have
shifted from an **unstrained reference region**, and fits that shift to a
local lattice distortion at every scan position.

.. tip::

   A small tutorial scan (**Tutorial Data → Strain Mapping**, a strained
   precipitate) is loaded for you — no download needed.

2. Start from a diffraction pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strain mapping is computed **from diffraction vectors** — the Bragg peaks
found in each pattern — so we first run Find Diffraction Vectors, the same as
the Finding Diffraction Vectors walkthrough.

3. The plot toolbar
~~~~~~~~~~~~~~~~~~~

Hover the diffraction-pattern window to reveal its floating toolbar, where
**Find Diffraction Vectors** lives.

4. Find the diffraction vectors first
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click **Find Diffraction Vectors** to open its wizard, tune the detection on
the live preview, then **Compute** across the whole scan — same as the Finding
Diffraction Vectors walkthrough.

.. tip::

   This is the slow step (it runs on every scan position) — give it a minute
   on a real scan.

5. Compute the vectors
~~~~~~~~~~~~~~~~~~~~~~

Click **Compute** to detect peaks across the whole scan. Once it finishes, the
result window’s toolbar gains a **Strain Mapping** button — it only appears
once vectors exist.

6. Open Strain Mapping
~~~~~~~~~~~~~~~~~~~~~~

Click **Strain Mapping** on the vectors result window. It opens a strain-map
window plus a dedicated **cyan reference crosshair** — drag it to an
unstrained region of the scan and the whole field recomputes live.

7. Reading the component maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Toggle between **εxx**, **εyy**, **εxy** (shear), and **ω** (rotation) to see
each strain component. Double-click a spot in the reference window to
include/exclude it from the fit, and use **Submit** to freeze the current
field as a new result.

More information
----------------

Strain mapping in 4D-STEM reads lattice distortion straight off the
diffraction pattern. Reciprocal-space disk positions are the inverse of the
real-space lattice, so a lattice that is stretched by a few tenths of a
percent moves its Bragg disks by a correspondingly small amount. Fitting the
shift of every disk in a pattern against an **unstrained reference region** of
the same scan gives a 2×2 displacement-gradient tensor per position,
decomposed into the strain components **εxx, εyy, εxy** and a rigid **rotation
ω**.

It is a relative measurement: the numbers are only as good as the reference.
Pick a region that really is unstrained and single-crystal, and remember that
everything is measured with respect to it. Accuracy also depends on sub-pixel
disk positions, which is why strain runs on a refined **diffraction-vector**
set rather than the raw patterns.

.. tip::

   Run the Find Vectors tour first — strain mapping only appears on a vectors
   result window.

Further reading
~~~~~~~~~~~~~~~

SpyDE wraps pyxem, HyperSpy, eXSpy, kikuchipy and orix; those projects
document the underlying methods in far more depth than a walkthrough can.

* `pyxem — Strain mapping <https://pyxem.org/v0.21.0/examples/strain_mapping/strain_mapping.html>`_

  The full notebook workflow: find peaks, filter vectors, fit a
  DisplacementGradientMap, plot the components.

* `pyxem — Finding diffraction vectors <https://pyxem.org/v0.21.0/examples/processing/vector_finding.html>`_

  The prerequisite step, including the sub-pixel refinement that sets the
  strain precision.

* `pyxem — Data processing gallery <https://pyxem.org/v0.21.0/examples/processing/index.html>`_

  Centring the zero beam and other corrections worth applying before a strain
  fit.


