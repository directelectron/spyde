..
   GENERATED FILE — do not edit by hand.
   Source: guides/virtual-imaging.ts (the same walkthrough the in-app
   guided tour renders). Regenerate with:
       node scripts/gen_guide_docs.mjs

.. _tutorial-virtual-imaging:

Virtual Imaging
===============

Place a virtual detector over the diffraction pattern and form a real-space
image from what it integrates at every scan position.

.. admonition:: Follow along in the app
   :class: note

   Every step below is also a live walkthrough inside SpyDE:
   **Help → Virtual Imaging → Guided tour**. The tour loads the same small
   tutorial dataset for you (no download), highlights each control as you
   go, and closes the example data again when you exit.

Steps
-----

1. What you’ll do
~~~~~~~~~~~~~~~~~

A **virtual image** integrates the diffraction intensity inside a chosen
detector region at every scan position, forming a real-space map. Move or
resize the detector and the image updates live.

.. tip::

   A small tutorial scan (**Tutorial Data → Navigation & Virtual Imaging**) is
   loaded for you — no download needed.

2. Start from a diffraction pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **signal** window shows the pattern under the navigator crosshair. Virtual
Imaging lives on this window’s toolbar.

.. image:: media/virtual-imaging/vi-windows.png
   :alt: Start from a diffraction pattern
   :width: 100%

3. Open the Virtual Imaging tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click **Virtual Imaging** on the toolbar. A sub-toolbar appears where you add
and manage detector regions.

.. image:: media/virtual-imaging/vi-subtoolbar.png
   :alt: Open the Virtual Imaging tools
   :width: 100%

4. Add a detector → a virtual image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a detector region and a **virtual image** window opens, filled from the
intensity it integrates across the scan. Drag or resize the detector on the
pattern to update the image live.

.. tip::

   Try it below — drag the green detector over a diffraction spot and watch
   the scan map light up wherever that spot appears.

.. image:: media/virtual-imaging/vi-output.png
   :alt: Add a detector → a virtual image
   :width: 100%

More information
----------------

A **virtual image** is formed after the fact, in software, from a 4D-STEM
dataset: you choose a region of the diffraction pattern (a virtual detector)
and integrate the intensity inside it at every scan position. Because the
choice is made after acquisition, one dataset yields as many images as you
want — a small disk on the direct beam gives virtual bright field, an annulus
gives virtual annular dark field, and a disk on one Bragg reflection gives a
**virtual dark-field** image showing only the grains that satisfy that
reflection.

The detector shape is the experiment. Moving it across the pattern and
watching the real-space image change is usually more informative than any
single fixed choice, which is why SpyDE recomputes it live as you drag.

.. tip::

   The same operation in a notebook, with pyxem, is linked below.

Further reading
~~~~~~~~~~~~~~~

SpyDE wraps pyxem, HyperSpy, eXSpy, kikuchipy and orix; those projects
document the underlying methods in far more depth than a walkthrough can.

* `pyxem — Interactive virtual images <https://pyxem.org/v0.21.0/examples/virtual_imaging/interactive_virtual_images.html>`_

  A draggable ROI over the pattern with a live-updating virtual image — the
  closest analogue to this tour.

* `pyxem — Virtual images from diffraction vectors <https://pyxem.org/v0.21.0/examples/virtual_imaging/creating_virtual_images_from_vectors.html>`_

  Turn a set of unique vectors into one virtual dark-field image per
  reflection.

* `pyxem — Virtual imaging gallery <https://pyxem.org/v0.21.0/examples/virtual_imaging/index.html>`_

  All four pyxem virtual-imaging examples, including integration over
  non-rectangular detectors.

* `HyperSpy — Data visualisation <https://hyperspy.org/hyperspy-doc/current/user_guide/visualisation.html>`_

  How the navigator/signal pairing and region-of-interest widgets work in the
  library underneath.


