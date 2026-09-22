.. _reports-index:

Reports
=======

A **report** is a whole analysis of a real dataset, written in SpyDE and exported
as one self-contained HTML page: the figures are baked in, and the interactive
panels run entirely in the reader's browser — no server, no Python, no install.
Each one below is a single file you can open, download, or link to.

They are also meant to be *handed out*. Every report has a stable URL and a QR
code sized for a conference poster, so a reader can scan it and explore the
actual dataset on their phone instead of squinting at a static figure of it.

.. _report-pdcusi:

PdCuSi metallic glass — crystallization, in situ
------------------------------------------------

A Pd–Cu–Si metallic glass crystallizing under the beam, recorded as a 4D-STEM
series: 400 series steps × 47 × 39 probe positions × 128 × 128 detector, or
**733,200 diffraction patterns** at 200 kV.

Every pattern went through SpyDE's neural disk detector at a spot size of 8 px
and a threshold of 0.30, followed by the scan-neighbour refine — a second pass
that drops peaks no neighbouring probe position confirms. That gives 1,498,719
diffraction vectors.

The crystallization onset then falls out of the vector *count* alone, with no
phase identification and no fitting: the number of disks per probe position
holds flat at 1.18 for the first hundred steps, rises steeply through step 149,
tops out around 2.6, and drifts back to 2.28 by the end of the run — a 93 % rise
from the first tenth of the series to the last.

Asking the scan for a second opinion is what makes that step so clean. Run at a
bare 0.35 threshold with no refine, the same transition sat on a noise floor
four times higher and showed only a 25 % rise.

The report carries that curve, the summed patterns either side of the
transition, and a live explorer over the series: pick a step, point at the scan,
and the diffraction pattern is redrawn in your browser from the vectors
themselves.

.. This page is itself served from ``/reports/``, and the report + QR files are
   staged into that same directory by ``conf.py``'s ``_stage_reports`` — so these
   hrefs are BARE FILENAMES. Prefixing them with ``reports/`` resolves to
   ``/reports/reports/…`` and silently breaks both the link and the QR image.

.. raw:: html

   <p style="margin:1.2em 0 0.4em;">
     <a class="reference external" href="pdcusi-crystallization.html"
        target="_blank" rel="noopener"
        style="font-size:1.05em;font-weight:600;">
       Open the report &#8599;
     </a>
   </p>
   <div style="display:flex;gap:1.5em;align-items:center;flex-wrap:wrap;
               margin:1em 0 0.5em;">
     <a href="pdcusi-crystallization.html" target="_blank" rel="noopener">
       <img src="pdcusi-crystallization-qr.svg"
            alt="QR code linking to the PdCuSi crystallization report"
            style="width:150px;height:150px;display:block;
                   border:1px solid #d0d0d6;border-radius:6px;padding:6px;
                   background:#fff;">
     </a>
     <div style="max-width:26em;font-size:0.92em;">
       <strong>For a poster.</strong> Print the SVG at 3&nbsp;cm or larger —
       below about 2&nbsp;cm phone cameras start to struggle at poster-viewing
       distance. The code is error-correction level H (~30&nbsp;% recoverable),
       so it survives being scuffed or photographed at an angle.
     </div>
   </div>

:Data: `em-database <https://pypi.org/project/em-database/>`_ —
   ``PdCuSiCrystallization``, Carter Francis (University of Wisconsin–Madison)

Making your own
---------------

A report is built in SpyDE, not written by hand:

#. Open your data and do the analysis — here that was **Find Diffraction
   Vectors** over the whole series.
#. Drag the result window into the report sidebar. It arrives as a live figure,
   and a vectors window brings its explorer with it.
#. Write around it. Text, images and figures are cells you can reorder.
#. **File → Export → HTML (interactive)** — one self-contained file, figures
   baked in, explorer running client-side.

That export is exactly what this page links to. Nothing about it is special to
the docs: the same file works from a USB stick, an email attachment, or a
collaborator's laptop with no SpyDE installed.

Published reports live in ``docs-site/public/media/reports/``, which
``doc/conf.py`` stages into the site (``html_extra_path``) so each is served at
``/reports/<file>.html`` — a stable URL to link, or to point a poster QR at.
