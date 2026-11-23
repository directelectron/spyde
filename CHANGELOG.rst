=========
Changelog
=========

All notable changes to SpyDE are recorded here. Entries are written per pull
request as fragment files under ``upcoming_changes/`` and assembled at release
time by `towncrier <https://towncrier.readthedocs.io/>`_ — see
``upcoming_changes/README.rst``.

This file starts at 0.4.0, the first release cut with towncrier in place.
Earlier releases are described by their GitHub release notes and tags.

.. towncrier release notes start

0.6.0 (2026-11-23)
==================

API and Behaviour Changes
-------------------------

- Crop now opens its result as a new dataset whose navigators show only the cropped scan and frames; it used to add a "Cropped" node whose navigator still showed the whole source, so positions outside the crop displayed the nearest edge frame. (`#262 <https://github.com/directelectron/spyde/pull/262>`_)


New Features
------------

- Find Diffraction Vectors now centers its result on the zero beam and makes the beam the origin ("Center on zero beam", on by default), and says so in the status bar when it leaves the vectors alone because no vector stands out as the beam, as behind a beam stop. (`#254 <https://github.com/directelectron/spyde/pull/254>`_)
- Crop now cuts a dataset on every axis it has — the image or detector, the scan, and a movie's time axis — with a box on every plot that draws one and a slider and a from/to pair per axis in the panel, all kept in step whichever one you move. (`#262 <https://github.com/directelectron/spyde/pull/262>`_)
- Clicking a phase's structure in the Composition dock now pops its viewer out beside the sidebar: the master pattern on a sphere with the detector's outline, the unit cell, and the detector pattern, all turning together as you drag. Under them is the microscope geometry, read from (and written back to) the metadata — SEM for an EBSD scan, TEM for diffraction data — with a choice between kinematic bands and a full dynamical master pattern (with de_ebsd installed), and diffraction spots in TEM. On an EBSD scan, "Solve from data" fits the pattern centre to patterns read across the scan, and writes it only when the fit is good enough to trust. Several open phases stack.


Bug Fixes
---------

- Center Zero Beam now makes the centered beam the origin of the detector axes, on patterns and on vectors alike, so strain and orientation read every spot relative to the zero beam rather than to wherever the detector's offset put it. (`#254 <https://github.com/directelectron/spyde/pull/254>`_)
- The neural Find Diffraction Vectors method no longer reports one disk two or three times when refining moves two of its detections onto the same center, which inflated vector counts and anything summed over vectors. (`#254 <https://github.com/directelectron/spyde/pull/254>`_)
- The time navigator of a time series of scans (5-D data) now draws the time axis's calibration instead of plain frame numbers, and :class:`~spyde.actions.base.CropAction`'s ``t0``/``t1`` on such data cut its time axis rather than the scan's x axis. (`#262 <https://github.com/directelectron/spyde/pull/262>`_)
- The crop box now sits on the pixel edges of the region it keeps; it was drawn half a pixel right of and below it. (`#262 <https://github.com/directelectron/spyde/pull/262>`_)
- A part of the interface that fails to draw now shows its error in its own place — one window, its toolbar or the Plot Control dock — while the rest keeps working; the whole window used to go blank. The error is kept for Help → Report a Problem and in lifecycle.log. (`#263 <https://github.com/directelectron/spyde/pull/263>`_)
- On Windows, an update no longer stalls at "SpyDE cannot be closed. Please close it manually and click Retry" because the analysis backend was running from the install folder: it now runs from its Python environment. The Python that first-run setup downloads now lives in SpyDE's own data folder, so uninstalling can remove it, and setup works behind an institutional proxy that re-signs HTTPS. (`#264 <https://github.com/directelectron/spyde/pull/264>`_)
- Fixed fcc and bcc phases loaded from a CIF losing their {200} and {220} Kikuchi bands in the EBSD wizard's dictionary and band overlay: an extinct reflection along the same line was kept in their place.
- Fixed the EBSD wizard simulating its dictionary and drawing its band overlay for an untilted detector: both now use the scan's sample and detector tilt (70° by default), so the bands line up with real patterns at every orientation rather than only some.


Performance
-----------

- Solving a scan's EBSD pattern center from its own patterns now takes about 25 s, down from minutes, because the candidate orientations are sampled once and reused across the search. (`#246 <https://github.com/directelectron/spyde/pull/246>`_)
- The neural Find Vectors batch reused its GPU memory from one chunk to the next instead of reallocating it, which made a 13,312-pattern SPED-Ag scan 14% faster (28.0 s to 24.1 s on a TITAN X) with identical vectors. (`#256 <https://github.com/directelectron/spyde/pull/256>`_)


0.6.0-rc.4 (2026-10-26)
=======================

API and Behaviour Changes
-------------------------

- The Rebin fields now read Scan X, Scan Y, Diffraction X, Diffraction Y, in that order; the diffraction factors used to come first, so values typed in by position land on different axes than before. (`#250 <https://github.com/directelectron/spyde/pull/250>`_)


New Features
------------

- Center Zero Beam can now fit the descan plane through the four corners of the scan only, the same corner boxes the DPC Center tab uses, so a sample in the middle of the scan no longer pulls the centering off. (`#249 <https://github.com/directelectron/spyde/pull/249>`_)
- The Rebin fields are now split by a separator: Scan X  Scan Y | Diffraction X  Diffraction Y. (`#252 <https://github.com/directelectron/spyde/pull/252>`_)


Bug Fixes
---------

- The strain window's rotation arrows on the reference pattern no longer sometimes sit as a speck in the pattern's corner; they are centred on the pattern from the moment the window opens. (`#248 <https://github.com/directelectron/spyde/pull/248>`_)
- After binning or cropping the scan, the navigator crosshair showed the pattern from a different position (binned by 2, everything past half the scan width showed the last column); it now shows the binned or cropped pattern under the crosshair. (`#250 <https://github.com/directelectron/spyde/pull/250>`_)
- Orientation-mapping libraries built from hexagonal, trigonal and monoclinic CIFs (barium titanate, GaN, ZnO, sapphire and Ti from COD among them) now use the atom positions in the file, so they keep their space group, and a library too large for the free memory is built at a coarser zone-axis step and says so, where it used to report 0 orientations. (`#251 <https://github.com/directelectron/spyde/pull/251>`_)
- With a binned or cropped node selected, Plot Control showed the original dataset's size and axis calibration; it now describes the selected node, and an axis edited there is applied consistently to every node of the dataset. (`#252 <https://github.com/directelectron/spyde/pull/252>`_)


0.6.0-rc.3 (2026-10-20)
=======================

Bug Fixes
---------

- Find Diffraction Vectors with the neural detector works on Apple-silicon Macs again: the centre-refine step converted its result to 64-bit floats on the GPU, which Apple's Metal backend does not support, so every batch failed and no peaks appeared. (`#244 <https://github.com/directelectron/spyde/pull/244>`_)


0.6.0-rc.2 (2026-10-14)
=======================

API and Behaviour Changes
-------------------------

- Find Vectors' controls were rearranged: the heatmap became a switch beside Method, "Mask beam stop" moved under a collapsed "Extra", "Check for new models" moved into the Model menu, and a taught model is named from a Save button beside Adapt and Revert. (`#229 <https://github.com/directelectron/spyde/pull/229>`_)
- Strain Mapping changed its fit, so existing strain maps change when re-run: each vector was weighted by its confidence and positional uncertainty, only one vector per reflection was kept, vectors far from the fitted lattice lost their influence (Tukey biweight), and the reference was centred on its Friedel pairs — halving the in-grain noise on SPED-Ag and making exact spot positions behind a beam stop give exact strain; ``compute_strain_field(robust=None, one_per_reflection=False)`` restores the earlier weighting. (`#229 <https://github.com/directelectron/spyde/pull/229>`_)


New Features
------------

- File → Export to NumPy… wrote what the focused window shows — a committed strain, DPC or orientation map with every component and the fit behind it, a virtual image, a line profile, or the spectrum under the navigator — to a ``.npz`` (or ``.npy`` / ``.csv``) with its calibrated axes and a record of each array's label, units and provenance. (`#193 <https://github.com/directelectron/spyde/pull/193>`_)
- While Orientation Mapping, EBSD Indexing or Vector Orientation Mapping is still
  filling its orientation map, that window's toolbar buttons are shown greyed out
  with a tooltip saying why, and come back when the map is finished, fails or is
  cancelled, as Find Diffraction Vectors' window already does. (`#217 <https://github.com/directelectron/spyde/pull/217>`_)
- The **Vector Orientation Mapping** Refine readout now also gives the residual over the matched peaks alone ("matched resid"). The existing "resid" counts every detected peak, so it rises when Find Vectors picks up weak reflections the simulated pattern leaves out, even though the fit itself has not got any worse. (`#218 <https://github.com/directelectron/spyde/pull/218>`_)
- Find Vectors' neural detector could be taught on your own data — double-click a wrong circle or a missed disk and a copy of the model is refitted on your marks, which you can name and reuse on other datasets — and the Model dropdown showed each model's icon, with the built-in models first and yours after a divider. (`#229 <https://github.com/directelectron/spyde/pull/229>`_)
- A new detector, "dynamical-mask8", was offered in Find Vectors → Model: it places each disk at the centre of its aperture rather than where the intensity sits inside it, halving the centre error on dynamically filled disks and no longer flooding sparse counted frames with false detections. (`#229 <https://github.com/directelectron/spyde/pull/229>`_)
- Every diffraction vector found with the neural method gained a confidence (0–1) and a positional uncertainty, and vectors the network was unsure of were drawn faint on the pattern; vectors saved by earlier versions, or found with NXCORR or DoG, open with both empty. (`#229 <https://github.com/directelectron/spyde/pull/229>`_)
- Find Vectors (neural) started re-placing every disk on the raw frame with a small bundled centre network, and a "Friedel partner" checkbox, on by default, added a second network that also reads each disk's partner across the direct beam — on real GaN this lowered the Friedel error of the centres from 1.13 to 0.64 px. (`#229 <https://github.com/directelectron/spyde/pull/229>`_)
- In Present mode all four split layouts render as chosen (text above or below the picture as well as beside it), text slides scale their type to fill the stage, and a PNG dropped on a split slide's picture slot fills that slot instead of being added below. Slides now size to the display: moving the presentation to a projector, maximising it or changing the resolution scales text, pictures and split panes together, and live figures redraw sharply at the new size. (`#232 <https://github.com/directelectron/spyde/pull/232>`_)
- A figure saved in a report reopens as an interactive figure you can pan and zoom, even when the data it came from is not loaded, instead of as a flat picture. (`#233 <https://github.com/directelectron/spyde/pull/233>`_)
- In a report or presentation, dropping an image or a live figure on an existing picture replaces it in place, and dropping one on a text slide turns it into a split slide. (`#234 <https://github.com/directelectron/spyde/pull/234>`_)
- SpyDE now opens Gatan DigitalMicrograph (.dm3, .dm4), EMD (.emd, including Velox EDS spectrum images), Quantum Detectors Merlin (.mib), blockfile (.blo) and Ripple (.rpl) files, all loaded lazily so a large scan is read from disk only as you look at it. (`#237 <https://github.com/directelectron/spyde/pull/237>`_)
- Double-clicking a ``.hspy``, ``.de5`` or ``.csb`` file now opens it in SpyDE, and
  SpyDE is offered under "Open with" for the other formats it reads (``.mrc``,
  ``.tif``/``.tiff``, ``.dm3``/``.dm4``, ``.emd``, ``.mib``, ``.blo``, ``.rpl``)
  without replacing the app those already open in. Opening a file while SpyDE is
  running adds it to the open window instead of starting a second copy. (`#238 <https://github.com/directelectron/spyde/pull/238>`_)


Bug Fixes
---------

- While Find Diffraction Vectors is still running, dragging the result window's
  navigator over positions the batch has already computed now shows their
  diffraction patterns, and once you touch the navigator the window stops
  replacing your pattern with automatic previews. The window's toolbar buttons,
  including the vector actions that were previously hidden until the run
  finished, are shown greyed out with a tooltip until the vectors are ready. (`#132 <https://github.com/directelectron/spyde/pull/132>`_)
- Side-by-side navigators now keep every panel's crosshair on the navigator's one position: dragging a crosshair in one panel moves the others to it without them snapping back, the panels start where the navigator already was, and a selector added while the navigators are tiled shows its crosshair on every panel. (`#171 <https://github.com/directelectron/spyde/pull/171>`_)
- Choosing a signal type while the navigator window had focus re-typed the navigator image instead of the dataset. (`#194 <https://github.com/directelectron/spyde/pull/194>`_)
- Sizes were given one format across the app, with the unit on every value (download progress, and the compute monitor's memory and GPU columns), and a phase's atomic percentage was made to save as it is typed and to stay between 0 and 100. (`#197 <https://github.com/directelectron/spyde/pull/197>`_)
- A CSB movie opened or re-cut (**To Frames**) at an exposure that is not a whole number of camera frames now has a correct time axis. Planes then alternate in width, and the axis was spaced by the first gap alone, so it drifted with every plane — at 3.5 frames per plane the last plane of a 100 ms movie was labelled 112 ms. Every plane now sits within one camera frame of when it actually started. (`#199 <https://github.com/directelectron/spyde/pull/199>`_)
- Emptying a figure cell back to a placeholder no longer leaves it holding its
  baked image, which a later save could write out as an asset the cell no longer
  owns. (`#215 <https://github.com/directelectron/spyde/pull/215>`_)
- A window dropped on the report body between two cells now becomes a figure cell
  even when the drag arrives with an unreadable payload, which is how a real drag
  out of the operating system can look. (`#215 <https://github.com/directelectron/spyde/pull/215>`_)
- A split block's text side is now the same markdown editor a text cell has, with
  the formatting toolbar and the Ctrl-B / Ctrl-I shortcuts it was missing. (`#215 <https://github.com/directelectron/spyde/pull/215>`_)
- An exported interactive figure is now sized from its own shape the way the
  sidebar cell is, rather than dropped into one fixed box that letterboxed a wide
  row of panels and stretched a single square pattern, and a figure with no pixels
  keeps its caption instead of disappearing. (`#215 <https://github.com/directelectron/spyde/pull/215>`_)
- DPC: a beam-shift pass could be handed the dataset's live signal, instead of a copy of its own, when a new pass started just as the previous one was checking its beam region. The two copies of that signal could then fail, or show other readers an empty placeholder in place of the data. Each pass now reads its own copy, and taking that copy no longer touches the original. (`#216 <https://github.com/directelectron/spyde/pull/216>`_)
- Stopping **Find Vectors** while its live preview was drawing no longer leaves the preview's circles on the diffraction pattern. A removal that landed mid-draw used to be undone by that draw. (`#216 <https://github.com/directelectron/spyde/pull/216>`_)
- **Find Vectors** used to start with whichever automatic settings had arrived when you clicked Compute. The spot size estimated from the pattern and the neural calibration land a moment after the caret opens, so an early click could run the whole batch with the defaults instead. Compute is now greyed out and reads "Calibrating…" until both have arrived. A spot size or threshold you set by hand in the meantime is no longer replaced when they do. (`#218 <https://github.com/directelectron/spyde/pull/218>`_)
- Find Vectors (NXCORR) now places each spot at its subpixel centre. Positions used to snap to whole pixels (up to ~1 px off on flat-topped disks), and a small minimum distance reported the same disk several times; strain maps built from NXCORR vectors are no longer quantised. (`#219 <https://github.com/directelectron/spyde/pull/219>`_)
- Find Vectors' neural detector placed spots up to 0.6 % too close to the pattern's top-left corner (up to 0.7 px at the far edge), which biased absolute, CIF-referenced strain by the same amount. (`#229 <https://github.com/directelectron/spyde/pull/229>`_)
- The "Open full Dask dashboard" menu item and the dashboard button no longer disappear when the analysis backend finishes starting after its cluster. (`#231 <https://github.com/directelectron/spyde/pull/231>`_)
- Updating on Windows no longer dead-ends in "SpyDE cannot be closed. Please close it manually and click Retry" when an analysis backend from an earlier session is still running: the installer now finds and stops one wherever it is running from, not only inside the installation directory.


0.6.0-rc.1 (2026-09-22)
=======================

API and Behaviour Changes
-------------------------

- A 4D-STEM dataset calibrated in nm⁻¹ now indexes correctly. Å⁻¹ is what diffsims, orix and pyxem's matcher work in and none of them checks, so a scan like the ZrNb precipitate example — which ships at 0.0513 nm⁻¹ per pixel — built a template library ten times too large and returned an orientation map with nothing raised; the CIF strain reference matched no reflection at all and silently produced no map. Detector axes are re-expressed in Å⁻¹ when a file opens (the scan axes are untouched, and the conversion RESCALES rather than relabels), the crystallographic paths ask what one pixel is worth in Å⁻¹ instead of reading the axis scale, and a detector with no reciprocal calibration now says so rather than assuming. ``spyde.reciprocal_units`` converts between px, mrad, nm⁻¹ and Å⁻¹ — only mrad needs a beam energy — and the Plot Control dock shows ``A^-1`` and ``nm^-1`` as Å⁻¹ and nm⁻¹ instead of "A⁻1" and "nm⁻1".
- A drift solve against a fixed reference frame (:func:`spyde.drift.solve_translation` with ``reference="fixed:<i>"``) now registers every other frame against that frame, frame 0 included — previously frame 0 was declared the origin whatever the reference was, so unless it happened to be aligned already its shift came back as zero instead of its real displacement, and its correlation quality was reported as perfect.
- Vector Orientation Mapping is now indexed by correlation against a zone-axis template library — the ACOM method — instead of the per-pattern pose fit that preceded it, which has been removed. The practical differences on opening a scan: the fit reports a ``reliability`` (the best correlation minus the best one outside the exclusion ball, which is what a grain-boundary mask is thresholded on) alongside the correlation; **Generate** now stops at the library and the live previews rather than also fitting the whole field, so the orientation map appears when **Compute Maps** is pressed and not before; and the Refine tab carries the matcher's own two arguments — pair distance and excitation σ, both in Å⁻¹ — in place of the strain cap, no-match tolerance, intensity gamma and high-k lever arm, which were parameters of the old fit and would have been four dials wired to nothing. The two report strain in opposite senses, so the new path deliberately has no fallback to the old one: a failure surfaces rather than silently flipping the sign of every strain map.


New Features
------------

- A figure's own "Save PNG…" (the ⤓ badge, or right-click, on any figure) now writes a file: the figure hands SpyDE its image and SpyDE asks where to save it. Before, the figure fell back to a preview captioned "Right-click the image → Save image as…", a menu the app does not have, so there was no way to save at all. (`#181 <https://github.com/directelectron/spyde/pull/181>`_)
- A multi-angle 4-D STEM acquisition can now be assembled a stage at a time —
  adding and removing member files (naming the scan grid for a format that does
  not record one), setting each member's tilt and azimuth, then solving the
  real-space and reciprocal alignments separately — with every stage inspectable
  and re-runnable before anything is composed.
- A multi-angle 4-D STEM acquisition can now be navigated by its angles directly:
  a polar window draws the acquisition as it was taken — one ring per tilt shell,
  one point per member at its azimuth, so a missing angle shows as a gap. While
  the summed scan is displayed every angle is lit and clicking one switches to
  that single angle; on the aligned stack the live angle carries a handle, and
  clicking the ring or dragging that handle round it scrubs through the angles.
  The ring and the ordinary angle slider always agree — both drive the same
  selection.
- A multi-angle 4-D STEM acquisition now opens as one dataset — the aligned
  stack of every angle, with the summed scan shown first and a per-shell sum for
  each tilt, all selectable from the Workflow section of the dock.
- A sample is now a list of phases, each its elements (with optional atomic percentages) and the structure that indexes it, built in one popout from Plot Control's Composition section ("＋ Elements and Phase") or from the Load tab of Orientation Mapping, Vector Orientation Mapping and EBSD Indexing. The periodic table edits the selected phase, so one element can belong to two phases (zirconia and alpha-Zr). An element can be marked trace — the O in an Fe phase: EELS and EDS still fit it, but it is left out of the phase's structure search. Each phase takes a .cif from a file, a recent file, or a COD search scoped to that phase's non-trace elements — COD matches elements exactly, so a Cu/Nb sample searched as one composition found nothing, while fcc Cu and bcc Nb are one query each. EELS and EDS fit exactly the elements of the phases: removing one also removes its X-ray lines, so it is no longer fitted. The indexing wizards use each phase's structure, and EBSD asks which one when several have one, and asks for a rebuild when it changes. A file whose metadata already lists elements opens with them as Phase 1, and an element added from the console joins Phase 1 as trace.
- Aligning a multi-angle acquisition in real space now opens a window showing
  the members summed with and without the solved offsets, with the sharpness
  gain between them.
- An action still under active development can now say so. Declaring ``beta: True`` in the toolbar schema puts a small β badge on its toolbar button and a ribbon across its caret reading "still under development, may change", so results and controls that are expected to move are labelled rather than quietly presented as settled. The flag never hides an action. Vector Orientation Mapping carries it first, while its matcher is rebuilt.
- Center Zero Beam works on a vectors window. The beam is the brightest vector
  inside the search box at each scan position, one plane is fitted through it
  across the scan, and every vector moves by centre minus plane into a new
  "Centered" vectors tree, so a scan can be centred after finding its vectors
  without centring a frame; the Manual tab's picked position applies as a
  constant shift the same way.
- Help → Record Screen records the SpyDE window to a video file, for a demo or to show someone a bug: it asks where to save first, then streams the capture straight to disk while a red REC pill in the bottom-left counts the elapsed time and stops it. What it captures is SpyDE's own window contents, so no source picker appears, nothing from another app can land in the clip, and macOS does not ask for Screen Recording permission — the trade-off being that native menus and dialogs, which the OS draws on top, are not in the recording. The file is mp4 where the runtime can mux it and webm otherwise.
- The multi-angle loader now aligns real space from a virtual image the files
  already carry — picking one every member has, or computing one when there are
  none — and aligns reciprocal space from four independently sized corners of
  each scan by default, reading a fraction of a percent of an acquisition where
  the older methods sample all of it.
- The multi-angle loader now shows each member as a picture rather than a row of
  text — every member's real-space image and the four corner diffraction sums
  behind the reciprocal stage, filled in one member at a time as soon as the
  files are opened rather than after a stage has been run.
- The orientation wizards' IPF window now follows the action, for both the
  dense and the vector mapper: selecting the action opens it, a phase with a
  structure draws its empty triangle, Generate fills the triangle while the
  library builds and then shows the orientations it sampled, Refine draws the
  correlation heat map into it, and deselecting the action hides it. Closing
  the window hides it too; reselecting the action brings it back with the heat
  map still live, where before a closed heat map was gone until the library
  was rebuilt.
- Vector Orientation Mapping draws its IPF maps by confidence: a position's
  colour is scaled by its correlation and a position nothing matched is grey,
  so an amorphous region no longer paints as loudly as a grain and an
  unmatched position no longer paints as the identity orientation (pure red
  in IPF-X). The Run tab gained a "Rescue passes" setting (default 3, was one
  pass): the neighbour rescue that re-fits a position from its neighbours'
  orientations now repeats until a pass changes nothing, so a mis-indexed
  patch wider than one position is cleaned up too.
- Vector Orientation Mapping shows a live IPF correlation heat map — one triangle per phase, coloured by how well every sampled orientation explains the pattern under the crosshair — and double-clicking a triangle restricts the match to the orientations inside the circle it draws. That is what a single best number cannot tell you: a confident position is one bright spot, an ambiguous one has several, and a phase the pattern does not belong to is uniformly dim; the circle is how you say which of several candidates is meant. The restriction reaches the match itself, not just the picture, so the matched pattern drawn over the diffraction spots is refitted under it; double-clicking inside the circle again lifts it, as does closing the window. It applies to the live fit under the crosshair rather than to Compute Maps, which indexes the scan unrestricted — the same scope the dense Orientation Mapping heat map's circles have. The window also toggles with the action now, so closing it no longer retires the heat map until the library is rebuilt.
- Vector Orientation Mapping takes several crystal structures at once, so one fit answers which phase, in what orientation, under what strain — the question a precipitate in a matrix actually poses. The wizard's Load tab holds a phase list (add from a file or from the Crystallography Open Database, remove with ✕) like the dense Orientation Mapping caret already did, and a run with more than one phase opens a Phase map beside the orientation and strain windows, colouring each position by the structure that best explains its pattern and leaving positions that did not fit grey. This also repairs the underlying library: a two-phase simulation stores its rotations one entry per phase, so the vector library was being built with two templates instead of the twelve hundred it contained, and every position in a two-phase scan indexed as whichever of the two survived.
- Vector Orientation Mapping's Compute Maps opens the orientation window at
  once and fills it in a band of scan rows at a time as the match lands, with
  a percentage that counts positions rather than sitting on "33%" until the
  whole scan had matched. The result is the same as before: the match is per
  position, and the refinement still runs over the whole map. The IPF Refine
  heat map also moved off the navigator thread, so dragging the crosshair with
  it open no longer waits on its correlation at every step.
- Vector Orientation Mapping's Run tab gained "Smooth orientations": each
  position is averaged with the neighbours within a grain threshold of it
  (default 5°), so the tilt noise that speckles IPF-X and IPF-Y inside a grain
  settles while a grain boundary stays where it was; the raw field is kept
  beside it. The Library tab gained the plan's in-plane angle step, which was
  fixed at 5°.


Bug Fixes
---------

- A callout dropped onto a report figure built from a 5-D dataset drew its
  connector rectangle over the wrong part of the navigator, because the outer
  (time or angle) navigation index was read as the region's x origin.
- A multi-angle acquisition that was rebinned in the app reopens as its tree
  again. Rebin summed into uint64 — four times the bytes asked for — and the
  sum over angles then refused a 64-bit stack, so the saved file opened as a
  plain 5-D array with no Summed node and no angle ring, and only the log said
  why. Rebin now sums into the narrowest exact width (uint32 for a uint16 scan
  binned 2x2x2x2), a 64-bit stack keeps its width, and a re-expansion that
  fails is reported in the app.
- A window dragged partly off the left or right edge of the work area keeps
  its floating toolbar reachable. The bar was centred under the window with
  no clamp, so its first buttons walked off the app with the window and their
  actions could not be clicked at all; it is now pushed just far enough to
  stay whole, the way an open caret already was.
- An integrating span on the outer navigator of a 5-D dataset, and a point
  selector given a frame width, both showed a single position instead of
  integrating the positions they selected.
- Closing a virtual image's output window with its own close box now retires
  the image the way deselecting it does: the ROI comes off the pattern, its
  chip leaves the sub-toolbar, and the Virtual Imaging button un-highlights.
  It used to stay lit with nothing behind it.
- Focusing a window that has no navigator of its own, such as an IPF Refine
  heat map or a strain window, no longer fills the Plot Control dock with a
  navigator selector row for every navigator in the app.
- In the multi-angle loader the arrow keys nudge the member being compared.
  The pair view's nudge pad shared a React ref with the alignment grid's, so
  whichever mounted last took the focus and the keys could move the member the
  other pad was showing. The tableau also grouped members onto rings by a 0.05°
  tolerance while the backend counted shells at 0.01°, so the picture could
  show one ring where the status line said two shells; both now use 0.01°.
- Multi-angle real-space alignment scores the whole field and each voting
  patch with the same sharpness measure. The patches used a second, slightly
  different one, so the decisive margin that lets a patch vote overrule the
  whole-field answer compared two different quantities. One measure now lives
  in the alignment package; a patch is scored without the edge margin, which
  on the four-member acquisition the agreement figures come from was the
  difference between an axis being called determined and not.
- Multi-angle real-space alignment: a member whose true offset on an axis is
  zero no longer has its correct votes discarded. The patch vote dropped every
  zero so an axis with no signal could not look unanimous, which left the few
  patches that mis-registered a genuinely unshifted member to set its offset
  unopposed, at an agreement of one. Zeros now count in the median and are left
  out only of the agreement. A patch too small to score no longer hands the
  choice to the raw-intensity default while reporting a gain of -inf; the
  sum-over-angles node keeps the accumulator it chose instead of numpy's
  uint64; and a saved 4-D sum that inherited the stack's signal type is no
  longer mistaken for a stack on reopening.
- Orientation Mapping with more than one phase now draws the matched template's spots on the diffraction pattern, and reports the template count for every phase; the spot overlay was skipped for a multi-phase library because the code that builds a template's spots counted phases as templates and filed every template under the first phase. The Refine tab also says when the library is ready, or that it failed, instead of sitting on "Generating library…" forever, and zooming an IPF heatmap no longer blanks it (anyplotlib 0.10.1).
- Switching the displayed node of a multi-angle stack to a child that summed its
  leading navigation axis away painted one row of a diffraction pattern instead
  of the pattern.
- The Examples menu offers one SPED-Ag scan, the copy whose reciprocal
  calibration is correct. The half-scale copy em-database ships was listed
  beside it under the familiar name, and loading that one left every
  orientation fit unable to match a crystal; the old name now loads the
  calibrated copy too.
- The Virtual Imaging "+" bar opens below the toolbar again for a window in
  the lower part of the work area. It was being placed as if it were a
  320-pixel caret, so it flipped above the window whenever that much room
  was missing below, even though it needs about forty pixels.
- The angle ring works on a reopened multi-angle acquisition. It asked only
  the runtime recipe whether one angle was on screen, and a reopened tree has
  none, so the ring stayed on "all angles" with every member lit, no handle,
  and a full node switch on every pick. A pick now stays on the node being
  worked on when that is itself a stack (a binned copy, say) instead of
  jumping back to the full-resolution root. A reopened tree also reports the
  recorded member offsets rather than a sum's zeroed ones, names its shells
  as the composition does, is brought up to the current signal type, reuses
  the navigator planes the composition now keeps in the file instead of
  reducing the whole stack to draw a thumbnail, and a failure while
  rebuilding the tree no longer opens the dataset a second time.
- The corner labels of an IPF Refine triangle are no longer clipped at the
  panel's edges.
- Vector Orientation Mapping's Compute Maps no longer fails on a scan with one
  degenerate position ("The input matrix is singular"); that position's strain
  is NaN like one with too few pairs. A multi-phase result puts the phase map
  and one orientation map per phase on the orientation window as chips, so a
  single phase can be looked at on its own; the phase map was painting every
  position grey for the matcher's result. The matched pattern's circles now
  fade with each reflection's intensity, so the spots the match rests on stand
  out from the barely excited ones.
- Vector Orientation Mapping's status line now says which stage is running
  (building the plan for a phase, matching rows m-n of N, refining) instead of
  a percentage that stood still through the plan build and the refinement,
  and its first bands are one, two and four rows, so on a wide scan the map
  starts filling after one row's match rather than after eight.
- Vector Orientation Mapping's strain maps no longer carry values in the
  thousands of percent. Those came from positions whose paired peaks all lay
  along one line of reflections, which cannot determine the other direction;
  such a pairing is now refused, and any solved stretch beyond fifty percent
  is treated as a failed pairing rather than a strain, so a few bad positions
  no longer set the colour scale of the whole map.


Performance
-----------

- Opening a ``.zspy`` or ``.zarr`` no longer stalls before the "Reading…"
  message appears. The size behind that message was measured by walking every
  file in the store, which on a frame-chunked scan is one file per frame —
  65,579 of them, and 35.6 s of stat calls, on a real 256 x 256 acquisition. A
  directory store is now taken as large without being measured. (`#178 <https://github.com/directelectron/spyde/pull/178>`_)
- Resizing a window sends the figure at most ten size updates a second
  instead of one per frame. Each update round-trips through the backend and
  re-sends the figure's whole image, so an orientation, phase or strain map
  window used to queue dozens of full-image pushes during a corner drag and
  lag its cursor.
- Scrubbing the Summed node of a reopened multi-angle acquisition reads at the
  store's decode floor. Each angle was read through a lazy slice of the stack,
  55–80 ms per angle against a 25 ms chunk decode; the node now reads every
  angle through the stack's one store reader. Measured on a real four-angle
  stack with 51 MB chunks: a cold frame 282 → 99 ms, a chunk crossing 287 →
  108 ms, a frame inside a decoded chunk 2 ms. Composed nodes also tell the
  read-ahead where their next chunk boundary is, so a drag decodes the next
  chunk before it arrives.


0.5.1 (2026-09-14)
==================

New Features
------------

- Strain maps now read in percent strain (lattice rotation in degrees) under a labelled colorbar, over the scan's own calibrated axes with a scale bar — in the live Strain window, the ⌘-tiled comparison and the committed tree alike; the committed tree keeps its diverging colormap and a zero-centred contrast on every node (dragging one histogram handle mirrors the other), each component gets a scale of its own, and the tensor can be expressed in the scan's x/y rather than the detector's — a pink x / gold y arrow pair on the reference pattern that turns by dragging, a Rotation / Flip control in the caret, and a picker that takes the angle and handedness from any DPC result open in the session; committed DPC, orientation, fit-component and virtual-image maps carry the scan calibration too, and DPC component maps say their units on the colorbar.


Bug Fixes
---------

- Opening a Direct Electron ``.de5`` no longer reads the whole datacube into memory (an acquisition stopped early tried to allocate its full planned scan), and its navigator now fills in about 2 s instead of 80 s on a 1 GB 128×64 scan, with a frame read in 0.1 ms. (`#164 <https://github.com/directelectron/spyde/pull/164>`_)
- A component added from the Fit caret is placed against the spectrum on screen
  again. Since the curves became an overlay the caret learned the spectrum only
  from that overlay's own value, which is produced when the navigator runs and
  not when the caret opens — so a component added before anything moved arrived
  at the catalogue's default amplitude, five orders of magnitude below the data,
  and drew as a flat line on the axis.

  Entering a transform view — the Find Vectors detector response, and every
  overlay that replaces the diffraction pattern with an image of its own — no
  longer flashes the raw pattern underneath first. The navigator's own frame
  waits for the transform to say whether it owns the display, and an overlay
  whose function fails now clears its groups instead of leaving the window on
  the last image it drew. (`#166 <https://github.com/directelectron/spyde/pull/166>`_)


Performance
-----------

- Distributed batches over a compressed ``.zspy`` decode each chunk with blosc's
  thread pool inside every dask worker process, where they used to decode
  single-threaded; on a 4x4-chunk square of a 512x512 float32 scan the navigator
  sum went from 693 ms to 401 ms and a Find Vectors batch from 1085 ms to 948 ms. (`#157 <https://github.com/directelectron/spyde/pull/157>`_)
- A lazily opened ``.hspy`` or ``.zspy`` stored in small chunks is now re-blocked at load into ~64 MB dask chunks of whole frames, so a scan written one frame per chunk no longer costs a dask task per frame; the readers ignore ``chunks=``, so this happens by rebuilding the wrap of the stored dataset, which reads nothing. (`#165 <https://github.com/directelectron/spyde/pull/165>`_)


0.5.0 (2026-09-11)
==================

Bug Fixes
---------

- Direct Electron ``.de5`` files open again as 4D-STEM scans; the EMD reader rejected the camera's column-shaped axis arrays with a ``TypeError``, and once past that it handed the datacube back transposed with no navigation axes. (`#162 <https://github.com/directelectron/spyde/pull/162>`_)


0.4.4 (2026-09-11)
==================

Bug Fixes
---------

- Every display that follows the navigator (the Find Vectors preview and its found
  vectors, the orientation template, the EBSD bands, the refine panels, the strain
  selection, image layers from another window, the fit's curves, the vectors
  window, the progressive result preview, the CSB raw frame) is now a node of the
  signal tree read through the same cached path as the pattern itself and drawn by
  the one painter thread. After centring a pattern, Find Vectors and Orientation
  Mapping run on the node the window shows, so their overlays land on the centred
  disks instead of the root's; on a centred lazy scan a preview move costs under a
  millisecond instead of a whole-block compute. (`#155 <https://github.com/directelectron/spyde/pull/155>`_)


Performance
-----------

- Displaying a node made by a hyperspy ``map`` (a centred pattern, an azimuthal
  integration, a per-pattern filter) went from about 2.2 s to 0.55 ms for the
  first frame in every chunk, by evaluating the mapped function on the parent's
  one frame instead of computing the whole dask block; switching between tree
  nodes now also keeps the root's decoded chunks instead of re-decoding them. (`#153 <https://github.com/directelectron/spyde/pull/153>`_)


0.4.3 (2026-08-31)
==================

API and Behaviour Changes
-------------------------

- The DPC beam region no longer has an "off" setting — it is what the centre of
  mass is taken over, so it is always on the pattern. Its default radius is now
  half the shorter detector axis rather than a quarter: with the region always
  on, a smaller default clipped the beam and under-read every field.


New Features
------------

- Help → Report a Problem… sends a description of what went wrong together with the machine's OS, app and runtime versions, GPU, Python-environment state and the backend's recent output — shown in full before anything is sent, and saved to a file when the machine is offline.


Bug Fixes
---------

- The DPC beam region no longer collapses the moment the pointer enters the
  diffraction pattern. Its radius was being written to the widget alone, while
  the figure's own state kept the radius the widget was created with — a fifth
  as large — and reverted to it on the next redraw, so the region that was
  actually measured was a few pixels wide and dragging it moved that.
- The DPC field map now recomputes as the beam region is dragged, instead of
  staying frozen until the pass before it had finished. The beam region is a
  real selector, so a superseded measurement is cancelled rather than left to
  run, and the centre of mass is ~37x faster.
- Updating on Windows no longer dead-ends in "SpyDE cannot be closed. Please close it manually and click Retry": the app now shuts its analysis backend down before handing off to the installer, and the installer waits for whole process trees to exit instead of giving up after two rounds.


Maintenance
-----------

- The desktop shell moved from Electron 34 to Electron 44 (Chromium 132 to 152, Node 20 to 24), returning SpyDE to a supported Electron line that still receives security fixes.


0.4.2 (2026-08-25)
==================

Bug Fixes
---------

- The installed app could not start: first launch failed with ``Distribution
  not found at: .../resources/python/packages/de-shell``. Extracting the shell
  made this a uv workspace, and the installer payload shipped the lock that
  refers to the ``de-shell`` member without shipping the member. Both workspace
  wheels are now built and installed, and the sync no longer tries to build
  either from the read-only payload.


0.4.1 (2026-08-25)
==================

Bug Fixes
---------

- The 0.4.0 release build could not be packaged: electron-builder refused the
  ``^34.0.0`` Electron range because it could no longer resolve the installed
  version. Making the repository an npm workspace hoisted ``electron`` out of
  ``electron/node_modules``, and electron-builder resolves from the project
  directory. Nothing but a tag build runs electron-builder, so it did not
  surface until the release. The version is now pinned exactly, in both
  ``package.json`` files.


0.4.0 (2026-08-25)
==================

API and Behaviour Changes
-------------------------

- The application shell was extracted into shared packages: ``de_shell``
  (Python — the actions framework, ``SessionBase``, the backend loop, IPC and a
  shared figure) and ``@de/shell-main`` / ``@de/shell-preload`` /
  ``@de/shell-renderer`` (the Electron window, message pipe, chrome reducer and
  figure bridge), all under ``packages/``. SpyDE is now one app built on that
  shell rather than the only one, so code that used to be imported from
  ``spyde.*`` may now live under ``de_shell.*``.
- Ground Crew and Autopilot moved out into their own repositories. They were
  developed here while the shell was being carved out; they are no longer part
  of this codebase.
- ``SpyDEDiffractionVectors`` now inherits the shared ``RaggedStore``. The CSR
  machinery it used to own moved up into that base class; the public methods are
  unchanged.

New Features
------------

- Differential phase contrast: electric and magnetic field mapping, with
  measure-once centering (corner, vacuum or manual), a rotation solver, and a
  live colour-wheel readout.
- Rigid drift correction — solver, model, warp and a staged wizard, driven end
  to end on real pixels.
- ``RaggedStore``, a shared per-navigation-position column store for ragged
  results, so vectors and their downstream results share one backing format.
- ``FrameStream``, which refreshes a figure from a future or a background thread
  without the caller marshalling the result itself.
- ``lifecycle.attach_container`` — one seam for attaching a result container to
  a tree, replacing the per-action variants.

Bug Fixes
---------

- Superseded computes are now cancelled by every action that dispatches one, so
  a rapid sequence of requests no longer leaves earlier work running and
  painting stale results over newer ones.
- A ``.tif`` file could open as a black window stuck on "Calculating…".
- Figures now fill their pane in the live apps instead of sitting at their
  built size in a larger box.
- Shrinking the workspace reflows the windows instead of stranding them
  off-screen where they could not be reached.
- A concurrent HyperSpy operation could park a ``(1,)`` placeholder on ``.data``
  and break the navigator read; it is now skipped.
- Caret placement tracks the caret's own size, so a tall caret no longer
  overhangs its anchor.
- EBSD band-simulator uploads take the shared device lock, which on Apple MPS is
  the difference between a result and an uncatchable native crash.
- Frame-read fallbacks report instead of passing silently, so a failed read is
  visible rather than showing as an empty frame.
- ``SPYDE_NO_HMR`` stops the dev server reloading the page when the machine
  sleeps.

Maintenance
-----------

- CI grew a fast PR tier and duration-balanced end-to-end groups, with npm,
  browser and dataset caches, and now covers the platforms that actually catch
  platform-specific failures.
- The end-to-end suite was silently running Electron 43 instead of the app's
  own 34 — every e2e run was testing the wrong runtime.
- The numba JIT is disabled on the macOS CI legs, where the toolchain
  miscompiles its kernels and the crash lands wherever a jitted kernel happens
  to run.
- ``deapi`` is now a pinned pre-release dependency from PyPI rather than a git
  reference.

Documentation
-------------

- 34 standalone Markdown design and plan documents were deleted. They described
  an application that no longer existed, and a reader could not tell that from
  current guidance; the facts worth keeping moved into tests, docstrings and
  commit messages next to the code that makes them true.
- ``CLAUDE.md`` gained a code style section.
