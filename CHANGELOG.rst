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
