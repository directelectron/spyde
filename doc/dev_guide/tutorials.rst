Writing a tutorial
==================

A technique tutorial is authored **once** and rendered in four places, so the
app, the website and the docs can never drift apart:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Rendering
     - Built by
   * - The in-app guided tour (a coachmark walkthrough over the live UI)
     - ``electron/src/renderer/src/components/Tour.tsx``
   * - **Help → <technique> → Info…** (background + further reading)
     - ``electron/src/renderer/src/components/GuideInfoDialog.tsx``
   * - The docs pages under :ref:`tutorials-index`
     - ``scripts/gen_guide_docs.mjs`` → ``doc/tutorials/*.rst``
   * - The standalone docs website (with interactive embeds)
     - ``docs-site/src/DocsApp.tsx``

The source
----------

One file per technique in ``guides/`` at the repository root, registered in
``guides/index.ts``. The format is ``guides/types.ts``:

``steps``
    The click-by-click walkthrough. Each step names the UI element it is about
    by its stable ``data-testid`` (``anchor``), carries a short markdown
    ``body``, and optionally an ``image`` (a screenshot) and a ``drive`` (a
    screenplay for reaching the step when generating screenshots).

``autoload``
    A backend action run once when the tour opens — in practice
    ``tutorial_load`` with the name of a small, instant, no-download dataset
    from ``spyde/backend/tutorial_data.py``. This is what makes a tutorial
    **self-contained**: it brings its own data rather than assuming the user
    already has the right thing open.

``info``
    The background half: a ``blurb`` on what the technique is and when to reach
    for it, plus ``links`` to the upstream documentation. It becomes the tour's
    final "More info" step, the Help → Info… dialog, and the
    **More information** section of the docs page.

Guidelines
----------

* **Link, don't restate.** SpyDE wraps pyxem, HyperSpy, eXSpy, kikuchipy and
  orix; those projects own the science and document it far better than a
  walkthrough can. Put their pages in ``info.links`` with a one-line note on
  what the reader will find there, and keep the blurb to what SpyDE's own
  interface does with the method.
* **Keep the dataset tiny.** A tutorial must load in a couple of seconds with no
  download. See the size assertions in
  ``spyde/tests/migrated/test_tutorial_data.py``.
* **Everything the tour opens is closed again on exit.** The Tour brackets
  itself with ``tutorial_session_begin`` / ``tutorial_close_all``; between those,
  ``Session._add_signal`` records every non-file-backed tree so the teardown gets
  the result windows too, not just the dataset. You do not need to do anything
  for this — but do not open windows from outside that lifecycle and expect them
  to be cleaned up.

Regenerating the docs pages
---------------------------

``doc/tutorials/*.rst`` is generated and **committed**, so a docs build needs
only Python. After editing any guide, regenerate and commit the result::

    node scripts/gen_guide_docs.mjs            # write doc/tutorials/
    node scripts/gen_guide_docs.mjs --check    # fail if it would change

(The script bundles the TypeScript guides with the ``esbuild`` already installed
under ``electron/node_modules``, so run ``npm install`` in ``electron/`` first.)

Screenshots are **not** copied into ``doc/``. They are captured by the
Playwright run ``electron/tests/guide_screenshots.spec.ts``, which walks each
step's ``drive`` and writes into ``docs-site/public/media/<guide>/``;
``doc/conf.py`` mirrors that tree into ``doc/tutorials/media/`` at build time. A
step whose screenshot has not been captured simply renders without one::

    cd electron
    SPYDE_E2E_REAL=1 npx playwright test guide_screenshots.spec.ts --project=electron-real
