"""quantem — correlation-based orientation matching (ACOM), vendored.

The matcher is Ophus et al., Microsc. Microanal. 28, 390 (2022): sparse polar
correlation of measured Bragg peaks against a zone-axis template library, with
the in-plane angle as an FFT axis. SpyDE's own vector orientation mapping
(``spyde/actions/vector_orientation*.py``) is a convergent implementation of
the same idea, so this is here to be measured against it, one piece at a time.

**Why a copy rather than a dependency.** quantem is on PyPI, but its
``diffraction`` subpackage is not in any release — upstream ``dev`` holds only
an ``__init__.py``, and the whole ACOM stack lives on a fork branch that is
landing upstream in pieces. Installing ``quantem`` gets none of it. The
released package also pulls optuna, tensorboard, torchvision and friends, none
of which orientation matching needs. MIT licensed; see ``LICENSE-quantem``.

**Nothing in here imports SpyDE, and the files are upstream's byte for byte
apart from the import rewrites in** ``_sync.py``. Keep it that way: it is what
makes a re-sync a diff to read rather than a merge to do, and what lets the
directory be deleted outright once the code is released upstream. Everything
SpyDE-shaped — our vectors, our phases, our result container — lives in the
adapter at ``spyde/actions/vector_orientation_quantem.py``.

Vendored: the closure of ``OrientationMap.match_orientations`` and
``.refine_orientations``. Deliberately not vendored, and so raising ImportError
if reached:

``calculate_strain`` / ``plot_orientation`` / ``plot_pole_figure``
    Need upstream's ``StrainMap``, ``Dataset2d`` and matplotlib visualisation
    modules. SpyDE has its own strain plumbing and IPF colouring; the ~20-line
    affine solve inside ``calculate_strain`` belongs in the adapter when we
    want it.
``calculate_dynamical_structure_factors``
    Bloch-wave absorptive potentials, another 3900 lines.
``match_residual``
    Needs upstream's ragged ``Vector`` container to build residual peaks. The
    adapter duck-types the read-only part of that container, not its
    constructors — see :class:`_compat.Vector`.

Peak finding is emphatically not vendored. SpyDE's disk detection stays.

Requires ``ase`` and ``spglib`` (``crystal.py``), currently declared in the
``tests`` extra because nothing shipped imports this package yet.
"""
