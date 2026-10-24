"""Crystal structures from the Crystallography Open Database build a library.

Reported as "a barium titanate from COD never creates a library — it is stuck
at 0 rotations". Two separate faults sat behind it:

* orix's ``Phase.from_cif`` turns the lattice of any cell whose ``a`` axis is
  not already along x (hexagonal, trigonal, monoclinic) without moving the
  atoms, so the structure loses its own symmetry. Hexagonal BaTiO3 came out
  as Ama2, rhombohedral as P1.
* The matcher's plan build holds a (zone images × zones) table, quadratic in
  the number of zone axes. A low-symmetry phase at the default 1° step
  (orthorhombic BaTiO3: ~20 600 zone axes, ~20 GB) cannot be built, and the
  failure was swallowed, leaving a "ready" library with 0 orientations.

Fixtures are trimmed COD entries (cell, symmetry, atom sites):
1507756 (tetragonal P4mm), 1507757 (cubic Pm-3m), 9014074 (rhombohedral R3m),
9014492 (orthorhombic Amm2) and 2009488 (hexagonal P6_3/mmc).
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

pytest.importorskip("ase", reason="quantem's crystal.py needs ase + spglib")
pytest.importorskip("spglib", reason="quantem's crystal.py needs ase + spglib")

from spyde.tests.migrated.conftest import close_session, make_session  # noqa: E402
from spyde.tests.migrated.test_vector_orientation_om import (  # noqa: E402
    _make_vectors_tree, _signal_plot, _wait,
)

CIF_DIR = os.path.join(os.path.dirname(__file__), "data", "cif")


def _cif(name: str) -> str:
    return os.path.join(CIF_DIR, f"BaTiO3_{name}.cif")


#: polymorph → (fixture, International Tables space group number)
POLYMORPHS = {
    "tetragonal": ("tetragonal_COD1507756", 99),
    "cubic": ("cubic_COD1507757", 221),
    "rhombohedral": ("rhombohedral_COD9014074", 160),
    "orthorhombic": ("orthorhombic_COD9014492", 38),
    "hexagonal": ("hexagonal_COD2009488", 194),
}


def _space_group_number(crystal) -> int:
    """The number in quantem's "P4mm (99)" spelling."""
    return int(str(crystal.spacegroup).rsplit("(", 1)[1].rstrip(")"))


class _Messages:
    """What the vector-orientation wizard tells the user."""

    def __init__(self, monkeypatch):
        import spyde.actions.vector_orientation_om as vom
        self.sent, self.status, self.errors = [], [], []
        monkeypatch.setattr(vom, "emit", self.sent.append)
        monkeypatch.setattr(vom, "emit_status", self.status.append)
        monkeypatch.setattr(vom, "emit_error", self.errors.append)

    def library_ready(self):
        return next((m for m in self.sent if m.get("type") == "vom_library_ready"), None)


@pytest.fixture(scope="module")
def session():
    session = make_session()
    try:
        yield session
    finally:
        close_session(session)


@pytest.fixture(scope="module")
def vectors_tree(session):
    return _make_vectors_tree(session)


def _generate(session, tree, monkeypatch, cif, resolution):
    import spyde.torch_device as torch_device
    monkeypatch.setattr(torch_device, "select_device", lambda: None)
    from spyde.actions.vector_orientation_om import vom_generate_library
    messages = _Messages(monkeypatch)
    tree._vom_wizard = None
    vom_generate_library(session, _signal_plot(session, tree), {
        "cif_path": cif, "accelerating_voltage": 200.0,
        "resolution": resolution, "minimum_intensity": 1e-4,
    })
    assert _wait(lambda: messages.library_ready() is not None, timeout=180), \
        "the library build never answered"
    return messages


class TestEveryPolymorphBuildsALibrary:
    @pytest.mark.parametrize("polymorph", sorted(POLYMORPHS))
    def test_the_library_has_orientations_and_the_file_symmetry(
            self, polymorph, session, vectors_tree, monkeypatch):
        fixture, space_group = POLYMORPHS[polymorph]
        messages = _generate(session, vectors_tree, monkeypatch, _cif(fixture), 4.0)
        ready = messages.library_ready()
        assert ready["ok"], ready
        assert ready["n_templates"] > 0, messages.status
        crystal = vectors_tree._vom_wizard.fitter._maps[0].crystal
        assert _space_group_number(crystal) == space_group


class TestTheAtomsStayWhereTheFilePutsThem:
    def test_hexagonal_barium_sits_on_one_third_two_thirds(self):
        from spyde.crystal_phase import phase_from_cif
        phase = phase_from_cif(_cif("hexagonal_COD2009488"))
        positions = np.array([atom.xyz for atom in phase.structure
                              if str(atom.element).startswith("Ba")])
        on_axis = np.isclose(positions[:, 0], 1 / 3, atol=1e-6) & \
            np.isclose(positions[:, 1], 2 / 3, atol=1e-6)
        assert on_axis.any(), positions

    def test_an_orthogonal_cell_is_unchanged(self):
        from orix.crystal_map import Phase
        from spyde.crystal_phase import phase_from_cif
        cif = _cif("tetragonal_COD1507756")
        ours = np.array([atom.xyz for atom in phase_from_cif(cif).structure])
        theirs = np.array([atom.xyz for atom in Phase.from_cif(cif).structure])
        assert np.allclose(ours, theirs)


class TestALibraryTooBigForMemory:
    """Orthorhombic BaTiO3 at a fine step needs more memory than the machine
    has; the plan is built at the finest step that fits, and the user is told."""

    def test_the_step_is_coarsened_and_the_user_told(
            self, session, vectors_tree, monkeypatch):
        import psutil
        available = 1.5e9
        monkeypatch.setattr(psutil, "virtual_memory",
                            lambda: types.SimpleNamespace(available=available))
        messages = _generate(session, vectors_tree, monkeypatch,
                             _cif("orthorhombic_COD9014492"), 2.0)
        ready = messages.library_ready()
        assert ready["ok"], ready
        assert ready["n_templates"] > 0
        step = vectors_tree._vom_wizard.fitter._maps[0].zone_step_deg
        assert step > 2.0
        assert ready.get("note"), "the coarser step was not reported"
        assert any("memory" in text for text in messages.status)

    def test_a_phase_that_never_fits_is_an_error(
            self, session, vectors_tree, monkeypatch):
        import psutil
        monkeypatch.setattr(psutil, "virtual_memory",
                            lambda: types.SimpleNamespace(available=1e6))
        messages = _generate(session, vectors_tree, monkeypatch,
                             _cif("orthorhombic_COD9014492"), 2.0)
        ready = messages.library_ready()
        assert not ready["ok"]
        assert "memory" in ready["error"]
        assert messages.errors


class TestAFailedPlanIsReported:
    def test_a_plan_that_raises_is_not_a_ready_library(
            self, session, vectors_tree, monkeypatch):
        import spyde.actions.vector_orientation_quantem as quantem

        def refuse(*args, **kwargs):
            raise RuntimeError("not enough memory: you tried to allocate 26 GB")

        monkeypatch.setattr(quantem, "SinglePatternFitter", refuse)
        messages = _generate(session, vectors_tree, monkeypatch,
                             _cif("cubic_COD1507757"), 4.0)
        ready = messages.library_ready()
        assert not ready["ok"], "a library with no plan was reported ready"
        assert "26 GB" in ready["error"]
        assert messages.errors
