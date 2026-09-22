"""The calibrated SPED-Ag dataset added to em-database at runtime.

``em_database.data.SPEDAg`` resolves to a Zenodo record whose copy of the scan
carries exactly half the true reciprocal calibration. Nothing raises on it —
every measured vector simply reaches a template matcher at half its length, so
no crystal fits and the orientation overlay draws simulated spots beside the
measured ones. These tests pin the addition that offers the corrected copy
alongside it, and the pointer it uses.
"""
from __future__ import annotations

import pytest

from spyde.external.emdatabase import sped_ag


class TestTheRecordItPointsAt:
    """The URL and checksum are the whole point of the entry: they are what is
    different from the one em-database ships."""

    def test_it_points_at_the_calibrated_record(self):
        assert sped_ag.DATASET["source"].endswith("/records/21790591/files")

    def test_it_carries_the_verified_checksum(self):
        """Verified by downloading, hashing, and reading the scale back as
        0.02672830388733737 Å⁻¹/px. The stale copy hashes to 8556346…."""
        assert sped_ag.DATASET["checksum"] == \
            "md5:964e638c1476e0337f0d474f2e691748"

    def test_it_is_the_same_file_name(self):
        assert sped_ag.DATASET["file"] == "SPED-Ag.zspy"

    def test_the_harness_loader_uses_this_record(self):
        """One declaration, not two: a second copy of a URL and a hash is
        exactly the pair that drifts. Read off the loader itself, not the
        module — the module mentions plenty of things the loader does not.
        """
        import inspect

        from spyde.backend._session_testharness import TestHarnessMixin

        source = inspect.getsource(TestHarnessMixin._load_test_data_sped_ag)
        assert "DATASET[" in source, \
            "the harness should read the record from the external module"
        assert "zenodo.org" not in source, \
            "the harness should not carry its own copy of the URL"


class TestRegistration:
    def test_apply_is_idempotent(self):
        """The external contract: safe to call twice."""
        first = sped_ag.apply()
        second = sped_ag.apply()
        assert first == second

    def test_it_registers_with_the_external_registry(self):
        """apply_all() is what runs it at startup, via the subpackage import."""
        from spyde.external import _PATCH_SUBPACKAGES

        assert "spyde.external.emdatabase" in _PATCH_SUBPACKAGES


class TestTheCatalogueOffersIt:
    """The Examples menu lists whatever ``em_database.data`` holds, so adding
    the class there is the whole integration."""

    @pytest.fixture(autouse=True)
    def _applied(self):
        if not sped_ag.apply():
            pytest.skip("em-database is not installed (it needs Python >= 3.12)")

    def test_it_takes_the_originals_place_in_the_menu(self):
        """One SPED-Ag in the menu, and it is the usable one. Listing both
        left the half-scale copy one click away under the familiar name, and
        that is the one that got clicked."""
        from spyde.backend import example_catalogue

        names = [key for key, _ in example_catalogue.datasets()]
        assert sped_ag.NAME in names
        assert "SPEDAg" not in names, "the stale copy must not be offered"

    def test_it_resolves_to_the_calibrated_record(self):
        from spyde.backend import example_catalogue

        dataset = example_catalogue.resolve(sped_ag.NAME)
        assert dataset is not None
        assert dataset.source == sped_ag.DATASET["source"]
        assert dataset.checksum == sped_ag.DATASET["checksum"]

    def test_the_old_key_loads_the_calibrated_record(self):
        """A tutorial or saved session that names SPEDAg keeps working, and
        gets the copy a crystal can be fitted to."""
        from spyde.backend import example_catalogue

        dataset = example_catalogue.resolve("SPEDAg")
        assert dataset is not None
        assert dataset.source == sped_ag.DATASET["source"]

    def test_the_original_still_points_at_the_stale_record(self):
        """Not a complaint — a tripwire. When em-database repoints SPEDAg this
        fails, which is the signal that this whole module can be deleted.
        """
        import em_database.data as data

        cls = getattr(data, "SPEDAg", None)
        original = cls() if cls is not None else None
        if original is None:
            pytest.skip("em-database no longer ships SPEDAg")
        assert "21790591" not in original.source, (
            "em-database has repointed SPEDAg at the calibrated record — "
            "delete spyde/external/emdatabase/sped_ag.py and this test")
