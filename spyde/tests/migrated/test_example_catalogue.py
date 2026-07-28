"""The Examples menu's catalogue (``spyde/backend/example_catalogue.py``).

The catalogue is rebuilt every time the menu opens, so the property that
matters most is that building it is CHEAP — it must not open a data file to
answer "what is in here?". The rest pins the three things the menu draws that
were not simply handed over by em-database: the technique grouping order, the
downloaded marker, and where a shape comes from.
"""
from __future__ import annotations

import os

import pytest

from spyde.backend import example_catalogue as ec

pytestmark = pytest.mark.skipif(not ec.available(),
                                reason="em-database is not installed "
                                       "(it requires Python >= 3.12)")


class TestCatalogue:
    def test_lists_datasets_grouped_by_technique(self):
        cat = ec.catalogue()
        assert cat["available"]
        assert cat["n_total"] > 5, "em-database returned almost nothing"
        techs = [g["technique"] for g in cat["groups"]]
        assert "4D-STEM" in techs
        assert len(techs) == len(set(techs)), "a technique was split in two"
        assert sum(len(g["items"]) for g in cat["groups"]) == cat["n_total"]

    def test_the_modalities_spyde_is_built_around_come_first(self):
        """4D-STEM before Cryo-EM, whatever order em-database enumerates in —
        the menu should open on the things this app is for."""
        techs = [g["technique"] for g in ec.catalogue()["groups"]]
        known = [t for t in techs if t in ec.TECHNIQUE_ORDER]
        assert known == sorted(known, key=ec.TECHNIQUE_ORDER.index)

    def test_every_entry_has_what_the_menu_draws(self):
        for group in ec.catalogue()["groups"]:
            for item in group["items"]:
                assert item["key"] and item["label"]
                assert item["technique"] == group["technique"]
                assert isinstance(item["downloaded"], bool)
                assert item["size"], f"{item['key']} has no size to show"

    def test_the_base_class_is_not_a_dataset(self):
        """``em_database.data`` also holds the DownloadableDataset base and an
        incidental ``Path`` import; neither is an example."""
        keys = {k for k, _ in ec.datasets()}
        assert "DownloadableDataset" not in keys
        assert "Path" not in keys

    def test_resolve_round_trips_a_key(self):
        key = ec.catalogue()["groups"][0]["items"][0]["key"]
        ds = ec.resolve(key)
        assert ds is not None and getattr(ds, "file", None)
        assert ec.resolve("NoSuchDataset") is None

    def test_downloaded_reflects_the_file_on_disk(self):
        """``filepath()`` returns the path or None — that IS the question the
        marker asks, so the two must not disagree."""
        for group in ec.catalogue()["groups"]:
            for item in group["items"]:
                assert item["downloaded"] == bool(item["path"])
                if item["downloaded"]:
                    assert os.path.exists(item["path"])


class TestBuildingItIsCheap:
    def test_the_catalogue_never_opens_a_data_file(self, monkeypatch):
        """The menu asks for this on every open. Reading five multi-GB stores
        inline made that a 7-second click, so the catalogue uses only declared
        and already-cached shapes — warm_shapes() does the reading, off the
        menu's path.
        """
        opened = []
        monkeypatch.setattr(ec, "_open_lazily",
                            lambda path: opened.append(path) or (_ for _ in ()).throw(
                                AssertionError(f"catalogue opened {path}")))
        cat = ec.catalogue()
        assert cat["n_total"] > 0
        assert not opened

    def test_data_dir_is_reported_even_when_nothing_is_downloaded(self):
        path = ec.data_dir()
        assert path and os.path.isabs(path)


class TestShape:
    def test_declared_shape_wins_over_reading_the_file(self, monkeypatch):
        """When em-database's YAML carries a shape, it is authoritative — the
        menu can then show a shape BEFORE anything is downloaded, and no file
        is opened to find one."""
        class _Declared:
            metadata = {"technique": "4D-STEM", "shape": {"navigation": (32, 32),
                                                          "signal": (256, 256)}}
            data_size = "1 GB"
            file = "x.zspy"
            description = ""
            license = ""
            source = ""

            def filepath(self):
                return None

        monkeypatch.setattr(ec, "read_shape",
                            lambda p: pytest.fail("read a file despite a declared shape"))
        entry = ec.entry("Declared", _Declared())
        assert entry["shape"] == "32×32 | 256×256"
        assert entry["downloaded"] is False

    def test_an_undownloaded_dataset_without_a_declared_shape_shows_none(self):
        class _Bare:
            metadata = {"technique": "EELS"}
            data_size = "1 kB"
            file = "y.hspy"
            description = ""
            license = ""
            source = ""

            def filepath(self):
                return None

        assert ec.entry("Bare", _Bare())["shape"] is None

    def test_record_shape_stamps_what_a_load_already_knows(self, tmp_path,
                                                          monkeypatch):
        """Loading an example gives us the signal — so the shape is free and
        exact, and the menu shows it from then on without reopening anything."""
        import hyperspy.api as hs
        import numpy as np

        monkeypatch.setattr(ec, "data_dir", lambda: str(tmp_path))
        monkeypatch.setattr(ec, "_shape_cache", None, raising=False)
        target = tmp_path / "example.hspy"
        target.write_bytes(b"not really a file")

        sig = hs.signals.Signal2D(np.zeros((4, 5, 6, 7), np.float32))
        ec.record_shape(str(target), sig)
        assert ec._cached_shape(str(target)) == "5×4 | 7×6"

    def test_shape_of_signal_is_nav_then_signal(self):
        import hyperspy.api as hs
        import numpy as np
        sig = hs.signals.Signal2D(np.zeros((3, 8, 9), np.float32))
        assert ec.shape_of_signal(sig) == "3 | 9×8"
