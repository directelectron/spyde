"""Navigator sidecar cache (spyde.nav_sidecar) — save the computed navigator
beside the source file; a later open of the same (unchanged) file loads it
instead of re-reading the whole dataset.

Also pins the nav-flood deferral: the progressive navigator fill must not start
until the signal plot's first frame has painted (or the bounded wait expires),
so a cold large file's first frame is never starved of the disk by the fill.
"""
from __future__ import annotations

import os
import time

import numpy as np
import hyperspy.api as hs
import pytest

from spyde.nav_sidecar import (
    SIDECAR_SUFFIX, load_nav_sidecar, save_nav_sidecar, sidecar_path,
)


@pytest.fixture
def data_file(tmp_path):
    """A small real file standing in for the dataset (fingerprint source)."""
    p = tmp_path / "scan.mrc"
    p.write_bytes(b"\x00" * 4096)
    return str(p)


class TestSidecarRoundTrip:
    def test_save_then_load(self, data_file):
        nav = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert save_nav_sidecar(data_file, nav)
        assert os.path.exists(sidecar_path(data_file))
        out = load_nav_sidecar(data_file, (3, 4))
        assert out is not None
        np.testing.assert_array_equal(out, nav)

    def test_shape_mismatch_misses(self, data_file):
        save_nav_sidecar(data_file, np.ones((3, 4), dtype=np.float32))
        assert load_nav_sidecar(data_file, (4, 3)) is None
        assert load_nav_sidecar(data_file, (12,)) is None

    def test_modified_source_invalidates(self, data_file):
        save_nav_sidecar(data_file, np.ones((3, 4), dtype=np.float32))
        # Change the source file (size change → fingerprint mismatch).
        with open(data_file, "ab") as f:
            f.write(b"\x01")
        assert load_nav_sidecar(data_file, (3, 4)) is None

    def test_partial_fill_never_cached(self, data_file):
        nav = np.full((3, 4), np.nan, dtype=np.float32)
        nav[0] = 1.0
        assert not save_nav_sidecar(data_file, nav)
        assert not os.path.exists(sidecar_path(data_file))

    def test_missing_or_corrupt_sidecar(self, data_file):
        assert load_nav_sidecar(data_file, (3, 4)) is None
        with open(sidecar_path(data_file), "wb") as f:
            f.write(b"not a zip")
        assert load_nav_sidecar(data_file, (3, 4)) is None

    def test_1d_movie_navigator(self, data_file):
        nav = np.linspace(0, 1, 977).astype(np.float32)
        assert save_nav_sidecar(data_file, nav)
        out = load_nav_sidecar(data_file, (977,))
        np.testing.assert_array_equal(out, nav)


def _lazy_4d(tmp_path):
    """A small lazy 4-D signal saved to a real file (real fingerprint source)."""
    data = (np.random.RandomState(0).rand(4, 5, 8, 8) * 100).astype(np.float32)
    path = str(tmp_path / "scan4d.hspy")
    hs.signals.Signal2D(data).save(path, overwrite=True)
    sig = hs.load(path, lazy=True)
    return sig, path, data


def _wait_for(cond, timeout=20.0, interval=0.05):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(interval)
    return False


class TestSessionIntegration:
    def test_fill_saves_sidecar_and_reopen_skips_compute(
            self, window, tmp_path):
        session = window["window"]
        sig, path, data = _lazy_4d(tmp_path)

        tree = session._add_signal(sig, source_path=path)
        # The threaded progressive fill (deferred until the signal plot's first
        # frame paints) completes and writes the sidecar.
        assert _wait_for(lambda: os.path.exists(path + SIDECAR_SUFFIX)), \
            "progressive fill never wrote the navigator sidecar"
        expected_nav = data.sum(axis=(2, 3))
        cached = load_nav_sidecar(path, expected_nav.shape)
        np.testing.assert_allclose(cached, expected_nav, rtol=1e-5)

        # Reopen: the navigator must come from the sidecar — no pending
        # progressive compute, real (non-NaN) data immediately.
        sig2 = hs.load(path, lazy=True)
        tree2 = session._add_signal(sig2, source_path=path)
        assert tree2._pending_nav_dask is None, \
            "sidecar hit must not stash a progressive nav compute"
        nav_data = tree2.navigator_signals["base"][0].data
        assert np.all(np.isfinite(nav_data)), \
            "sidecar-loaded navigator must have no NaN placeholder"
        np.testing.assert_allclose(nav_data, expected_nav, rtol=1e-5)

    def test_no_sidecar_for_pseudo_paths(self, window):
        """Test/example loaders pass pseudo-paths — no sidecar, normal fill."""
        session = window["window"]
        s = hs.signals.Signal2D(
            np.random.RandomState(1).rand(3, 3, 8, 8).astype(np.float32)
        ).as_lazy()
        tree = session._add_signal(s, source_path="test_data")
        assert tree.source_path is None

    def test_override_navigator_never_uses_sidecar(self, window, tmp_path):
        """A navigator_override (e.g. vectors count map) shares the nav shape
        but is a DIFFERENT quantity — it must not be served from the sidecar."""
        session = window["window"]
        sig, path, data = _lazy_4d(tmp_path)
        # Plant a sidecar with recognisable wrong-for-override values.
        planted = np.full((4, 5), 777.0, dtype=np.float32)
        save_nav_sidecar(path, planted)

        override = hs.signals.Signal2D(
            np.arange(20, dtype=np.float32).reshape(4, 5))
        tree = session._add_signal(
            hs.load(path, lazy=True), source_path=path,
            navigator_override=override)
        nav_data = np.asarray(tree.navigator_signals["base"][0].data)
        assert not np.allclose(nav_data, planted), \
            "override navigator was wrongly replaced by the sidecar"


class TestNavFillDeferral:
    def test_fill_waits_for_first_signal_frame(self, window, tmp_path,
                                               monkeypatch):
        """The progressive fill must start only after a signal plot has data
        (or the bounded wait expires) — the first frame gets the disk first."""
        from spyde.signal_tree import BaseSignalTree

        started = {"t": None, "had_frame": None}
        orig = BaseSignalTree._start_progressive_nav_compute

        def spy(self, nav_dask=None):
            started["t"] = time.monotonic()
            started["had_frame"] = any(
                isinstance(p.current_data, np.ndarray) for p in self.signal_plots)
            return orig(self, nav_dask)

        monkeypatch.setattr(
            BaseSignalTree, "_start_progressive_nav_compute", spy)

        session = window["window"]
        sig, path, data = _lazy_4d(tmp_path)
        session._add_signal(sig, source_path=path)

        assert _wait_for(lambda: started["t"] is not None), \
            "progressive fill never started"
        assert started["had_frame"], \
            "fill started before the signal plot's first frame painted"
