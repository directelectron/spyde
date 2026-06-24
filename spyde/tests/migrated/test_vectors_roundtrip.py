"""
Diffraction-vectors save/load round-trip.

A Find-Vectors result must serialise as the *vectors themselves* (a dense
flat-buffer ``DenseDiffractionVectors`` signal in a ``.zspy`` store) — NOT the
rendered-disk image — and reload back into a Find-Vectors result tree with the
vectors reattached, the count-map navigator filled, and the vector toolbar
actions unlocked.

Covers:
  - to_dense_signal / from_dense_signal lossless conversion (4D + 5D)
  - .zspy round-trip through hyperspy preserves flat buffer + calibration
  - session save picks the dense carrier (not the lazy rendered image)
  - session load reconstructs a result tree with tree.diffraction_vectors
  - the result root stays a cheap placeholder (no to_rendered_dask materialised)
"""
import os

import numpy as np
import hyperspy.api as hs
import pytest

from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors, _AxisLite
from spyde.signals.dense_diffraction_vectors import (
    DenseDiffractionVectors,
    to_dense_signal,
    from_dense_signal,
    is_dense_vectors_signal,
    SIGNAL_TYPE,
)


def _sig_axes():
    return [_AxisLite(scale=0.01, offset=-0.5, size=32, units="1/A", name="kx"),
            _AxisLite(scale=0.01, offset=-0.5, size=32, units="1/A", name="ky")]


def _nav_axes(n):
    names = ("x", "y") if n == 2 else ("t", "x", "y")
    return [_AxisLite(scale=2.0 + i, offset=float(i), size=4, units="nm", name=nm)
            for i, nm in enumerate(names)]


def _make_vecs_4d(ny=3, nx=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            for _ in range(int(rng.integers(1, 4))):
                rows.append([ix, iy, rng.uniform(-0.4, 0.4),
                             rng.uniform(-0.4, 0.4), -1.0, rng.uniform(1, 5)])
    flat = np.array(rows, dtype=np.float32)
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(ny, nx), sig_shape=(32, 32),
        sig_axes=_sig_axes(), kernel_radius_px=5.0, kernel_radius_data=0.05,
        params={"method": "dog", "threshold": 10.0}, nav_axes=_nav_axes(2))


def _make_vecs_5d(nt=2, ny=3, nx=4, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for it in range(nt):
        for iy in range(ny):
            for ix in range(nx):
                for _ in range(int(rng.integers(0, 3))):
                    rows.append([ix, iy, rng.uniform(-0.4, 0.4),
                                 rng.uniform(-0.4, 0.4), float(it), rng.uniform(1, 5)])
    flat = np.array(rows, dtype=np.float32)
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(nt, ny, nx), sig_shape=(32, 32),
        sig_axes=_sig_axes(), kernel_radius_px=5.0, kernel_radius_data=0.05,
        params={"method": "nxcorr"}, nav_axes=_nav_axes(3))


def _assert_vecs_equal(a, b):
    assert np.array_equal(a.flat_buffer, b.flat_buffer)
    assert a.full_nav_shape == b.full_nav_shape
    assert a.nav_shape == b.nav_shape
    assert a.kernel_radius_px == b.kernel_radius_px
    assert a.kernel_radius_data == b.kernel_radius_data
    assert np.array_equal(a.count_map(), b.count_map())
    for ax_a, ax_b in zip(a.sig_axes, b.sig_axes):
        assert ax_a.scale == ax_b.scale and ax_a.offset == ax_b.offset
        assert ax_a.units == ax_b.units and ax_a.name == ax_b.name
    for ax_a, ax_b in zip(a.nav_axes, b.nav_axes):
        assert ax_a.scale == ax_b.scale and ax_a.units == ax_b.units


class TestDenseConversion:
    def test_to_from_dense_4d(self):
        vecs = _make_vecs_4d()
        sig = to_dense_signal(vecs)
        assert isinstance(sig, DenseDiffractionVectors)
        assert sig.data.shape == (len(vecs.flat_buffer), 6)
        assert is_dense_vectors_signal(sig)
        _assert_vecs_equal(vecs, from_dense_signal(sig))

    def test_to_from_dense_5d(self):
        vecs = _make_vecs_5d()
        sig = to_dense_signal(vecs)
        v2 = from_dense_signal(sig)
        _assert_vecs_equal(vecs, v2)
        assert v2.n_time == vecs.n_time

    def test_signal_type_tag(self):
        sig = to_dense_signal(_make_vecs_4d())
        assert sig.metadata.get_item("Signal.signal_type") == SIGNAL_TYPE

    def test_column_names_in_metadata(self):
        """The dense (N,6) buffer is self-documenting for external readers
        (mirrors pyxem DiffractionVectors2D's VectorMetadata.column_names)."""
        sig = to_dense_signal(_make_vecs_4d())
        cn = list(sig.metadata.get_item("SpyDE.DiffractionVectors.column_names"))
        assert cn == ["nav_x", "nav_y", "kx", "ky", "time", "intensity"]

    def test_empty_vectors_roundtrip(self, tmp_path):
        """A result with zero vectors must still save/load (zarr can't chunk a
        zero-length axis, so we write a sentinel row + n_vectors=0)."""
        empty = SpyDEDiffractionVectors.from_arrays(
            flat_buffer=np.zeros((0, 6), dtype=np.float32),
            full_nav_shape=(3, 4), sig_shape=(32, 32), sig_axes=_sig_axes(),
            kernel_radius_px=5.0, kernel_radius_data=0.05, nav_axes=_nav_axes(2))
        p = str(tmp_path / "empty.zspy")
        to_dense_signal(empty).save(p)
        v2 = from_dense_signal(hs.load(p))
        assert v2.flat_buffer.shape == (0, 6)
        assert int(v2.count_map().sum()) == 0
        assert v2.count_map().shape == (3, 4)

    def test_non_vectors_signal_rejected(self):
        s = hs.signals.Signal2D(np.zeros((4, 4), dtype=np.float32))
        assert not is_dense_vectors_signal(s)
        with pytest.raises(ValueError):
            from_dense_signal(s)


class TestZspyRoundTrip:
    def test_zspy_4d(self, tmp_path):
        vecs = _make_vecs_4d()
        p = str(tmp_path / "v.zspy")
        to_dense_signal(vecs).save(p)
        loaded = hs.load(p)
        assert is_dense_vectors_signal(loaded)
        assert loaded.metadata.get_item("Signal.signal_type") == SIGNAL_TYPE
        _assert_vecs_equal(vecs, from_dense_signal(loaded))

    def test_zspy_5d(self, tmp_path):
        vecs = _make_vecs_5d()
        p = str(tmp_path / "v5.zspy")
        to_dense_signal(vecs).save(p)
        _assert_vecs_equal(vecs, from_dense_signal(hs.load(p)))

    def test_rendered_frame_after_reload(self, tmp_path):
        """A reloaded vectors result can still render frames on demand."""
        vecs = _make_vecs_4d()
        p = str(tmp_path / "v.zspy")
        to_dense_signal(vecs).save(p)
        v2 = from_dense_signal(hs.load(p))
        f = v2.render_frame(1, 2)
        assert f.shape == (32, 32)
        assert np.isfinite(f).all()


class TestSessionRoundTrip:
    def test_save_picks_dense_carrier(self, stem_4d_dataset, tmp_path, monkeypatch):
        """session._save_signal on a vectors result writes the dense carrier,
        not the rendered image."""
        from spyde.actions.find_vectors_action import build_vectors_result_tree
        session = stem_4d_dataset["window"]
        vecs = _make_vecs_4d()
        tree = build_vectors_result_tree(session, vecs, title="V")

        # Find a signal plot on the result tree.
        plot = next(p for p in session._plots
                    if getattr(p, "signal_tree", None) is tree
                    and getattr(getattr(p, "plot_state", None), "current_signal", None) is not None)

        saved = {}

        def fake_thread(signal, path, name):
            saved["signal"] = signal
            saved["path"] = path

        monkeypatch.setattr(session, "_save_signal_thread", fake_thread)
        # _save_signal spawns a thread; patch Thread to run inline.
        import spyde.backend.session as sm
        monkeypatch.setattr(sm.threading, "Thread",
                            lambda target, args, daemon, name: type(
                                "T", (), {"start": lambda self: target(*args)})())

        session._save_signal(str(tmp_path / "out.zspy"), plot)
        assert isinstance(saved["signal"], DenseDiffractionVectors)
        v2 = from_dense_signal(saved["signal"])
        _assert_vecs_equal(vecs, v2)

    def test_load_reconstructs_result_tree(self, window, tmp_path):
        session = window["window"]
        vecs = _make_vecs_4d()
        p = str(tmp_path / "v.zspy")
        to_dense_signal(vecs).save(p)

        n_before = len(session.signal_trees)
        loaded = hs.load(p)
        handled = session._open_if_dense_vectors(loaded, p)
        assert handled is True
        assert len(session.signal_trees) == n_before + 1
        tree = session.signal_trees[-1]
        assert getattr(tree, "diffraction_vectors", None) is not None
        _assert_vecs_equal(vecs, tree.diffraction_vectors)

    def test_result_root_is_placeholder_not_rendered(self, window, tmp_path):
        """The reconstructed result keeps a cheap zero placeholder root — the
        rendered disks come from render_frame on demand, never materialised
        into tree.root.data."""
        session = window["window"]
        vecs = _make_vecs_4d()
        p = str(tmp_path / "v.zspy")
        to_dense_signal(vecs).save(p)
        session._open_if_dense_vectors(hs.load(p), p)
        root = session.signal_trees[-1].root
        # Placeholder is all-zeros (disks are rendered per-frame, not stored).
        assert float(np.asarray(root.data).max()) == 0.0
