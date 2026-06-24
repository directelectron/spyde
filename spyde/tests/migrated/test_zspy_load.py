"""
.zspy / .zarr datasets are DIRECTORY stores (Zarr nested groups), not files.

Regression: the load path gated on ``os.path.isfile`` and sized with
``os.path.getsize``, both of which reject/raise on a directory — so a .zspy
folder was reported "File not found" and never reached ``hs.load``. The path-type
helpers now treat a .zspy/.zarr directory as a present dataset.
"""
from __future__ import annotations

import numpy as np
import hyperspy.api as hs

from spyde.backend.session import (
    Session, _is_supported_dataset_path, _dataset_size_bytes, _path_ext,
    SUPPORTED_EXTS,
)


def _write_zspy_4d(tmp_path, name="scan", nav=(4, 5), sig=(8, 8)):
    ny, nx = nav[1], nav[0]
    s = hs.signals.Signal2D(np.zeros((ny, nx, sig[1], sig[0]), dtype=np.float32))
    s.set_signal_type("electron_diffraction")
    p = tmp_path / f"{name}.zspy"
    s.save(str(p))
    return str(p)


class TestZspyPathHelpers:
    def test_zspy_in_supported_exts(self):
        assert ".zspy" in SUPPORTED_EXTS

    def test_zspy_dir_is_supported_dataset(self, tmp_path):
        p = _write_zspy_4d(tmp_path)
        import os
        assert os.path.isdir(p) and not os.path.isfile(p)   # it's a folder
        assert _path_ext(p) == ".zspy"
        assert _is_supported_dataset_path(p) is True        # the fix

    def test_missing_zspy_dir_is_not_supported(self, tmp_path):
        assert _is_supported_dataset_path(str(tmp_path / "nope.zspy")) is False

    def test_regular_file_still_works(self, tmp_path):
        f = tmp_path / "a.hspy"
        f.write_bytes(b"x")
        assert _is_supported_dataset_path(str(f)) is True

    def test_dataset_size_walks_zspy_dir(self, tmp_path):
        p = _write_zspy_4d(tmp_path)
        # A directory store has real bytes on disk (chunks); size must be > 0 and
        # must NOT raise (os.path.getsize would raise on the dir).
        assert _dataset_size_bytes(p) > 0


class TestZspyOpen:
    def test_open_zspy_folder_loads(self, window, tmp_path):
        session = window["window"]
        p = _write_zspy_4d(tmp_path, nav=(4, 5))
        # .zspy is self-describing (correct axes from the store) → it opens
        # DIRECTLY, no nav-shape prompt (that's only for ambiguous raw .mrc/.tif).
        session._load_file_thread(p)
        assert getattr(session, "_pending_load", None) is None   # no prompt
        assert len(session.signal_trees) == 1
        root = session.signal_trees[0].root
        assert tuple(root.axes_manager.navigation_shape) == (4, 5)

    def test_zspy_is_self_describing(self):
        assert Session._is_self_describing("x.zspy")
        assert Session._is_self_describing("x.hspy")
        assert not Session._is_self_describing("x.mrc")
        assert not Session._is_self_describing("x.tif")

    def test_open_missing_zspy_emits_error(self, window, tmp_path):
        session = window["window"]
        session.open_file(str(tmp_path / "ghost.zspy"))   # synchronous validation
        assert len(session.signal_trees) == 0
        msgs = window["messages"]
        assert any(
            isinstance(m, dict) and m.get("type") == "error"
            and "not found" in str(m.get("text", "")).lower()
            for m in msgs
        )


class TestNavigatorFutureGuard:
    """A lazy navigator's data can be a dask Future — or a length-1 OBJECT ndarray
    wrapping one — before its progressive compute lands. Painting that sent a
    Future into anyplotlib's np.asarray(data, dtype=float) → 'float() argument ...
    not Future', which crashed loading a 5-D .zspy under a distributed client.
    _set_array must skip non-numeric/object data instead of raising."""

    def _a_plot(self, session):
        for p in session._plots:
            if getattr(p, "_set_array", None) is not None:
                return p
        return None

    def test_set_array_skips_future_bearing_object_array(self, stem_4d_dataset):
        plot = self._a_plot(stem_4d_dataset["window"])
        assert plot is not None

        class _FakeFuture:
            def __float__(self):
                raise TypeError("float() argument must be a string or a real "
                                "number, not 'Future'")

        bad = np.empty((1,), dtype=object)
        bad[0] = _FakeFuture()
        # Must NOT raise (previously crashed in np.asarray(..., dtype=float)).
        plot._set_array(bad)

    def test_set_array_skips_bare_future(self, stem_4d_dataset):
        plot = self._a_plot(stem_4d_dataset["window"])

        class _FakeFuture:
            pass

        plot._set_array(_FakeFuture())     # not an ndarray → skipped, no raise

    def test_set_array_still_paints_real_array(self, stem_4d_dataset):
        plot = self._a_plot(stem_4d_dataset["window"])
        # A normal 2-D float frame paints without error.
        plot.needs_auto_level = True
        plot._set_array(np.ones((8, 8), dtype=np.float32))


class TestZspySaveDefault:
    """_save_signal defers the actual write to a daemon thread (_save_signal_thread)
    so a big lazy save doesn't block the loop. Tests drive the thread body directly
    for synchronous I/O assertions, and _save_signal for the resolution/defaulting."""

    def _first_plot(self, session):
        for p in session._plots:
            if getattr(getattr(p, "plot_state", None), "current_signal", None) is not None:
                return p
        return None

    def _sig(self, session):
        return self._first_plot(session).plot_state.current_signal

    def test_save_signal_writes_zspy_folder(self, stem_4d_dataset, tmp_path):
        session = stem_4d_dataset["window"]
        sig = self._sig(session)
        out = str(tmp_path / "saved.zspy")
        session._save_signal_thread(sig, out, "saved.zspy")
        import os
        assert os.path.isdir(out)                       # Zarr store = a directory
        r = hs.load(out, lazy=True)
        assert r.data.ndim == sig.data.ndim
        msgs = stem_4d_dataset["messages"]
        assert any(isinstance(m, dict) and m.get("type") == "saved" for m in msgs)

    def test_no_extension_defaults_to_zspy(self, stem_4d_dataset, tmp_path):
        # _save_signal does the extension-defaulting then spawns the writer thread;
        # join it so the assertion is deterministic.
        import threading
        session = stem_4d_dataset["window"]
        plot = self._first_plot(session)
        base = str(tmp_path / "noext")
        session._save_signal(base, plot)
        for t in threading.enumerate():
            if t.name.startswith("save-"):
                t.join(30)
        import os
        assert os.path.isdir(base + ".zspy")            # .zspy appended
        msgs = stem_4d_dataset["messages"]
        assert any(
            isinstance(m, dict) and m.get("type") == "saved"
            and str(m.get("path", "")).endswith(".zspy")
            for m in msgs
        )

    def test_explicit_hspy_extension_kept(self, stem_4d_dataset, tmp_path):
        session = stem_4d_dataset["window"]
        sig = self._sig(session)
        out = str(tmp_path / "explicit.hspy")
        session._save_signal_thread(sig, out, "explicit.hspy")
        import os
        assert os.path.isfile(out)                      # .hspy = a single file
        assert not os.path.exists(out + ".zspy")        # not coerced to zspy

    def test_save_resolves_active_window_when_no_plot(self, stem_4d_dataset, tmp_path):
        # The File→Save menu sends no window id; _save_signal must fall back to the
        # active window (or the sole signal plot) instead of "no active plot".
        import threading
        session = stem_4d_dataset["window"]
        plot = self._first_plot(session)
        session._active_window_id = plot.window_id
        out = str(tmp_path / "viaactive.zspy")
        session._save_signal(out, None)                 # NO plot passed
        for t in threading.enumerate():
            if t.name.startswith("save-"):
                t.join(30)
        import os
        assert os.path.isdir(out)
        # And no "no active plot" error was emitted.
        msgs = stem_4d_dataset["messages"]
        assert not any(
            isinstance(m, dict) and m.get("type") == "error"
            and "active" in str(m.get("text", "")).lower()
            for m in msgs
        )
