"""
Tests for the SpyDE math console execution engine (spyde.backend.console).

Qt-free: they build a real Session via the conftest fixtures + captured_messages
and drive the console through the SAME dispatch the renderer uses. The console
runs cells on its own daemon thread, so the helpers poll the captured-message list
for the expected console_result / console_vars rather than sleeping a fixed time.

Coverage:
  * expression echo value + out<N> registration
  * assignment registers the name, echoes nothing (but still a chip)
  * multi-statement code with a trailing expression
  * captured stdout
  * error → ok:false + traceback, namespace intact afterwards
  * persistence across execs
  * np / hs / da available in the namespace
  * signal auto-binding: sanitized title name + positional s1 alias, with
    window_ids in console_vars
  * binding removed when the tree is closed
  * `s1 > 100` on a LAZY signal stays lazy — a compute() guard proves the engine
    never materialises the dataset
  * materialise an ndarray → a new tree with the right shape/class
  * materialise a lazy result stays lazy
  * completion returns prefix matches incl. one attribute level
  * console_vars schema exact
"""
from __future__ import annotations

import time

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest
from unittest.mock import patch


# ── helpers ──────────────────────────────────────────────────────────────────


def _console(session):
    return session.console


def _exec(session, msgs, code, timeout=6.0):
    """Run a cell and return its console_result message (waits for the console
    thread). exec_ids are monotonically assigned per call so results don't cross."""
    exec_id = getattr(_exec, "_next", 1)
    _exec._next = exec_id + 1
    session.console.submit_exec(code, exec_id)
    return _wait_for(
        msgs, lambda m: m.get("type") == "console_result"
        and m.get("exec_id") == exec_id, timeout=timeout,
    )


def _wait_for(msgs, pred, timeout=6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for m in msgs:
            if pred(m):
                return m
        time.sleep(0.01)
    return None


def _latest_vars(msgs):
    """The most recent console_vars message (the full current list)."""
    vs = [m for m in msgs if m.get("type") == "console_vars"]
    return vs[-1] if vs else None


def _wait_vars(session, msgs, pred, timeout=6.0):
    """Poke a binding refresh and wait for a console_vars satisfying pred."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        cv = _latest_vars(msgs)
        if cv is not None and pred(cv):
            return cv
        time.sleep(0.02)
    return _latest_vars(msgs)


def _lazy_4d(nav=(3, 4), sig=(8, 8)):
    """A lazy 4D-STEM signal (dask-backed) for the no-compute tests."""
    ny, nx = nav
    ky, kx = sig
    arr = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    arr[:, :, ky // 2, kx // 2] = 200.0
    d = da.from_array(arr, chunks=(1, 1, ky, kx))
    s = hs.signals.Signal2D(d).as_lazy()
    s.set_signal_type("electron_diffraction")
    return s


# ── expression echo + out registration ───────────────────────────────────────


class TestEchoAndRegistration:
    def test_expression_echoes_value_and_registers_out(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "1 + 2")
        assert res is not None
        assert res["ok"] is True
        assert res["value_repr"] == "3"
        assert res["error"] is None and res["traceback"] is None
        assert res["result"] is not None
        assert res["result"]["name"] == "out1"
        assert res["result"]["kind"] == "scalar"
        # out1 is now usable as a variable in a later cell.
        res2 = _exec(session, msgs, "out1 * 10")
        assert res2["value_repr"] == "30"
        assert res2["result"]["name"] == "out2"

    def test_assignment_registers_name_but_echoes_nothing(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "x = 5")
        assert res["ok"] is True
        assert res["value_repr"] == ""      # bare assignment → no echo
        assert res["result"] is None        # nothing registered as out
        # …but x is a chip (source=="assign") and is usable.
        cv = _wait_vars(session, msgs, lambda c: any(
            v["name"] == "x" and v["source"] == "assign" for v in c["vars"]))
        assert any(v["name"] == "x" for v in cv["vars"])
        res2 = _exec(session, msgs, "x + 1")
        assert res2["value_repr"] == "6"

    def test_multi_statement_with_trailing_expression(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "a = 3\nb = 4\na * b")
        assert res["ok"] is True
        assert res["value_repr"] == "12"
        assert res["result"]["name"] == "out1"
        # both a and b were registered as assigned chips.
        cv = _wait_vars(session, msgs, lambda c: {"a", "b"}.issubset(
            {v["name"] for v in c["vars"]}))
        names = {v["name"] for v in cv["vars"]}
        assert {"a", "b"}.issubset(names)

    def test_none_valued_expression_registers_no_out(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "print('hi')")   # print returns None
        assert res["ok"] is True
        assert res["result"] is None
        assert res["value_repr"] == ""


# ── stdout capture ────────────────────────────────────────────────────────────


class TestStdout:
    def test_stdout_captured(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "print('hello world')")
        assert res["ok"] is True
        assert "hello world" in res["stdout"]

    def test_stdout_with_trailing_expression(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "print('side effect')\n42")
        assert "side effect" in res["stdout"]
        assert res["value_repr"] == "42"


# ── errors ─────────────────────────────────────────────────────────────────────


class TestErrors:
    def test_runtime_error_reports_traceback_and_keeps_namespace(self, window):
        session, msgs = window["window"], window["messages"]
        _exec(session, msgs, "keep = 99")
        res = _exec(session, msgs, "1 / 0")
        assert res["ok"] is False
        assert res["error"].startswith("ZeroDivisionError")
        assert res["traceback"] and "ZeroDivisionError" in res["traceback"]
        assert res["result"] is None
        # Namespace survived the error — keep is still bound.
        res2 = _exec(session, msgs, "keep")
        assert res2["ok"] is True and res2["value_repr"] == "99"

    def test_syntax_error_reports_ok_false(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "x = = 5")
        assert res["ok"] is False
        assert "SyntaxError" in res["error"]

    def test_name_error_surfaces(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "undefined_name + 1")
        assert res["ok"] is False
        assert res["error"].startswith("NameError")


# ── persistence + builtins ─────────────────────────────────────────────────────


class TestNamespace:
    def test_persistence_across_execs(self, window):
        session, msgs = window["window"], window["messages"]
        _exec(session, msgs, "import math")
        _exec(session, msgs, "r = math.sqrt(16)")
        res = _exec(session, msgs, "r")
        assert res["value_repr"] == "4.0"

    def test_numpy_available(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "int(np.arange(4).sum())")
        assert res["ok"] is True
        assert res["value_repr"] == "6"

    def test_numpy_array_expression_kind_ndarray(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "np.random.rand(8, 8)")
        assert res["ok"] is True
        assert res["result"]["kind"] == "ndarray"
        assert res["result"]["shape"] == [8, 8]
        assert res["result"]["lazy"] is False

    def test_hs_and_da_available(self, window):
        session, msgs = window["window"], window["messages"]
        res = _exec(session, msgs, "da.zeros((4, 4)).shape")
        assert res["ok"] is True
        res2 = _exec(session, msgs, "hs.signals.Signal1D(np.arange(5))")
        assert res2["ok"] is True
        assert res2["result"]["kind"] == "signal"


# ── signal auto-binding ─────────────────────────────────────────────────────────


class TestSignalBindings:
    def test_signal_exposed_as_name_and_positional_alias(self, stem_4d_dataset):
        session, msgs = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        con = session.console      # create the engine → it exposes current trees
        cv = _wait_vars(session, msgs, lambda c: any(
            v["name"] == "s1" and v["source"] == "signal" for v in c["vars"]))
        names = {v["name"] for v in cv["vars"]}
        assert "s1" in names, f"positional alias missing: {names}"
        # s1 resolves in the namespace to the tree root and is usable.
        res = _exec(session, msgs, "s1.data.shape")
        assert res["ok"] is True

    def test_signal_binding_has_window_ids(self, stem_4d_dataset):
        session, msgs = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _ = session.console
        cv = _wait_vars(session, msgs, lambda c: any(
            v["name"] == "s1" for v in c["vars"]))
        s1 = next(v for v in cv["vars"] if v["name"] == "s1")
        assert s1["source"] == "signal"
        assert s1["window_ids"] is not None and len(s1["window_ids"]) >= 1
        # window_ids should match the tree's real plot windows.
        tree = session.signal_trees[0]
        expected = session._tree_window_ids(tree)
        assert set(s1["window_ids"]) == set(expected)

    def test_titled_signal_gets_sanitized_name(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        s = hs.signals.Signal2D(np.zeros((8, 8), dtype=np.float32))
        s.metadata.General.title = "Vector count map"
        session._add_signal(s, source_path=None)
        cv = _wait_vars(session, msgs, lambda c: any(
            v["name"] == "Vector_count_map" for v in c["vars"]))
        names = {v["name"] for v in cv["vars"]}
        assert "Vector_count_map" in names
        assert "s1" in names   # positional alias too

    def test_binding_removed_on_tree_close(self, stem_4d_dataset):
        session, msgs = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _ = session.console
        _wait_vars(session, msgs, lambda c: any(
            v["name"] == "s1" for v in c["vars"]))
        # Close the tree via the navigator window's X (closes the whole tree).
        tree = session.signal_trees[0]
        nav_plot = next(p for p in session._plots
                        if getattr(p, "signal_tree", None) is tree
                        and getattr(p, "is_navigator", False))
        session._close_window(nav_plot.window_id)
        # After close, s1 should be gone from console_vars.
        cv = _wait_vars(session, msgs, lambda c: not any(
            v["name"] == "s1" for v in c["vars"]))
        assert not any(v["name"] == "s1" for v in cv["vars"])
        # …and the namespace no longer resolves s1 either (stale binding cleaned).
        res = _exec(session, msgs, "s1")
        assert res["ok"] is False and res["error"].startswith("NameError")

    def test_positional_aliases_track_load_order(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        a = hs.signals.Signal2D(np.zeros((4, 4), dtype=np.float32))
        a.metadata.General.title = "alpha"
        b = hs.signals.Signal2D(np.zeros((6, 6), dtype=np.float32))
        b.metadata.General.title = "beta"
        session._add_signal(a, source_path=None)
        session._add_signal(b, source_path=None)
        cv = _wait_vars(session, msgs, lambda c: {"s1", "s2"}.issubset(
            {v["name"] for v in c["vars"]}))
        s1 = next(v for v in cv["vars"] if v["name"] == "s1")
        s2 = next(v for v in cv["vars"] if v["name"] == "s2")
        # s1 → alpha (4x4), s2 → beta (6x6).
        assert s1["shape"] == [4, 4]
        assert s2["shape"] == [6, 6]
        assert {"alpha", "beta"}.issubset({v["name"] for v in cv["vars"]})


# ── lazy end-to-end: the engine NEVER computes ────────────────────────────────


class TestLazyNeverComputes:
    def test_lazy_signal_comparison_stays_lazy(self, window):
        """`s1 > 100` on a LAZY signal builds a graph, does NOT compute. A guard
        on da.Array.compute proves the engine materialises nothing (mirrors the
        find-vectors memory-safety guard pattern)."""
        session, msgs = window["window"], window["messages"]
        _ = session.console
        sig = _lazy_4d()
        full_shape = sig.data.shape
        session._add_signal(sig, source_path=None)
        _wait_vars(session, msgs, lambda c: any(v["name"] == "s1" for v in c["vars"]))

        computed = {"full": False}
        _orig = da.Array.compute

        def _spy(self, *a, **k):
            if self.shape == full_shape:
                computed["full"] = True
                raise AssertionError(
                    f"compute() on full-dataset shape {self.shape} — the console "
                    f"must never materialise user data.")
            return _orig(self, *a, **k)

        with patch.object(da.Array, "compute", _spy):
            res = _exec(session, msgs, "mask = s1 > 100")
            assert res["ok"] is True, res.get("error")
            res2 = _exec(session, msgs, "mask")
            assert res2["ok"] is True
            # The echoed mask is a lazy signal — shape/dtype only, still lazy.
            assert res2["result"]["kind"] == "signal"
            assert res2["result"]["lazy"] is True
            assert res2["result"]["shape"] == list(full_shape)
        assert computed["full"] is False

    def test_lazy_dask_array_expression_reports_lazy(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        _orig = da.Array.compute
        with patch.object(da.Array, "compute",
                          lambda self, *a, **k: (_ for _ in ()).throw(
                              AssertionError("computed a dask array"))):
            res = _exec(session, msgs, "da.ones((256, 256)) + da.ones((256, 256))")
            assert res["ok"] is True
            assert res["result"]["kind"] == "dask"
            assert res["result"]["lazy"] is True
            assert res["result"]["shape"] == [256, 256]


# ── materialisation ─────────────────────────────────────────────────────────────


class TestMaterialisation:
    def test_materialise_ndarray_creates_tree(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        _exec(session, msgs, "img = np.random.rand(16, 16).astype('float32')")
        before = len(session.signal_trees)
        session.console.create_window("img")
        deadline = time.time() + 6.0
        while time.time() < deadline and len(session.signal_trees) <= before:
            time.sleep(0.02)
        assert len(session.signal_trees) == before + 1
        new_tree = session.signal_trees[-1]
        root = new_tree.root
        assert tuple(root.data.shape) == (16, 16)
        assert isinstance(root, hs.signals.Signal2D)
        # Named after the variable, with console provenance recorded.
        assert root.metadata.get_item("General.title") == "img"
        assert "console" in str(root.metadata.get_item("General.notes", ""))

    def test_materialise_1d_ndarray_is_signal1d(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        _exec(session, msgs, "line = np.arange(64).astype('float32')")
        before = len(session.signal_trees)
        session.console.create_window("line")
        deadline = time.time() + 6.0
        while time.time() < deadline and len(session.signal_trees) <= before:
            time.sleep(0.02)
        assert len(session.signal_trees) == before + 1
        assert isinstance(session.signal_trees[-1].root, hs.signals.Signal1D)

    def test_materialise_lazy_result_stays_lazy(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        _exec(session, msgs, "lz = da.zeros((5, 32, 32), chunks=(1, 32, 32))")
        before = len(session.signal_trees)
        _orig = da.Array.compute
        computed = {"any": False}

        def _spy(self, *a, **k):
            computed["any"] = True
            return _orig(self, *a, **k)

        with patch.object(da.Array, "compute", _spy):
            session.console.create_window("lz")
            deadline = time.time() + 6.0
            while time.time() < deadline and len(session.signal_trees) <= before:
                time.sleep(0.02)
        assert len(session.signal_trees) == before + 1
        new_root = session.signal_trees[-1].root
        assert bool(getattr(new_root, "_lazy", False)) is True
        # A 3-D array → Signal2D with one navigation axis.
        assert isinstance(new_root, hs.signals.Signal2D)
        assert new_root.axes_manager.navigation_dimension == 1
        assert new_root.axes_manager.signal_dimension == 2

    def test_show_helper_opens_window(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        before = len(session.signal_trees)
        _exec(session, msgs, "show(np.zeros((12, 12), dtype='float32'))")
        deadline = time.time() + 6.0
        while time.time() < deadline and len(session.signal_trees) <= before:
            time.sleep(0.02)
        assert len(session.signal_trees) == before + 1

    def test_materialise_scalar_refuses_politely(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        _exec(session, msgs, "n = 5")
        before = len(session.signal_trees)
        msgs.clear()
        session.console.create_window("n")
        # A status message explaining the refusal; no new tree.
        st = _wait_for(msgs, lambda m: m.get("type") == "status"
                       and "can't open" in m.get("text", ""), timeout=3.0)
        assert st is not None
        assert len(session.signal_trees) == before


# ── completion ─────────────────────────────────────────────────────────────────


class TestCompletion:
    def test_prefix_completion(self, window):
        session, msgs = window["window"], window["messages"]
        _exec(session, msgs, "apple = 1")
        _exec(session, msgs, "apricot = 2")
        session.console.submit_complete("ap", 7)
        comp = _wait_for(msgs, lambda m: m.get("type") == "console_completions"
                         and m.get("complete_id") == 7)
        assert comp is not None
        assert "apple" in comp["matches"]
        assert "apricot" in comp["matches"]

    def test_builtin_names_complete(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        session.console.submit_complete("n", 8)
        comp = _wait_for(msgs, lambda m: m.get("type") == "console_completions"
                         and m.get("complete_id") == 8)
        assert "np" in comp["matches"]

    def test_attribute_completion_one_level(self, window):
        session, msgs = window["window"], window["messages"]
        _exec(session, msgs, "arr = np.arange(9)")
        session.console.submit_complete("arr.su", 9)
        comp = _wait_for(msgs, lambda m: m.get("type") == "console_completions"
                         and m.get("complete_id") == 9)
        assert comp is not None
        # ndarray has .sum — the returned match is the full replacement text.
        assert "arr.sum" in comp["matches"]

    def test_attribute_completion_is_side_effect_free(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        # A dotted chain that would need to CALL something is not evaluated.
        session.console.submit_complete("np.arange(3).su", 10)
        comp = _wait_for(msgs, lambda m: m.get("type") == "console_completions"
                         and m.get("complete_id") == 10, timeout=3.0)
        assert comp is not None
        assert comp["matches"] == []   # no eval of the call → no matches


# ── console_vars schema ─────────────────────────────────────────────────────────


class TestVarsSchema:
    _KEYS = {"name", "kind", "shape", "dtype", "lazy", "source", "window_ids"}

    def test_vars_schema_exact(self, stem_4d_dataset):
        session, msgs = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _ = session.console
        _exec(session, msgs, "y = np.zeros((3, 3))")
        cv = _wait_vars(session, msgs, lambda c: any(
            v["name"] == "y" for v in c["vars"]) and any(
            v["name"] == "s1" for v in c["vars"]))
        assert cv["type"] == "console_vars"
        for v in cv["vars"]:
            assert set(v.keys()) == self._KEYS, f"unexpected keys: {v.keys()}"
            assert v["kind"] in ("signal", "ndarray", "dask", "scalar", "other")
            assert v["source"] in ("signal", "assign", "out")
            if v["source"] == "signal":
                assert v["window_ids"] is not None
            else:
                assert v["window_ids"] is None

    def test_out_chip_appears_in_vars(self, window):
        session, msgs = window["window"], window["messages"]
        _exec(session, msgs, "np.ones((2, 2))")
        cv = _wait_vars(session, msgs, lambda c: any(
            v["name"] == "out1" and v["source"] == "out" for v in c["vars"]))
        out1 = next(v for v in cv["vars"] if v["name"] == "out1")
        assert out1["kind"] == "ndarray"
        assert out1["shape"] == [2, 2]
        assert out1["window_ids"] is None


# ── dispatch wiring (the flat command envelope) ───────────────────────────────


class TestDispatchWiring:
    def test_console_exec_via_app_command_envelope(self, window):
        """The renderer sends {"command": "console_exec", …}; app._dispatch_console
        routes it to the engine."""
        from spyde.backend import app
        session, msgs = window["window"], window["messages"]
        app._dispatch_console(session, {"command": "console_exec",
                                        "code": "2 ** 8", "exec_id": 55})
        res = _wait_for(msgs, lambda m: m.get("type") == "console_result"
                        and m.get("exec_id") == 55)
        assert res is not None and res["value_repr"] == "256"

    def test_console_action_dispatch(self, window):
        """dispatch_action also accepts the console_* actions (belt-and-suspenders
        with the flat command envelope)."""
        session, msgs = window["window"], window["messages"]
        session.dispatch_action({"action": "console_exec",
                                 "payload": {"code": "7 * 6", "exec_id": 66}})
        res = _wait_for(msgs, lambda m: m.get("type") == "console_result"
                        and m.get("exec_id") == 66)
        assert res is not None and res["value_repr"] == "42"
