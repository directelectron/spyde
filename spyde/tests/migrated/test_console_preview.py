"""
Tests for the SpyDE math-console live PREVIEW evaluator
(spyde.backend.console_preview + ConsoleSession.submit_preview / _do_preview).

Qt-free, mirroring test_console.py: a real Session via the conftest fixtures +
captured_messages, driven through the SAME dispatch the renderer uses. The
console runs previews on its own daemon thread, so the helpers poll the
captured-message list for the matching console_preview_result rather than
sleeping.

Coverage:
  * the auto-tier AST classifier (parse_expr / is_auto_safe) — a pure whitelist;
    a call / comprehension / literal-container / lambda / walrus is NOT auto-safe
  * a preview of `s1 > 100` on a LAZY 4-D signal slices ONE frame and NEVER
    computes the full dataset (a da.Array.compute guard proves it)
  * the cost guard refuses an expensive nav-axis sum (and never computes) and
    nudges toward Ctrl+Enter on the auto path
  * render kinds: scalar / sparkline (≤512 pts, NaN→null) / image (base64
    thumbnail ≤192px, values match a locally-reproduced pipeline)
  * nav-position resolution: the thumbnail reflects the selector's current
    cursor, and falls back to frame (0,0) with no selector
  * newest-wins coalescing (a superseded preview never emits; an exec cancels a
    queued preview)
  * previews are side-effect-free (no out<N>/assign chips, no console_result)
"""
from __future__ import annotations

import base64
import time

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest
from unittest.mock import patch

from spyde.backend import console_preview as cp

# Import the heavy stack (pyxem/hyperspy) ONCE, synchronously, up front — the
# console engine's _init_namespace imports pyxem on its own thread, which can
# race a test's `set_signal_type("electron_diffraction")` (a partially-
# initialized-module circular import). Warming it here removes the race (the
# established testharness pattern: ensure_heavy_imports BEFORE set_signal_type).
from spyde.backend.heavy_imports import ensure_heavy_imports
ensure_heavy_imports()


# ── helpers (mirror test_console.py) ─────────────────────────────────────────


def _wait_for(msgs, pred, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for m in msgs:
            if pred(m):
                return m
        time.sleep(0.01)
    return None


def _preview(session, msgs, code, auto=True, timeout=30.0):
    """Submit a live preview and return its console_preview_result message.
    preview_ids are monotonic per call so results don't cross."""
    pid = getattr(_preview, "_next", 1)
    _preview._next = pid + 1
    session.console.submit_preview(code, pid, auto)
    return _wait_for(
        msgs, lambda m: m.get("type") == "console_preview_result"
        and m.get("preview_id") == pid, timeout=timeout,
    )


def _exec(session, msgs, code, timeout=30.0):
    """Run a cell and return its console_result message (from test_console.py)."""
    exec_id = getattr(_exec, "_next", 1000)
    _exec._next = exec_id + 1
    session.console.submit_exec(code, exec_id)
    return _wait_for(
        msgs, lambda m: m.get("type") == "console_result"
        and m.get("exec_id") == exec_id, timeout=timeout,
    )


def _wait_vars(session, msgs, pred, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        vs = [m for m in msgs if m.get("type") == "console_vars"]
        if vs and pred(vs[-1]):
            return vs[-1]
        time.sleep(0.02)
    vs = [m for m in msgs if m.get("type") == "console_vars"]
    return vs[-1] if vs else None


def _lazy_4d(nav=(3, 4), sig=(8, 8)):
    """A lazy 4D-STEM signal, 1 nav position per chunk (from test_console.py)."""
    ny, nx = nav
    ky, kx = sig
    arr = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    arr[:, :, ky // 2, kx // 2] = 200.0
    d = da.from_array(arr, chunks=(1, 1, ky, kx))
    s = hs.signals.Signal2D(d).as_lazy()
    s.set_signal_type("electron_diffraction")
    return s


def _stamped_4d(nav=(3, 4), sig=(8, 8)):
    """A lazy 4D-STEM signal whose every nav position has a DISTINCT spatial
    pattern: frame[iy, ix] is a diagonal ramp scaled by the position index, so
    each position's thumbnail differs even AFTER robust normalization (a constant
    per-frame offset would wash out to an identical thumbnail — the ramp gives a
    real spatial structure that survives the 2–99.5% clip)."""
    ny, nx = nav
    ky, kx = sig
    yy, xx = np.mgrid[0:ky, 0:kx].astype(np.float32)
    arr = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            pos = iy * nx + ix + 1
            arr[iy, ix] = (yy * pos) + (xx * (pos + 1))
    d = da.from_array(arr, chunks=(1, 1, ky, kx))
    s = hs.signals.Signal2D(d).as_lazy()
    s.set_signal_type("electron_diffraction")
    return s, arr


def _pipeline_thumb(frame):
    """Reproduce console_preview's 2-D render pipeline on a raw numpy frame so a
    test can assert the emitted thumbnail matches byte-for-byte."""
    return cp._render_2d(np.squeeze(np.asarray(frame)))


def _decode_image(payload):
    """Decode an ``image`` payload's base64 bytes into a (h, w) uint8 array."""
    raw = base64.b64decode(payload["data_b64"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(payload["h"], payload["w"])


# ── 1. the auto-tier AST classifier (pure, no session) ───────────────────────


class TestAutoTierClassifier:
    @pytest.mark.parametrize("code,expected", [
        ("s1 > 100", True),
        ("s1[0]", True),
        ("s1[0:2, ::2]", True),
        ("s1.data", True),
        ("s1.sum", True),
        ("(a, b)", True),
        ("-s1 + 2 * b", True),
        ("a and not b", True),
        ("s1.sum()", False),
        ("np.log(s1)", False),
        ("s1.sum(axis=(0,1))", False),
        ("x = 1", False),          # a statement → parse_expr None → not auto-safe
        ("a if b else c", False),
        ("[s1]", False),
        ("{'a': 1}", False),
        ("f'{s1}'", False),
        ("lambda: 1", False),
        ("(x := 1)", False),
        ("[i for i in s1]", False),
    ])
    def test_auto_safe(self, code, expected):
        tree = cp.parse_expr(code)
        if tree is None:
            # An unparseable / statement input is never auto-safe.
            assert expected is False
            return
        assert cp.is_auto_safe(tree) is expected


# ── 2. a preview NEVER computes the full dataset ─────────────────────────────


class TestPreviewNeverComputesFull:
    def test_lazy_comparison_previews_one_frame(self, window):
        """`s1 > 100` on a LAZY 4-D signal → an image thumbnail, and da.Array.compute
        is never called on the full-dataset shape (mirrors TestLazyNeverComputes)."""
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
                    f"compute() on full-dataset shape {self.shape} — a preview "
                    f"must never materialise user data.")
            return _orig(self, *a, **k)

        with patch.object(da.Array, "compute", _spy):
            res = _preview(session, msgs, "s1 > 100")
        assert res is not None
        assert res["kind"] == "image", res
        assert computed["full"] is False


# ── 3. the cost guard refuses (and never computes) an expensive view ─────────


class TestCostGuard:
    def test_nav_sum_is_too_expensive(self, window):
        """A nav-axis sum touches every nav chunk (12 > MAX_SOURCE_CHUNKS) → the
        cost guard refuses and compute is never called."""
        session, msgs = window["window"], window["messages"]
        _ = session.console
        # 1 nav position per chunk → 3*4 = 12 source chunks for a full-nav sum.
        sig = _lazy_4d(nav=(3, 4))
        session._add_signal(sig, source_path=None)
        _wait_vars(session, msgs, lambda c: any(v["name"] == "s1" for v in c["vars"]))

        with patch.object(da.Array, "compute",
                          lambda self, *a, **k: (_ for _ in ()).throw(
                              AssertionError("compute ran on a too-expensive preview"))):
            # Manual (Ctrl+Enter) path — a call needs auto=False to run at all.
            res = _preview(session, msgs, "s1.sum(axis=(0, 1))", auto=False)
        assert res is not None
        assert res["kind"] == "unavailable"
        assert "expensive" in res["reason"], res

        # The auto path with the same call is gated even earlier (it's a Call) —
        # the reason points the user at Ctrl+Enter.
        res_auto = _preview(session, msgs, "s1.sum(axis=(0, 1))", auto=True)
        assert res_auto["kind"] == "unavailable"
        assert "Ctrl+Enter" in res_auto["reason"], res_auto


# ── 4. render kinds ──────────────────────────────────────────────────────────


class TestRenderKinds:
    def test_scalar(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        res = _preview(session, msgs, "1 + 2")
        assert res["kind"] == "scalar"
        assert res["text"] == "3"

    def test_sparkline_strided_and_capped(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        _exec(session, msgs, "import numpy as np")
        _exec(session, msgs, "v = np.arange(100000.0)")
        res = _preview(session, msgs, "v * 2")
        assert res["kind"] == "sparkline"
        assert len(res["points"]) <= cp.SPARK_MAX
        assert all(p is None or isinstance(p, float) for p in res["points"])

    def test_sparkline_nan_becomes_null(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        _exec(session, msgs, "import numpy as np")
        _exec(session, msgs, "w = np.array([1.0, np.nan, 3.0, np.inf])")
        res = _preview(session, msgs, "w")
        assert res["kind"] == "sparkline"
        # NaN / inf → null (json NaN literal would break the renderer's parse).
        assert res["points"] == [1.0, None, 3.0, None]

    def test_image_thumbnail_matches_pipeline(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        sig = _lazy_4d()
        session._add_signal(sig, source_path=None)
        _wait_vars(session, msgs, lambda c: any(v["name"] == "s1" for v in c["vars"]))
        res = _preview(session, msgs, "s1")
        assert res["kind"] == "image"
        assert res["w"] <= cp.THUMB_MAX and res["h"] <= cp.THUMB_MAX
        img = _decode_image(res)
        assert img.size == res["w"] * res["h"]
        # Reproduce the pipeline on the source frame at the default position (0,0).
        expected = _pipeline_thumb(sig.data[0, 0].compute())
        exp_img = _decode_image(expected)
        assert np.array_equal(img, exp_img)


# ── 5. nav-position resolution ───────────────────────────────────────────────


class TestNavPositionResolution:
    def _signal_selector(self, session, tree):
        """The selector driving *tree*'s signal DP (has current_indices)."""
        for p in session._plots:
            if getattr(p, "signal_tree", None) is not tree:
                continue
            sel = getattr(p, "parent_selector", None)
            if sel is not None and getattr(sel, "current_indices", None) is not None:
                return sel
        # Fall back: any selector on the tree (may not have indices yet).
        for p in session._plots:
            if getattr(p, "signal_tree", None) is tree:
                sel = getattr(p, "parent_selector", None)
                if sel is not None:
                    return sel
        return None

    @staticmethod
    def _set_indices(sel, value):
        """Set current_indices on *sel*, going through the inner active selector
        for a composite (IntegratingSSelector2D exposes current_indices as a
        read-only property that delegates to ``.selector``)."""
        inner = getattr(sel, "selector", None)
        target = inner if inner is not None else sel
        target.current_indices = value

    def _move_all(self, session, tree, value):
        """Set current_indices on EVERY selector bound to *tree* (selector
        attachment is async — the resolver takes the first match, which may
        not be the one a single-selector set touched on a slow runner)."""
        for p in session._plots:
            if getattr(p, "signal_tree", None) is tree:
                sel = getattr(p, "parent_selector", None)
                if sel is not None:
                    self._set_indices(sel, value)

    def _settle_cursor(self, session, tree, value, expect, timeout=30.0):
        """Keep applying *value* to all selectors until the console's own
        resolver reports *expect* (late selector attaches re-introduce state)."""
        from spyde.backend.console_preview import _nav_indices_for
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._move_all(session, tree, value)
            idx, _shape = _nav_indices_for(session.console, ["s1"])
            got = None if idx is None else tuple(int(v) for v in np.ravel(idx))
            if got == expect:
                return True
            time.sleep(0.05)
        return False

    def test_thumbnail_reflects_cursor(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        sig, arr = _stamped_4d()
        session._add_signal(sig, source_path=None)
        _wait_vars(session, msgs, lambda c: any(v["name"] == "s1" for v in c["vars"]))
        tree = session.signal_trees[0]

        assert self._settle_cursor(session, tree, np.array([1, 2]), (1, 2)),             "cursor never settled at (1, 2)"

        res = _preview(session, msgs, "s1")
        assert res["kind"] == "image"
        img = _decode_image(res)
        # Must equal the pipeline on frame (1, 2) and DIFFER from frame (0, 0).
        at_12 = _decode_image(_pipeline_thumb(arr[1, 2]))
        at_00 = _decode_image(_pipeline_thumb(arr[0, 0]))
        assert np.array_equal(img, at_12)
        assert not np.array_equal(at_12, at_00)

    def test_no_selector_falls_back_to_frame_zero(self, window):
        """With no resolvable cursor the preview must not raise — it renders
        frame (0,0)."""
        session, msgs = window["window"], window["messages"]
        _ = session.console
        sig, arr = _stamped_4d()
        session._add_signal(sig, source_path=None)
        _wait_vars(session, msgs, lambda c: any(v["name"] == "s1" for v in c["vars"]))
        tree = session.signal_trees[0]
        # Clear any resolved indices so _nav_indices_for returns (None, None) —
        # keep clearing until the resolver agrees (a selector can attach late,
        # already carrying an initial cursor).
        assert self._settle_cursor(session, tree, None, None),             "selector indices never cleared"

        res = _preview(session, msgs, "s1")
        assert res["kind"] == "image"
        img = _decode_image(res)
        assert np.array_equal(img, _decode_image(_pipeline_thumb(arr[0, 0])))


# ── 6. newest-wins coalescing (exec wins) ────────────────────────────────────


class TestCoalescing:
    def test_superseded_preview_never_emits(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        # Block the console thread with a slow exec so both previews queue behind
        # it; the second bumps _latest_preview_id → the first is dropped as stale.
        _exec_id = 5000
        session.console.submit_exec("import time as _t; _t.sleep(0.4)", _exec_id)
        session.console.submit_preview("1 + 1", 1, True)
        session.console.submit_preview("2 + 2", 2, True)

        res2 = _wait_for(msgs, lambda m: m.get("type") == "console_preview_result"
                         and m.get("preview_id") == 2, timeout=30.0)
        assert res2 is not None and res2["text"] == "4"
        # id=1 was superseded before it ran → it never emits a result.
        res1 = _wait_for(msgs, lambda m: m.get("type") == "console_preview_result"
                         and m.get("preview_id") == 1, timeout=0.5)
        assert res1 is None

    def test_exec_cancels_queued_preview(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        # Block the thread, queue a preview, then queue an exec: submit_exec resets
        # _latest_preview_id to -1, so the queued preview (id=3) no-ops.
        session.console.submit_exec("import time as _t; _t.sleep(0.4)", 5100)
        session.console.submit_preview("3 + 3", 3, True)
        session.console.submit_exec("9 + 9", 5101)

        res_exec = _wait_for(msgs, lambda m: m.get("type") == "console_result"
                             and m.get("exec_id") == 5101, timeout=30.0)
        assert res_exec is not None and res_exec["value_repr"] == "18"
        # The preview (id=3) lost to the exec → never emits.
        res3 = _wait_for(msgs, lambda m: m.get("type") == "console_preview_result"
                         and m.get("preview_id") == 3, timeout=0.5)
        assert res3 is None


# ── 7. nav-change refresh (the preview tracks the navigator) ─────────────────


def _count_results(msgs, pid):
    return sum(1 for m in msgs if m.get("type") == "console_preview_result"
               and m.get("preview_id") == pid)


class TestNavRefresh:
    """The last AUTO preview re-runs when a navigator commits a NEW position
    (base_selector.NAV_CHANGE_HOOKS → ConsoleSession.notify_nav_changed), and
    STOPS re-running after the frontend's empty-code stop / an exec."""

    def test_hook_registered_and_removed(self, window):
        from spyde.drawing.selectors import base_selector
        session = window["window"]
        console = session.console
        assert console.notify_nav_changed in base_selector.NAV_CHANGE_HOOKS
        console.shutdown()   # idempotent — the fixture teardown re-calls it
        assert console.notify_nav_changed not in base_selector.NAV_CHANGE_HOOKS

    def test_nav_move_reemits_at_new_cursor(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        sig, arr = _stamped_4d()
        session._add_signal(sig, source_path=None)
        _wait_vars(session, msgs, lambda c: any(v["name"] == "s1" for v in c["vars"]))
        tree = session.signal_trees[0]
        nav = TestNavPositionResolution()
        assert nav._settle_cursor(session, tree, np.array([0, 0]), (0, 0)),             "cursor never settled at (0, 0)"

        res = _preview(session, msgs, "s1", auto=True)
        assert res is not None and res["kind"] == "image"
        pid = res["preview_id"]
        n0 = _count_results(msgs, pid)

        # Navigator moves → the console re-runs the SAME preview (same id) at
        # the new cursor; the frontend's newest-wins intake accepts it in place.
        assert nav._settle_cursor(session, tree, np.array([1, 2]), (1, 2)),             "cursor never settled at (1, 2)"
        # Re-emit + content poll: a selector can attach LATE with a stale
        # initial cursor and win the resolver scan between the settle and the
        # console's re-run — keep re-pinning the cursor and re-notifying until
        # the LATEST thumbnail is the frame at the new cursor. Stale
        # intermediates are acceptable by design (newest-wins intake).
        expected = _decode_image(_pipeline_thumb(arr[1, 2]))
        deadline = time.time() + 30.0
        img = None
        while time.time() < deadline:
            nav._move_all(session, tree, np.array([1, 2]))
            session.console.notify_nav_changed()
            _wait_for(
                msgs, lambda m: m.get("type") == "console_preview_result"
                and m.get("preview_id") == pid
                and _count_results(msgs, pid) > n0, timeout=5.0,
            )
            results = [m for m in msgs if m.get("type") == "console_preview_result"
                       and m.get("preview_id") == pid]
            if len(results) > n0:
                img = _decode_image(results[-1])
                if np.array_equal(img, expected):
                    break
        assert _count_results(msgs, pid) > n0, "nav move did not re-emit the preview"
        assert img is not None and np.array_equal(img, expected)
        assert not np.array_equal(img, _decode_image(_pipeline_thumb(arr[0, 0])))

    def test_stop_clears_nav_refresh(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        res = _preview(session, msgs, "1 + 2", auto=True)
        assert res is not None
        pid = res["preview_id"]
        # The frontend's STOP (eye off / cell emptied): empty code, no reply.
        session.console.submit_preview("", pid + 999, True)
        n0 = _count_results(msgs, pid)
        session.console.notify_nav_changed()
        time.sleep(0.6)
        assert _count_results(msgs, pid) == n0, "nav move re-ran a STOPPED preview"
        # And the stop itself emitted nothing.
        assert not any(m.get("type") == "console_preview_result"
                       and m.get("preview_id") == pid + 999 for m in msgs)

    def test_exec_clears_nav_refresh(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        res = _preview(session, msgs, "2 + 2", auto=True)
        assert res is not None
        pid = res["preview_id"]
        assert _exec(session, msgs, "1 + 1") is not None
        n0 = _count_results(msgs, pid)
        session.console.notify_nav_changed()
        time.sleep(0.6)
        assert _count_results(msgs, pid) == n0, "nav move resurrected a pre-exec preview"

    def test_explicit_preview_not_nav_tracked(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        # A Ctrl+Enter one-shot (auto=False) must NOT be re-run on nav moves —
        # it may contain arbitrary calls / be expensive per evaluation.
        res = _preview(session, msgs, "3 + 3", auto=False)
        assert res is not None
        pid = res["preview_id"]
        n0 = _count_results(msgs, pid)
        session.console.notify_nav_changed()
        time.sleep(0.6)
        assert _count_results(msgs, pid) == n0

    def test_run_update_fires_hook_only_on_change(self, window, monkeypatch):
        """The base_selector seam: _run_update fires NAV_CHANGE_HOOKS exactly
        when the committed position genuinely changed — not for a force re-fire
        at the same position (settle/contrast repaints)."""
        from spyde.drawing.selectors import base_selector
        session, msgs = window["window"], window["messages"]
        sig = _lazy_4d()
        session._add_signal(sig, source_path=None)
        tree = session.signal_trees[0]
        sel = TestNavPositionResolution()._signal_selector(session, tree)
        assert sel is not None

        # _run_update runs on the INNER widget selector (the composite exposes
        # current_indices as a read-only delegating property) — drive that one.
        target = getattr(sel, "selector", None) or sel

        fired = []
        base_selector.NAV_CHANGE_HOOKS.append(lambda: fired.append(1))
        try:
            pos = {"v": np.array([0, 1])}
            monkeypatch.setattr(target, "get_selected_indices", lambda: pos["v"])
            target.current_indices = np.array([0, 0])
            target._run_update()                 # (0,0) → (0,1): changed
            assert len(fired) == 1
            target._run_update(force=True)       # same position, forced repaint
            assert len(fired) == 1, "force re-fire at an unchanged position must not notify"
            pos["v"] = np.array([2, 2])
            target._run_update()                 # moved again
            assert len(fired) == 2
        finally:
            base_selector.NAV_CHANGE_HOOKS.pop()


# ── 8. previews are side-effect-free ─────────────────────────────────────────


class TestPreviewSideEffectFree:
    def test_no_chips_no_console_result(self, window):
        session, msgs = window["window"], window["messages"]
        _ = session.console
        # A preview whose exec-form WOULD register an out<N> chip (a bare value).
        res = _preview(session, msgs, "np.arange(9).reshape(3, 3)")
        assert res is not None
        # No console_result at all (previews never emit one).
        assert not any(m.get("type") == "console_result" for m in msgs)
        # No out<N> / assign chip appeared in console_vars.
        vs = [m for m in msgs if m.get("type") == "console_vars"]
        for cv in vs:
            names = {v["name"] for v in cv["vars"]}
            assert not any(n.startswith("out") for n in names), names
