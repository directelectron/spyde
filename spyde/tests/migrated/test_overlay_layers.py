"""
test_overlay_layers.py — Report Builder Phase 2, MDI live image layering.

Exercises spyde.actions.overlay against a real Qt-free ``Session`` built on a 4-D
STEM tree (navigator + signal windows share one tree). A second same-shape signal
window is added so a layer's SOURCE differs from its TARGET.

Covered: add / same-shape validation / mismatch refusal / self refusal,
``layers_state`` emissions, overlay_set reflected in the anyplotlib layer state,
nav-move → the layer refreshes with the SOURCE's frame at the new indices (driven
via the selector's ``_run_update``, like test_navigator_race), an async-tier move
skipping the layer refresh without error, teardown on window close, and tile-mode
refusal.
"""
from __future__ import annotations

import time

import numpy as np

from spyde.actions import overlay as ov


# ── setup helpers ───────────────────────────────────────────────────────────────


def _prime(session):
    for p in session._plots:
        if isinstance(getattr(p, "current_data", None), np.ndarray):
            continue
        try:
            sig = p.plot_state.current_signal
            frame = np.asarray(sig.data)
            if frame.ndim > 2:
                frame = frame.reshape(-1, *frame.shape[-2:])[0]
            p.current_data = np.ascontiguousarray(frame.astype(np.float32))
            p._last_levels = (float(np.nanmin(p.current_data)),
                              float(np.nanmax(p.current_data)))
        except Exception:
            pass


def _two_signal_windows(session):
    """Add a SECOND same-shape signal window to the 4-D tree and return
    (target_plot, source_plot, navigator_plot, multiplot_manager, nav_plot_window).
    Both signal plots read the same signal at the same nav indices (same tree)."""
    nav = [p for p in session._plots if p.is_navigator][0]
    mm = nav.multiplot_manager
    navpw = nav.plot_window
    mm.add_navigation_selector_and_signal_plot(navpw)
    time.sleep(0.3)
    _prime(session)
    sigs = sorted((p for p in session._plots if not p.is_navigator),
                  key=lambda p: p.window_id)
    return sigs[0], sigs[1], nav, mm, navpw


def _layers_states(messages):
    return [m for m in messages if m.get("type") == "layers_state"]


def _drive_selector_to(mm, navpw, target_plot, y, x):
    """Run the selector that drives ``target_plot`` at nav position (y, x) via
    ``_run_update`` (the same code path a real drag takes), forcing the update.

    ``IntegratingSSelector2D`` is a COMPOSITE that delegates ``_run_update`` to its
    ACTIVE inner sub-selector (the crosshair), so we override + drive the inner
    selector — its ``_run_update``'s ``self`` is the inner one."""
    sel = [s for s in mm.navigation_selectors[navpw] if target_plot in s.children][0]
    inner = getattr(sel, "selector", sel)   # active sub-selector of the composite
    inner.get_selected_indices = lambda: np.array([[int(x), int(y)]])  # widget (cx, cy)
    inner._run_update(force=True)
    time.sleep(0.4)   # let the painter thread apply base + layer frames
    return sel


# ── add / validation ────────────────────────────────────────────────────────────


class TestOverlayAdd:
    def test_add_layer_same_shape(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        messages.clear()
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        assert len(tgt._layers) == 1
        # A real anyplotlib layer exists on the target's plot2d.
        assert len(tgt._plot2d._state.get("layers", [])) == 1
        # layers_state emitted for the target.
        st = _layers_states(messages)
        assert st and st[-1]["window_id"] == tgt.window_id
        assert len(st[-1]["layers"]) == 1
        layer = st[-1]["layers"][0]
        assert set(layer) >= {"id", "title", "cmap", "alpha", "clim", "visible"}
        assert layer["visible"] is True

    def test_refuse_shape_mismatch(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        nav = [p for p in session._plots if p.is_navigator][0]
        sig = [p for p in session._plots if not p.is_navigator][0]
        # navigator (4x5) vs signal (16x16) → refused with a status, no layer.
        messages.clear()
        ov.overlay_add(session, sig, {"window_id": sig.window_id,
                                      "source_window_id": nav.window_id})
        assert not getattr(sig, "_layers", [])
        assert any(m.get("type") == "status" for m in messages)

    def test_refuse_self(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        sig = [p for p in session._plots if not p.is_navigator][0]
        messages.clear()
        ov.overlay_add(session, sig, {"window_id": sig.window_id,
                                      "source_window_id": sig.window_id})
        assert not getattr(sig, "_layers", [])
        assert any(m.get("type") == "status" for m in messages)

    def test_query_reemits_state(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        messages.clear()
        ov.overlay_query(session, tgt, {"window_id": tgt.window_id})
        st = _layers_states(messages)
        assert st and len(st[-1]["layers"]) == 1


# ── overlay_set ─────────────────────────────────────────────────────────────────


class TestShapeChangeDropsLayers:
    def test_shape_changing_paint_drops_layers_cleanly(self, stem_4d_dataset):
        """anyplotlib raises on a shape-changing set_data while layers exist; the
        plot must drop its layers FIRST (status + empty layers_state), not raise."""
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        assert len(tgt._layers) == 1
        messages.clear()
        old_shape = tgt._plot2d._state["image_height"], tgt._plot2d._state["image_width"]
        new_frame = np.random.rand(old_shape[0] * 2, old_shape[1] * 2).astype(np.float32)
        tgt._set_array(new_frame)          # must not raise
        assert tgt._layers == []
        st = _layers_states(messages)
        assert st and st[-1]["window_id"] == tgt.window_id
        assert st[-1]["layers"] == []
        assert any(m.get("type") == "status" and "shape changed" in m.get("text", "")
                   for m in messages)


class TestOverlaySet:
    def test_set_reflected_in_layer_state(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        lid = tgt._layers[0].layer_id
        messages.clear()
        ov.overlay_set(session, tgt, {"window_id": tgt.window_id, "layer_id": lid,
                                      "cmap": "plasma", "alpha": 0.2, "visible": False})
        # The anyplotlib layer entry reflects the change.
        entry = tgt._plot2d._state["layers"][0]
        assert entry["cmap"] == "plasma"
        assert abs(entry["alpha"] - 0.2) < 1e-9
        assert entry["visible"] is False
        # And the emitted layers_state.
        st = _layers_states(messages)[-1]["layers"][0]
        assert st["cmap"] == "plasma"
        assert st["visible"] is False


# ── live nav refresh ────────────────────────────────────────────────────────────


class TestLiveNavRefresh:
    def test_nav_move_refreshes_layer(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        layer = tgt._layers[0]

        # Record every frame the layer handle receives.
        pushed = []
        orig = layer.handle.set_data

        def _rec(frame):
            pushed.append(float(np.asarray(frame).mean()))
            return orig(frame)

        layer.handle.set_data = _rec

        # Drive to two DISTINCT nav positions; the layer must refresh from the
        # SOURCE's frame at each (the fixture data varies per nav index).
        _drive_selector_to(mm, navpw, tgt, 0, 0)
        _drive_selector_to(mm, navpw, tgt, 3, 4)

        assert len(pushed) >= 2, f"layer not refreshed on nav move (got {pushed})"
        # The two positions produce different source frames.
        assert pushed[0] != pushed[-1], (
            f"layer frame did not change across nav positions: {pushed}")

    def test_async_tier_move_skips_without_error(self, stem_4d_dataset, monkeypatch):
        """When a layer read would be EXPENSIVE (async tier), _read_source_frame
        returns None → refresh_plot_layers pushes nothing and does not raise (the
        selector's settle re-fire runs the cheap path and catches the layer up)."""
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        layer = tgt._layers[0]

        pushed = []
        orig = layer.handle.set_data
        layer.handle.set_data = lambda f: pushed.append(1) or orig(f)

        # Simulate the expensive-tier skip: the read helper returns None.
        import spyde.actions.overlay as ovmod
        monkeypatch.setattr(ovmod, "_read_source_frame", lambda *a, **k: None)
        ov.refresh_plot_layers(tgt, np.array([[1, 1]]))
        time.sleep(0.2)
        assert pushed == [], "layer refreshed on an expensive-tier (skipped) read"


# ── teardown + tile-mode refusal ────────────────────────────────────────────────


class TestOverlayTeardown:
    def test_remove_layer(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        lid = tgt._layers[0].layer_id
        messages.clear()
        ov.overlay_remove(session, tgt, {"window_id": tgt.window_id, "layer_id": lid})
        assert tgt._layers == []
        assert tgt._plot2d._state.get("layers", []) == []
        assert _layers_states(messages)[-1]["layers"] == []

    def test_target_close_drops_layers(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        assert len(tgt._layers) == 1
        # Closing the TARGET plot drops its layers cleanly.
        tgt.close()
        assert tgt._layers == []

    def test_source_close_drops_layers_on_target(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        assert len(tgt._layers) == 1
        messages.clear()
        # Closing the SOURCE plot must drop the target's layer that sourced from it.
        src.close()
        assert tgt._layers == []
        # A layers_state was re-emitted for the affected target.
        assert any(m.get("type") == "layers_state"
                   and m.get("window_id") == tgt.window_id for m in messages)

    def test_tile_mode_refused(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        # Force the target into tile mode.
        tgt._plot2d._tile_on = True
        messages.clear()
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        assert not getattr(tgt, "_layers", [])
        assert any(m.get("type") == "status" and "tile" in str(m.get("text", "")).lower()
                   for m in messages)


# ── finding #1: pending-frames lost-update race ───────────────────────────────────


class _FakePlot:
    """Minimal target-plot stand-in for the pending-frames slot machinery: it only
    needs `_pending_layer_frames`, `current_data`, and `enqueue_paint`. We make
    enqueue_paint drain the slot INLINE (as the real painter thread would) so the
    read-then-clear happens on a controllable thread."""

    def __init__(self):
        self._pending_layer_frames = None
        self.current_data = None            # None → _enqueue_layer_push drains inline

    def enqueue_paint(self, data):          # unused (current_data is None)
        pass


class _RecordingHandle:
    def __init__(self):
        self.frames = []

    def set_data(self, frame):
        self.frames.append(frame)


class TestPendingFramesRace:
    def test_write_between_read_and_clear_is_not_lost(self, monkeypatch):
        """The painter's take-and-clear of `_pending_layer_frames` must be ATOMIC
        against a dispatcher write. We wedge the painter INSIDE the critical section
        (a slow read) and fire a newer write; with the lock the write blocks until
        the painter finishes its swap, so the newer frames are applied on the next
        drain — never silently dropped (finding #1: the settle re-fire's frames were
        the lost ones)."""
        import threading

        plot = _FakePlot()
        handle = _RecordingHandle()
        layer = ov.PlotLayer(layer_id="L", source_plot=None, handle=handle)

        old_frame = np.zeros((2, 2), np.float32)
        new_frame = np.ones((2, 2), np.float32)

        # Choreograph: the painter enters the critical section and pauses (still
        # holding _PENDING_LOCK) so the writer thread races to set newer frames.
        in_crit = threading.Event()
        release = threading.Event()
        real_getattr_slot = {"read": 0}

        orig_lock = ov._PENDING_LOCK

        class _InstrumentedLock:
            """Wrap the real lock: on the painter's ACQUIRE (the first, i.e. the
            take-and-clear), signal that we're in the section and block until the
            writer has attempted its (lock-contended) write."""
            def __enter__(self):
                orig_lock.acquire()
                # First acquire is the painter's take-and-clear.
                if real_getattr_slot["read"] == 0:
                    real_getattr_slot["read"] = 1
                    in_crit.set()
                    release.wait(2.0)
                return self

            def __exit__(self, *a):
                orig_lock.release()
                return False

        monkeypatch.setattr(ov, "_PENDING_LOCK", _InstrumentedLock())

        # Seed OLD frames, then have the painter drain (it will wedge in-section).
        plot._pending_layer_frames = [(layer, handle, old_frame)]

        painter = threading.Thread(
            target=ov._apply_pending_layer_frames, args=(plot,))
        painter.start()
        assert in_crit.wait(2.0), "painter never entered the critical section"

        # Writer: set NEWER frames while the painter holds the lock. This ACQUIRE
        # must block until the painter's swap completes (atomic take-and-clear).
        wrote = threading.Event()

        def _writer():
            ov._enqueue_layer_push(plot, [(layer, handle, new_frame)])
            wrote.set()

        wt = threading.Thread(target=_writer)
        wt.start()
        # The writer is now blocked on _PENDING_LOCK (painter holds it).
        assert not wrote.wait(0.3), "writer wrote while painter held the lock (race!)"

        release.set()          # let the painter finish its swap (clears OLD)
        painter.join(2.0)
        assert wrote.wait(2.0)  # writer now proceeds, sets NEW into the slot
        wt.join(2.0)

        # Painter applied the OLD frames it read; the NEW frames are pending (not
        # dropped). Drain again → NEW applied. Nothing lost.
        ov._apply_pending_layer_frames(plot)
        means = [float(f.mean()) for f in handle.frames]
        assert means[0] == 0.0, "old frames not applied"
        assert 1.0 in means, "NEWER frames were LOST (lost-update race not fixed)"


# ── finding #2: cold derived-source read skips + warms off-thread ─────────────────


class TestColdSourceReadWarms:
    def test_cold_derived_read_skips_then_warms_then_paints(self, monkeypatch):
        """A single-point read on a LAZY source whose chunk is NOT resident must NOT
        block the dispatcher: the first refresh returns None (skip) and warms the
        chunk off-thread; once warm, a subsequent read returns the frame. We drive
        _read_source_frame directly with a fake lazy source + a _NavChunkCache."""
        import concurrent.futures as cf
        import types
        import dask.array as da
        import hyperspy.api as hs
        from spyde.drawing.update_functions import _NavChunkCache

        # A lazy derived-style movie: 6 frames, 1/chunk, NO CachedDaskArray.
        frames = np.stack([np.full((4, 4), i, np.float32) for i in range(6)])
        arr = da.from_array(frames, chunks=(1, 4, 4))
        sig = hs.signals.Signal2D(arr).as_lazy()

        # A synchronous ComputeBackend stand-in: submit() runs the warm inline and
        # returns a resolved concurrent.futures.Future (so the warm populates the
        # source's chunk cache immediately, as a real off-thread warm eventually does).
        def _submit(fn, *a, **k):
            f = cf.Future()
            try:
                f.set_result(fn(*a, **k))
            except Exception as e:              # pragma: no cover
                f.set_exception(e)
            return f

        backend = types.SimpleNamespace(submit=_submit)
        session_stub = types.SimpleNamespace(compute_backend=backend)
        plot_state = types.SimpleNamespace(current_signal=sig)
        src = types.SimpleNamespace(
            plot_state=plot_state, window_id=99,
            _nav_chunk_cache=_NavChunkCache(), session=session_stub)
        indices = np.array([2])                 # single nav point (frame 2)

        # COLD: chunk not resident → skip (None) and warm off-thread (synchronous
        # backend runs the warm immediately, populating the cache).
        first = ov._read_source_frame(src, indices)
        assert first is None, "cold derived read did not skip the dispatcher"
        assert src._nav_chunk_cache.is_resident(sig, sig.data, indices), \
            "cold miss did not warm the source chunk off-thread"

        # WARM: now resident → the read returns the actual frame (settle re-fire path).
        second = ov._read_source_frame(src, indices)
        assert second is not None, "warm read still skipped"
        assert float(np.asarray(second).mean()) == 2.0, "wrong frame after warm"


# ── finding #3: dead layer is skipped by refresh ──────────────────────────────────


class TestDeadLayerSkip:
    def test_dead_layer_not_read_or_painted(self, stem_4d_dataset):
        """A layer marked `dead` (a drop path in progress) must be skipped by
        refresh_plot_layers — its source is never dereferenced and no set_data is
        queued (finding #3: source-teardown race)."""
        session = stem_4d_dataset["window"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        layer = tgt._layers[0]
        pushed = []
        layer.handle.set_data = lambda f: pushed.append(1)

        # Mark dead (as drop_layers_for_source would, before removal) and refresh.
        layer.dead = True
        # A source that would RAISE if dereferenced — proves we short-circuit early.
        layer.source_plot = None
        ov.refresh_plot_layers(tgt, np.array([1]))
        time.sleep(0.15)
        assert pushed == [], "a dead layer was still read/painted"


# ── finding #6: eager region layer value matches the base (un-rounded) ────────────


class TestEagerRegionParity:
    def test_eager_region_mean_is_unrounded(self):
        """The overlay eager integrating-region read must return the UN-rounded float
        mean — identical to the base eager region read (update_functions ~1149,
        ``data[sl].mean(axis=0)``) — NOT ``np.rint(...).astype(dtype)``. Otherwise the
        layer diverges from the base frame it composites over (finding #6)."""
        import hyperspy.api as hs
        from spyde.drawing.update_functions import _prepare_nav_indices

        # Integer eager 4-D source, nav (2, 2) so no clamp ambiguity; per-nav frames
        # chosen so an averaged pair is fractional (…, .5).
        data = np.zeros((2, 2, 2, 2), dtype=np.uint16)
        for iy in range(2):
            for ix in range(2):
                data[iy, ix] = np.array([[iy, ix], [iy + ix, iy * 2 + ix]], np.uint16)
        sig = hs.signals.Signal2D(data)                       # eager (numpy)

        class _PS:
            current_signal = sig

        class _SrcPlot:
            plot_state = _PS()
            window_id = 7

        # A 2-nav-point integrating region (widget (cx, cy) pairs).
        region_idx = np.array([[0, 0], [1, 1]])
        frame = ov._read_source_frame(_SrcPlot(), region_idx, integrating=True)
        assert frame is not None

        # The BASE eager region read, driven through the SAME index prep the overlay
        # uses, then `data[sl].mean(axis=0)` UN-rounded (verbatim base behaviour).
        idx = _prepare_nav_indices(sig, region_idx, integrating=True)
        sl = tuple(idx[:, k].astype(int) for k in range(idx.shape[1]))
        base = np.asarray(sig.data[sl]).mean(axis=0)
        assert np.array_equal(frame, base), "layer eager region diverges from base"
        # It is genuinely fractional — not rounded to the int dtype.
        assert np.any(frame != np.rint(frame)), "layer region was rounded (divergence)"
