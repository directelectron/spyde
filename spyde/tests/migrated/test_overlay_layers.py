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
