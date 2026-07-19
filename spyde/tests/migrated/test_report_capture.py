"""
test_report_capture.py — "Capture to presentation" (one-click).

``report_add_figure`` already snapshots a source window's CURRENT live state
(nav position, contrast, colormap, overlays) into a figure cell via
``_snapshot_plot``. The capture feature is mostly wiring on top of that:

  * ``slide_break`` in the payload marks the new cell as the START of its own
    slide (mirrors ``report_add_cell``'s existing handling of the field).
  * a payload with NO ``source_window_id`` falls back to
    ``session._active_window_id`` (the focused window) — the renderer's
    per-window camera button / sidebar Capture button always pass an explicit
    id, but this fallback lets a purely backend-driven capture work too, and
    is the natural "capture whatever I'm looking at" contract.

Exercises both against a real Qt-free ``Session`` (no Qt, no mocked snapshot).
"""
from __future__ import annotations

import numpy as np

from spyde.actions.report import handlers as h


def _prime_plot_data(session):
    """Stamp current_data/_last_levels on every Plot so _snapshot_plot has a
    real frame to capture (mirrors test_report_compose.py's helper)."""
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


def _signal_wid(session):
    for p in session._plots:
        if not getattr(p, "is_navigator", False) and p.window_id is not None:
            return p.window_id
    return session._plots[0].window_id


def _last_state(messages):
    states = [m for m in messages if m.get("type") == "report_state"]
    assert states, "no report_state emitted"
    return states[-1]["report"]


class TestCaptureToPresentation:
    def test_capture_sets_slide_break_and_snapshot(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)

        h.report_add_figure(session, None, {
            "source_window_id": wid,
            "slide_break": True,
            "vectors_mode": "image",
        })

        st = _last_state(messages)
        fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
        assert fig_cells, "no figure cell appended"
        cell = fig_cells[-1]
        assert cell["slide_break"] is True
        assert cell.get("figure") is not None, "no FigureSpec on the captured cell"

        mgr = h._manager(session)
        assert cell["id"] in mgr._snapshots, "no snapshot map stored for the captured cell"
        assert mgr._snapshots[cell["id"]], "snapshot map is empty"

    def test_capture_defaults_slide_break_false_when_omitted(self, tem_2d_dataset):
        """Existing drag-and-drop callers that never pass slide_break must keep
        landing on the current slide (no accidental new-slide regressions)."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)

        h.report_add_figure(session, None, {"source_window_id": wid})

        st = _last_state(messages)
        fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
        assert fig_cells[-1]["slide_break"] is False

    def test_capture_falls_back_to_active_window(self, tem_2d_dataset):
        """No source_window_id in the payload → capture session._active_window_id
        (the focused window) — the pure-backend 'capture whatever I'm looking
        at' contract."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        session._active_window_id = wid

        h.report_add_figure(session, None, {"slide_break": True, "vectors_mode": "image"})

        st = _last_state(messages)
        fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
        assert fig_cells, "no figure cell appended via active-window fallback"
        assert fig_cells[-1]["slide_break"] is True

    def test_capture_no_active_window_errors_cleanly(self, tem_2d_dataset):
        """No source_window_id AND no active window → a clean error, no crash,
        no cell appended."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        session._active_window_id = None

        h.report_add_figure(session, None, {"slide_break": True})

        errors = [m for m in messages if m.get("type") == "error"]
        assert errors, "expected an error message when no window can be resolved"
        states = [m for m in messages if m.get("type") == "report_state"]
        if states:
            fig_cells = [c for c in states[-1]["report"]["cells"] if c["cell_type"] == "figure"]
            assert not fig_cells
