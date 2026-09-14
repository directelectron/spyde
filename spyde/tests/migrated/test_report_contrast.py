"""
test_report_contrast.py — display range (contrast) for a report figure layer.

A report figure's contrast used to be whatever the live plot happened to be
showing at drop time: ``LayerSpec.clim`` was captured from ``plot._last_levels``
and then unreachable, because no UI sent the ``clim`` key ``repfig_set_layer``
had always accepted. These are the two verbs that close that gap —
``repfig_histogram`` (what to aim at) and ``repfig_auto_clim`` (the dock's
Auto/Reset pair) — plus the contract that both bin through the SAME helper the
live plot dock does, so the two widgets cannot drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions.report import compose as C
from spyde.actions.report import handlers as h


def _signal_window_id(session):
    for p in session._plots:
        if not getattr(p, "is_navigator", False) and p.window_id is not None:
            return p.window_id
    return session._plots[0].window_id


def _prime_plot_data(session):
    for p in session._plots:
        if isinstance(getattr(p, "current_data", None), np.ndarray):
            continue
        try:
            frame = np.asarray(p.plot_state.current_signal.data)
            if frame.ndim > 2:
                frame = frame.reshape(-1, *frame.shape[-2:])[0]
            p.current_data = np.ascontiguousarray(frame.astype(np.float32))
            p._last_levels = (float(np.nanmin(p.current_data)),
                              float(np.nanmax(p.current_data)))
        except Exception:
            pass


def _figure_cell(session):
    for c in session._report.doc.cells:
        if c.cell_type == "figure":
            return c
    return None


def _one_layer(session):
    """``(cell, panel, layer)`` of the report's single figure cell."""
    cell = _figure_cell(session)
    panel = cell.spec.panels[0]
    return cell, panel, panel.layers[0]


def _args(cell, panel, layer):
    return {"cell_id": cell.id, "panel_id": panel.id, "layer_id": layer.id}


def _built(tem_2d_dataset):
    """A session with an open report holding one dropped figure cell."""
    session = tem_2d_dataset["window"]
    _prime_plot_data(session)
    h.report_new(session, None, {})
    h.report_add_figure(session, None,
                        {"source_window_id": _signal_window_id(session)})
    return session


def _histograms(messages):
    return [m for m in messages if m.get("type") == "repfig_histogram"]


class TestHistogram:
    def test_emits_bins_addressed_by_cell_panel_layer(self, tem_2d_dataset):
        session, messages = _built(tem_2d_dataset), tem_2d_dataset["messages"]
        cell, panel, layer = _one_layer(session)

        messages.clear()
        C.repfig_histogram(session, None, _args(cell, panel, layer))

        hist = _histograms(messages)
        assert len(hist) == 1
        msg = hist[0]
        # A report layer has no window of its own — it is addressed by identity.
        assert msg["cell_id"] == cell.id
        assert msg["panel_id"] == panel.id
        assert msg["layer_id"] == layer.id
        assert len(msg["counts"]) == 64
        assert len(msg["edges"]) == 65
        assert msg["data_min"] <= msg["data_max"]

    def test_bins_match_the_live_plot_dock(self, tem_2d_dataset):
        # The whole point of sharing histogram_payload: the report's bars and the
        # dock's bars describe the same frame the same way, so a user reading
        # them side by side is not comparing two different pictures.
        from de_shell.plotting.figure import histogram_payload, robust_levels

        session, messages = _built(tem_2d_dataset), tem_2d_dataset["messages"]
        cell, panel, layer = _one_layer(session)
        arr = np.asarray(session._report.snapshot_map(cell.id)[(panel.id, layer.id)])

        messages.clear()
        C.repfig_histogram(session, None, _args(cell, panel, layer))
        got = _histograms(messages)[0]

        band = robust_levels(arr, low=2.0, high=99.0)
        expected = histogram_payload(arr, *band, band)
        assert got["counts"] == expected["counts"]
        assert got["edges"] == expected["edges"]

    def test_unknown_cell_is_silent(self, tem_2d_dataset):
        session, messages = _built(tem_2d_dataset), tem_2d_dataset["messages"]
        messages.clear()
        C.repfig_histogram(session, None, {"cell_id": "nope", "panel_id": "p1",
                                           "layer_id": "l1"})
        assert _histograms(messages) == []

    def test_an_rgb_layer_has_nothing_to_window(self, tem_2d_dataset):
        session, messages = _built(tem_2d_dataset), tem_2d_dataset["messages"]
        cell, panel, layer = _one_layer(session)
        # An IPF / DPC direction map: colour, not a scalar the handles can move.
        session._report._snapshots[cell.id][(panel.id, layer.id)] = \
            np.zeros((8, 8, 3), np.uint8)

        messages.clear()
        C.repfig_histogram(session, None, _args(cell, panel, layer))
        assert _histograms(messages) == []


class TestAutoClim:
    def test_robust_writes_the_2_99_band_onto_the_spec(self, tem_2d_dataset):
        from de_shell.plotting.figure import robust_levels

        session = _built(tem_2d_dataset)
        cell, panel, layer = _one_layer(session)
        arr = np.asarray(session._report.snapshot_map(cell.id)[(panel.id, layer.id)])

        C.repfig_auto_clim(session, None, {**_args(cell, panel, layer),
                                           "mode": "robust"})
        _cell, _panel, layer = _one_layer(session)
        assert layer.clim == list(robust_levels(arr, low=2.0, high=99.0))

    def test_full_writes_the_true_extent(self, tem_2d_dataset):
        session = _built(tem_2d_dataset)
        cell, panel, layer = _one_layer(session)
        arr = np.asarray(session._report.snapshot_map(cell.id)[(panel.id, layer.id)])
        finite = arr[np.isfinite(arr)]

        C.repfig_auto_clim(session, None, {**_args(cell, panel, layer),
                                           "mode": "full"})
        _c, _p, layer = _one_layer(session)
        assert layer.clim == [float(finite.min()), float(finite.max())]

    def test_an_unknown_mode_clears_back_to_automatic(self, tem_2d_dataset):
        session = _built(tem_2d_dataset)
        cell, panel, layer = _one_layer(session)
        C.repfig_set_layer(session, None, {**_args(cell, panel, layer),
                                            "clim": [1.0, 2.0]})
        assert _one_layer(session)[2].clim == [1.0, 2.0]

        C.repfig_auto_clim(session, None, {**_args(cell, panel, layer),
                                           "mode": "clear"})
        assert _one_layer(session)[2].clim is None

    def test_re_levelling_re_emits_the_histogram(self, tem_2d_dataset):
        # The handles have to follow the new answer, so the round trip must end
        # with fresh bins rather than leaving them where the user last dragged.
        session, messages = _built(tem_2d_dataset), tem_2d_dataset["messages"]
        cell, panel, layer = _one_layer(session)
        messages.clear()
        C.repfig_auto_clim(session, None, {**_args(cell, panel, layer),
                                           "mode": "robust"})
        assert len(_histograms(messages)) == 1


class TestPersistence:
    def test_a_chosen_clim_survives_the_spec_round_trip(self, tem_2d_dataset):
        # Contrast is a property of the REPORT, not of whatever the live plot was
        # showing at drop time — so it has to reach figures/<id>.yaml.
        from spyde.actions.report.model import FigureSpec

        session = _built(tem_2d_dataset)
        cell, panel, layer = _one_layer(session)
        C.repfig_set_layer(session, None, {**_args(cell, panel, layer),
                                            "clim": [12.5, 88.25]})

        spec = FigureSpec.from_dict(cell.spec.to_dict())
        assert spec.panels[0].layers[0].clim == [12.5, 88.25]


# ── contrast in the EXPORTED page ─────────────────────────────────────────────


class TestExportedContrast:
    """A reader opening a saved report could not adjust contrast at all.

    Pixels cross the wire as 8-bit codes quantised over their clim, so anything
    outside the window is saturated to 0/255 before the file is written —
    widening was impossible, not merely unimplemented. The export now quantises
    over a wide band and windows with ``set_display_window``; these pin the
    packing that decides whether a control is offered.
    """

    @staticmethod
    def _spec(kind="image"):
        from spyde.actions.report.model import FigureSpec, LayerSpec, PanelSpec
        panel = PanelSpec(id="p1", kind=kind, layers=[LayerSpec(id="l1")],
                          title="Map")
        return FigureSpec(panels=[panel])

    @staticmethod
    def _light(raw, disp, dispatch="pABC123"):
        return {"p1": {"dispatch": dispatch,
                       "state": {"raw_min": raw[0], "raw_max": raw[1],
                                 "display_min": disp[0], "display_max": disp[1]}}}

    def _snap(self):
        return {("p1", "l1"): np.linspace(0, 100, 64).reshape(8, 8).astype(np.float32)}

    def test_headroom_yields_a_control(self):
        from spyde.actions.report.contrast_embed import pack_contrast

        payload = pack_contrast(self._spec(), self._snap(),
                                self._light((0.0, 100.0), (20.0, 80.0)))
        assert payload is not None
        entry = payload["panels"][0]
        assert (entry["raw_min"], entry["raw_max"]) == (0.0, 100.0)
        assert (entry["vmin"], entry["vmax"]) == (20.0, 80.0)
        assert len(entry["counts"]) == 64

    def test_no_headroom_offers_nothing(self):
        # What an older anyplotlib produces: set_clim quantised over the display
        # window, so raw == display and the LUT is the identity. A control that
        # cannot move anything is worse than none — the reader drags it and
        # concludes the report is broken.
        from spyde.actions.report.contrast_embed import pack_contrast

        assert pack_contrast(self._spec(), self._snap(),
                             self._light((20.0, 80.0), (20.0, 80.0))) is None

    def test_it_carries_the_DISPATCH_id_not_the_spec_id(self):
        # The page re-sends `panel_<id>_json`, and that id is anyplotlib's
        # dispatch id — not the spec's "p1". Sending the spec id addresses a
        # trait that does not exist, and nothing happens, silently.
        from spyde.actions.report.contrast_embed import pack_contrast

        payload = pack_contrast(self._spec(), self._snap(),
                                self._light((0.0, 100.0), (20.0, 80.0),
                                            dispatch="pDEADBEEF"))
        assert payload["panels"][0]["panel"] == "pDEADBEEF"

    def test_the_state_it_re_sends_is_the_light_one(self):
        from spyde.actions.report.contrast_embed import pack_contrast

        payload = pack_contrast(self._spec(), self._snap(),
                                self._light((0.0, 100.0), (20.0, 80.0)))
        state = payload["panels"][0]["state"]
        # Pixels ride the separate geom trait, which is what makes re-sending
        # this per drag tick cheap enough to be smooth.
        assert "image_b64" not in state

    def test_a_line_panel_is_skipped(self):
        from spyde.actions.report.contrast_embed import pack_contrast

        assert pack_contrast(self._spec(kind="line"), self._snap(),
                             self._light((0.0, 100.0), (20.0, 80.0))) is None

    def test_an_rgb_snapshot_is_skipped(self):
        from spyde.actions.report.contrast_embed import pack_contrast

        snap = {("p1", "l1"): np.zeros((8, 8, 3), np.uint8)}
        assert pack_contrast(self._spec(), snap,
                             self._light((0.0, 100.0), (20.0, 80.0))) is None

    def test_the_page_carries_no_absolute_url(self):
        # A self-contained export must never reach for the network; the SVG
        # namespace URI in createElementNS would have put one in the source.
        from spyde.actions.report.figure_chrome import CHROME_CSS, CHROME_JS

        assert "http://" not in CHROME_JS
        assert "https://" not in CHROME_JS
        assert "http" not in CHROME_CSS


class TestExportWidensTheQuantisationBand:
    """The other half: the export has to ENCODE with headroom, or the control
    above has nothing to work with."""

    def test_a_standalone_build_quantises_wider_than_the_window(self):
        # Must hold on the CURRENT pin, not just a future anyplotlib: gating this
        # on set_display_window is what left the exported report with no contrast
        # button at all.
        from spyde.actions.report.figure_builder import build_cell_figure
        from spyde.actions.report.model import FigureSpec, LayerSpec, PanelSpec

        frame = np.linspace(0, 100, 4096).reshape(64, 64).astype(np.float32)
        panel = PanelSpec(id="p1", layers=[LayerSpec(id="l1", clim=[20.0, 80.0])])
        spec = FigureSpec(layout={"kind": "single"}, panels=[panel])

        fig, _id, _html = build_cell_figure(spec, {("p1", "l1"): frame},
                                            standalone=True)
        state = fig._report_light_state["p1"]["state"]
        assert state["display_min"] == 20.0 and state["display_max"] == 80.0
        assert state["raw_min"] < 20.0 and state["raw_max"] > 80.0

    def test_the_in_app_build_keeps_its_precision(self):
        # In-app has a backend that can re-encode on demand, so it should spend
        # all 8 bits on what is visible rather than on unseen headroom.
        from spyde.actions.report.figure_builder import build_cell_figure
        from spyde.actions.report.model import FigureSpec, LayerSpec, PanelSpec

        frame = np.linspace(0, 100, 4096).reshape(64, 64).astype(np.float32)
        panel = PanelSpec(id="p1", layers=[LayerSpec(id="l1", clim=[20.0, 80.0])])
        spec = FigureSpec(layout={"kind": "single"}, panels=[panel])

        fig, _id, _html = build_cell_figure(spec, {("p1", "l1"): frame},
                                            standalone=False)
        plot = next(iter(fig._plots_map.values()))
        assert plot._state["raw_min"] == 20.0
        assert plot._state["raw_max"] == 80.0
