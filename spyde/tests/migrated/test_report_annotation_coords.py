"""test_report_annotation_coords.py — Report Builder annotation DATA→PIXEL fix.

Root cause: report annotation dicts (``PanelSpec.annotations``) store offsets
and sizes in calibrated DATA coordinates (the on-disk spec/YAML contract), but
anyplotlib's 2-D marker path (``drawMarkers2d`` → ``_imgToCanvas2d``) renders
offsets as IMAGE PIXELS with no data→px conversion. ``figure_builder._apply_
annotations`` now converts a COPY of each annotation dict at render time via
``spyde.actions.report.coords`` using the panel's calibrated axes, leaving the
spec's stored dicts (and the on-disk YAML contract) untouched.

Qt-free: builds a FigureSpec/PanelSpec/LayerSpec directly (no Session needed —
``build_cell_figure`` only touches anyplotlib + numpy) and inspects the built
Plot2D's wire-format marker state (``p2._state["markers"]``), matching the
idiom anyplotlib's own marker tests use.
"""
from __future__ import annotations

import copy

import numpy as np

from spyde.actions.report.figure_builder import build_cell_figure
from spyde.actions.report.model import FigureSpec, LayerSpec, PanelSpec, SignalRef


# ── helpers ────────────────────────────────────────────────────────────────────


def _linspace_axes(n=128, lo=0.0, hi=12.0, units="nm", reverse=False):
    vals = np.linspace(lo, hi, n)
    if reverse:
        vals = vals[::-1]
    return {"units": units, "x_axis": [float(v) for v in vals],
            "y_axis": [float(v) for v in vals]}


def _make_spec(annotations, axes=None, shape=(128, 128)):
    layer = LayerSpec(source=SignalRef(title="t"), cmap="gray")
    panel = PanelSpec(id="p1", grid_pos=[0, 0], kind="image",
                      layers=[layer], axes=axes, annotations=annotations)
    spec = FigureSpec(layout={"kind": "single"}, panels=[panel])
    snap = {("p1", layer.id): np.zeros(shape, dtype=np.float32)}
    return spec, snap


def _wires(p2, type_):
    return [m for m in p2._state["markers"] if m["type"] == type_]


def _first_panel_plot(fig):
    plots_map = getattr(fig, "_plots_map", None)
    assert plots_map, "figure has no panels"
    return next(iter(plots_map.values()))


# ── calibrated panel: text at data center → pixel center ───────────────────────


class TestCalibratedTextAnnotation:
    def test_text_center_converts_to_pixel_center(self):
        axes = _linspace_axes(n=128, lo=0.0, hi=12.0)
        ann = [{"kind": "text", "offsets": [6.0, 6.0], "texts": ["hi"],
               "fontsize": 10}]
        spec, snap = _make_spec(ann, axes=axes)
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        p2 = _first_panel_plot(fig)
        wires = _wires(p2, "texts")
        assert len(wires) == 1
        off = wires[0]["offsets"][0]
        # data 6.0 -> index (6.0 - 0.0) / (12/127) = 63.5
        assert abs(off[0] - 63.5) < 0.6
        assert abs(off[1] - 63.5) < 0.6
        assert wires[0]["texts"] == ["hi"]
        # fontsize is untouched (stays in points).
        assert wires[0]["fontsize"] == 10


# ── uncalibrated panel: identity ────────────────────────────────────────────────


class TestUncalibratedIdentity:
    def test_no_axes_is_identity(self):
        ann = [{"kind": "text", "offsets": [6.0, 6.0], "texts": ["hi"]}]
        spec, snap = _make_spec(ann, axes=None)
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        p2 = _first_panel_plot(fig)
        off = _wires(p2, "texts")[0]["offsets"][0]
        assert off == [6.0, 6.0]

    def test_unusable_axes_is_identity(self):
        # x_axis/y_axis missing -> _panel_data_to_pixel_scale returns None.
        ann = [{"kind": "text", "offsets": [6.0, 6.0], "texts": ["hi"]}]
        spec, snap = _make_spec(ann, axes={"units": "nm"})
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        p2 = _first_panel_plot(fig)
        off = _wires(p2, "texts")[0]["offsets"][0]
        assert off == [6.0, 6.0]


# ── size conversions: circle radius, rect widths/heights, arrow U/V ────────────


class TestSizeConversions:
    def test_circle_radius_converts_by_mean_scale(self):
        axes = _linspace_axes(n=128, lo=0.0, hi=12.0)
        step = 12.0 / 127.0
        ann = [{"kind": "circle", "offsets": [6.0, 6.0], "radius": step * 5.0}]
        spec, snap = _make_spec(ann, axes=axes)
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        p2 = _first_panel_plot(fig)
        size = _wires(p2, "circles")[0]["sizes"][0]
        assert abs(size - 5.0) < 1e-6

    def test_rect_widths_heights_convert(self):
        axes = _linspace_axes(n=128, lo=0.0, hi=12.0)
        step = 12.0 / 127.0
        ann = [{"kind": "rect", "offsets": [6.0, 6.0],
               "widths": step * 4.0, "heights": step * 8.0}]
        spec, snap = _make_spec(ann, axes=axes)
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        p2 = _first_panel_plot(fig)
        wire = _wires(p2, "rectangles")[0]
        assert abs(wire["widths"][0] - 4.0) < 1e-6
        assert abs(wire["heights"][0] - 8.0) < 1e-6

    def test_ellipse_widths_heights_convert(self):
        axes = _linspace_axes(n=128, lo=0.0, hi=12.0)
        step = 12.0 / 127.0
        ann = [{"kind": "ellipse", "offsets": [6.0, 6.0],
               "widths": step * 3.0, "heights": step * 6.0}]
        spec, snap = _make_spec(ann, axes=axes)
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        p2 = _first_panel_plot(fig)
        wire = _wires(p2, "ellipses")[0]
        assert abs(wire["widths"][0] - 3.0) < 1e-6
        assert abs(wire["heights"][0] - 6.0) < 1e-6

    def test_arrow_uv_scale_signed(self):
        axes = _linspace_axes(n=128, lo=0.0, hi=12.0)
        step = 12.0 / 127.0
        ann = [{"kind": "arrow", "offsets": [6.0, 6.0],
               "U": step * 2.0, "V": step * -3.0}]
        spec, snap = _make_spec(ann, axes=axes)
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        p2 = _first_panel_plot(fig)
        wire = _wires(p2, "arrows")[0]
        assert abs(wire["U"][0] - 2.0) < 1e-6
        assert abs(wire["V"][0] - (-3.0)) < 1e-6


# ── reversed axis: signed conversion for offsets and U ──────────────────────────


class TestReversedAxis:
    def test_reversed_x_axis_offsets_and_u_sign(self):
        # Descending x_axis (12 -> 0): step is negative.
        axes = _linspace_axes(n=128, lo=0.0, hi=12.0, reverse=True)
        # y_axis stays ascending (only x reversed) so this exercises independent
        # per-axis signed scale.
        axes["y_axis"] = list(np.linspace(0.0, 12.0, 128))
        x_step = (axes["x_axis"][1] - axes["x_axis"][0])  # negative
        assert x_step < 0
        ann = [{"kind": "arrow", "offsets": [6.0, 6.0], "U": 1.0, "V": 0.0}]
        spec, snap = _make_spec(ann, axes=axes)
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        p2 = _first_panel_plot(fig)
        wire = _wires(p2, "arrows")[0]
        # offset x: (6 - 12) / x_step = (-6) / x_step -> positive index near 63.5
        expected_off_x = (6.0 - axes["x_axis"][0]) / x_step
        assert abs(wire["offsets"][0][0] - expected_off_x) < 1e-6
        # U scaled (signed) by x_step -> negative since x_step is negative.
        expected_u = 1.0 / x_step
        assert abs(wire["U"][0] - expected_u) < 1e-6
        assert expected_u < 0


# ── spec dicts are not mutated by the build ─────────────────────────────────────


class TestSpecNotMutated:
    def test_build_twice_same_result_and_spec_untouched(self):
        axes = _linspace_axes(n=128, lo=0.0, hi=12.0)
        ann = [{"kind": "text", "offsets": [6.0, 6.0], "texts": ["hi"],
               "fontsize": 10}]
        spec, snap = _make_spec(ann, axes=axes)
        original = copy.deepcopy(spec.panels[0].annotations)

        fig1, _fid1, _html1 = build_cell_figure(spec, snap)
        p2_1 = _first_panel_plot(fig1)
        off1 = _wires(p2_1, "texts")[0]["offsets"][0]

        # The spec's stored annotation dicts must be untouched (still data coords).
        assert spec.panels[0].annotations == original
        assert spec.panels[0].annotations[0]["offsets"] == [6.0, 6.0]

        fig2, _fid2, _html2 = build_cell_figure(spec, snap)
        p2_2 = _first_panel_plot(fig2)
        off2 = _wires(p2_2, "texts")[0]["offsets"][0]

        assert off1 == off2
        assert spec.panels[0].annotations == original
