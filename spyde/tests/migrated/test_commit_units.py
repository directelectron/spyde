"""
test_commit_units.py — a committed result map states its scale and its meaning.

``commit_result_tree`` is the single funnel every result goes through — strain,
DPC, orientation, virtual images — and it used to build ``Signal2D(data)`` with
nothing but a title. So a strain map's ticks read in pixels rather than nm, and
nothing anywhere recorded that ω is radians or that a DPC magnetic map is in
mrad: the unit lived in a chip label, which is text, and in the head of whoever
ran it.

These pin the two things it now carries — the source scan's spatial calibration,
and ``metadata.Signal.quantity`` — plus the plot behaviour that makes the second
one visible.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions.commit import (
    commit_result_tree, copy_navigation_calibration,
)


def _calibrated_source(ny=6, nx=5, scale=2.5, units="nm"):
    """A 4-D scan whose NAVIGATION axes carry a real spatial calibration."""
    import hyperspy.api as hs
    sig = hs.signals.Signal2D(np.random.rand(ny, nx, 4, 4).astype(np.float32))
    for axis in sig.axes_manager.navigation_axes:
        axis.scale, axis.offset, axis.units = scale, -1.0, units
    return sig


class TestNavigationCalibration:
    def test_the_scan_calibration_lands_on_the_result(self):
        import hyperspy.api as hs
        source = _calibrated_source()
        target = hs.signals.Signal2D(np.zeros((6, 5), np.float32))

        copy_navigation_calibration(source, target)

        for axis in target.axes_manager.signal_axes:
            assert axis.scale == pytest.approx(2.5)
            assert axis.offset == pytest.approx(-1.0)
            assert str(axis.units) == "nm"

    def test_a_5d_scan_hands_over_its_SPATIAL_nav_axes_not_the_clock(self):
        # A time-resolved scan navigates (t, y, x) in memory, and the map it
        # produces is the (y, x) pair.
        import hyperspy.api as hs
        source = hs.signals.Signal2D(np.zeros((3, 6, 5, 4, 4), np.float32))
        # hyperspy presents nav axes REVERSED from the array: (x, y, t).
        x, y, t = source.axes_manager.navigation_axes
        t.scale, t.units = 0.5, "s"
        y.scale, y.units = 2.5, "nm"
        x.scale, x.units = 2.5, "nm"
        target = hs.signals.Signal2D(np.zeros((6, 5), np.float32))

        copy_navigation_calibration(source, target)

        # The clock must NOT end up labelling a spatial axis.
        assert [str(a.units) for a in target.axes_manager.signal_axes] == ["nm", "nm"]

    def test_a_mismatched_grid_keeps_pixel_axes(self):
        # A rebinned or cropped fit is NOT on the scan's grid. Labelling it with
        # the scan's calibration would be a confident lie; pixels are honest.
        import hyperspy.api as hs
        source = _calibrated_source(ny=6, nx=5)
        target = hs.signals.Signal2D(np.zeros((3, 3), np.float32))

        copy_navigation_calibration(source, target)

        for axis in target.axes_manager.signal_axes:
            assert axis.scale == pytest.approx(1.0)


class TestCommittedTree:
    def test_every_node_gets_the_scan_calibration(self, window):
        session = window["window"]
        source = _calibrated_source()
        tree = commit_result_tree(
            session, title="Strain", primary=np.zeros((6, 5), np.float32),
            primary_label="εxx",
            views=[("εyy", np.zeros((6, 5), np.float32))],
            source_signal=source,
        )
        # The root AND the child view — a saved tree carries every component, so
        # every component has to be calibrated, not just the one on screen.
        signals = [tree.root] + _child_signals(tree)
        assert len(signals) >= 2, "commit produced no child view node"
        for sig in signals:
            for axis in sig.axes_manager.signal_axes:
                assert str(axis.units) == "nm"
                assert axis.scale == pytest.approx(2.5)

    def test_per_component_units_are_stamped_by_label(self, window):
        session = window["window"]
        tree = commit_result_tree(
            session, title="Strain", primary=np.zeros((6, 5), np.float32),
            primary_label="εxx",
            views=[("εyy", np.zeros((6, 5), np.float32)),
                   ("ω", np.zeros((6, 5), np.float32))],
            value_units={"εxx": "dimensionless", "εyy": "dimensionless",
                         "ω": "rad"},
        )
        got = {str(s.metadata.General.title): _quantity(s)
               for s in [tree.root] + _child_signals(tree)}
        assert got["Strain"] == "dimensionless"          # the primary
        assert got["Strain ω"] == "rad"

    def test_one_string_applies_to_every_node(self, window):
        session = window["window"]
        tree = commit_result_tree(
            session, title="DPC (B)", primary=np.zeros((6, 5), np.float32),
            primary_label="Bx (mrad)",
            views=[("By (mrad)", np.zeros((6, 5), np.float32))],
            value_units="mrad",
        )
        for sig in [tree.root] + _child_signals(tree):
            assert _quantity(sig) == "mrad"

    def test_an_rgb_primary_states_no_value_unit(self, window):
        # An IPF / DPC direction map is colour, not a measurement — and its root
        # is a zeros placeholder, so a unit on it would describe nothing.
        session = window["window"]
        tree = commit_result_tree(
            session, title="DPC (B)", primary=np.zeros((6, 5, 3), np.uint8),
            primary_label="B direction",
            views=[("Bx (mrad)", np.zeros((6, 5), np.float32))],
            levels=None, value_units="mrad",
        )
        assert _quantity(tree.root) == ""

    def test_omitting_both_is_still_a_valid_commit(self, window):
        # Every existing caller passes neither; they must be unaffected.
        session = window["window"]
        tree = commit_result_tree(
            session, title="Plain", primary=np.zeros((6, 5), np.float32))
        assert tree is not None
        assert _quantity(tree.root) == ""


class TestColorbarShowsTheUnit:
    """The unit has to reach the FIGURE — a value in metadata nobody draws is
    the same as no value. This is the one place SpyDE has ever turned
    anyplotlib's colorbar on."""

    def test_a_quantity_turns_the_colorbar_on_with_that_label(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        plot = _signal_plot(session)
        plot.plot_state.current_signal.metadata.set_item("Signal.quantity", "mrad")
        plot._last_colorbar_label = ""          # as if freshly built

        plot._sync_value_colorbar()

        assert plot._plot2d._state["show_colorbar"] is True
        assert plot._plot2d._state["colorbar_label"] == "mrad"

    def test_no_quantity_draws_no_strip(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        plot = _signal_plot(session)
        before = plot._plot2d._state.get("show_colorbar")

        plot._sync_value_colorbar()

        # An unlabelled colorbar is decoration; a raw dataset gets none, and the
        # sync must not even push to say so.
        assert plot._plot2d._state.get("show_colorbar") == before

    def test_re_syncing_the_same_label_does_not_push_again(self, tem_2d_dataset):
        # This runs on the paint path, once per frame.
        session = tem_2d_dataset["window"]
        plot = _signal_plot(session)
        plot.plot_state.current_signal.metadata.set_item("Signal.quantity", "rad")
        plot._last_colorbar_label = ""
        plot._sync_value_colorbar()

        calls = []
        plot._plot2d.set_colorbar_label = lambda *a, **k: calls.append(a)
        plot._plot2d.set_colorbar_visible = lambda *a, **k: calls.append(a)
        plot._sync_value_colorbar()
        plot._sync_value_colorbar()

        assert calls == []

    def test_latex_units_are_cleaned_like_axis_units(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        plot = _signal_plot(session)
        plot.plot_state.current_signal.metadata.set_item(
            "Signal.quantity", "$\\AA^{-1}$")
        plot._last_colorbar_label = ""

        plot._sync_value_colorbar()

        assert plot._plot2d._state["colorbar_label"] == "Å⁻¹"


# ── helpers ───────────────────────────────────────────────────────────────────


def _quantity(signal) -> str:
    try:
        return str(signal.metadata.get_item("Signal.quantity", default="") or "")
    except Exception:
        return ""


def _child_signals(tree):
    """Every non-root signal in a committed tree."""
    return [s for s in tree.signals() if s is not tree.root]


def _signal_plot(session):
    for p in session._plots:
        if not getattr(p, "is_navigator", False):
            return p
    return session._plots[0]
