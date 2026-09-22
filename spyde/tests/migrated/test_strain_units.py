"""
test_strain_units.py — a strain map says how much, in the units people quote.

The fit produces fractional strain and radians (``StrainField``). Everything a
user reads — the live window, its histogram handles, the colorbar, the chip
views and the committed tree — shows PERCENT strain and DEGREES of rotation,
labelled as such, over the scan's own calibrated axes with a scale bar. Before
this a strain map was an unlabelled colour field in pixels: nothing on screen
said whether a red patch was 0.2 % or 2 %.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions._common import STRAIN_TITLES, strain_quantity
from spyde.actions.commit import navigation_extent
from spyde.actions.strain_display import (
    build_strain_figure, display_component, update_strain_view,
)
from spyde.actions.strain_mapping import StrainField


def _field(ny=4, nx=5):
    exx = np.full((ny, nx), 0.012, np.float64)          # 1.2 % tensile
    eyy = np.full((ny, nx), -0.005, np.float64)         # 0.5 % compressive
    exy = np.zeros((ny, nx), np.float64)
    omega = np.full((ny, nx), np.pi / 180.0)            # exactly one degree
    return StrainField(exx=exx, eyy=eyy, exy=exy, omega=omega,
                       coverage=np.ones((ny, nx)))


def _calibrated_scan(ny=4, nx=5, scale=2.5, units="nm"):
    """A 4-D scan whose navigation axes carry a real spatial calibration."""
    import hyperspy.api as hs
    sig = hs.signals.Signal2D(np.zeros((ny, nx, 3, 3), np.float32))
    for axis in sig.axes_manager.navigation_axes:
        axis.scale, axis.offset, axis.units = scale, 0.0, units
    return sig


class TestDisplayUnits:
    def test_strain_reads_in_percent_and_rotation_in_degrees(self):
        field = _field()
        assert np.allclose(display_component(field, "exx"), 1.2)
        assert np.allclose(display_component(field, "eyy"), -0.5)
        assert np.allclose(display_component(field, "omega"), 1.0)

    def test_the_fit_itself_is_untouched(self):
        # The scale is applied on the way to the screen, never to the field.
        field = _field()
        display_component(field, "exx")
        assert np.allclose(field.exx, 0.012)

    def test_the_quantity_names_the_component_and_its_unit(self):
        assert strain_quantity("exx") == "εxx (%)"
        assert strain_quantity("omega") == "ω (°)"


class TestLiveWindow:
    def test_the_map_is_labelled_and_calibrated(self):
        field = _field()
        x, y = np.arange(5) * 2.5, np.arange(4) * 2.5
        _fig, _fig_id, _html, p = build_strain_figure(
            field, component="exx", ref_yx=(1, 2), axes=(x, y, "nm"))

        state = p._state
        assert state["show_colorbar"] is True
        assert state["colorbar_label"] == "εxx (%)"
        assert state["has_axes"] is True and state["units"] == "nm"
        assert state["x_axis"] == pytest.approx(list(x))

    def test_the_reference_crosshair_lands_on_the_calibrated_pixel(self):
        # Markers are placed in data coordinates; once the map is in nm the
        # crosshair must be too, or it draws at the wrong place.
        field = _field()
        x, y = np.arange(5) * 2.5, np.arange(4) * 2.5
        _fig, _fig_id, _html, p = build_strain_figure(
            field, component="exx", ref_yx=(1, 2), axes=(x, y, "nm"))
        marker = next(m for m in p._state["markers"] if m["name"] == "strain_ref")
        horizontal, vertical = marker["segments"]
        assert horizontal[0][1] == pytest.approx(2.5)     # row 1 → 2.5 nm
        assert vertical[0][0] == pytest.approx(5.0)       # column 2 → 5.0 nm

    def test_an_uncalibrated_scan_keeps_pixel_axes_but_still_says_percent(self):
        _fig, _fig_id, _html, p = build_strain_figure(_field(), component="exx")
        assert not p._state["has_axes"]
        assert p._state["colorbar_label"] == "εxx (%)"

    def test_switching_the_component_relabels_the_colorbar(self):
        field = _field()
        _fig, _fig_id, _html, p = build_strain_figure(field, component="exx")
        update_strain_view(p, field, "omega")
        assert p._state["colorbar_label"] == "ω (°)"
        assert p._state["show_colorbar"] is True


class TestNavigationExtent:
    def test_a_calibrated_scan_gives_its_coordinates(self):
        x, y, units = navigation_extent(_calibrated_scan(), (4, 5))
        assert units == "nm"
        assert x == pytest.approx(np.arange(5) * 2.5)
        assert y == pytest.approx(np.arange(4) * 2.5)

    def test_a_5d_scan_hands_over_the_spatial_pair_not_the_clock(self):
        import hyperspy.api as hs
        sig = hs.signals.Signal2D(np.zeros((3, 4, 5, 2, 2), np.float32))
        x, y, t = sig.axes_manager.navigation_axes
        t.scale, t.units = 0.5, "s"
        for axis in (x, y):
            axis.scale, axis.units = 2.5, "nm"
        assert navigation_extent(sig, (4, 5))[2] == "nm"

    def test_a_mismatched_grid_or_pixel_units_gives_nothing(self):
        assert navigation_extent(_calibrated_scan(), (3, 3)) is None
        assert navigation_extent(_calibrated_scan(units="px"), (4, 5)) is None
        assert navigation_extent(None, (4, 5)) is None


class TestCommittedTree:
    def _commit(self, session, scan):
        import anyplotlib as apl
        from spyde.actions.strain_action import StrainController
        fig, ax = apl.subplots()
        p = ax.imshow(np.zeros((4, 5), "f4"))
        tree = type("T", (), {"root": scan})()
        ctrl = StrainController(None, p, window_id=4242, ref_yx=(0, 0),
                                session=session, src_tree=tree)
        ctrl.field = _field()
        return ctrl.commit()

    def test_the_tree_holds_percent_and_says_so(self, window):
        session = window["window"]
        tree = self._commit(session, _calibrated_scan())

        assert np.allclose(tree.root.data, 1.2)
        assert tree.root.metadata.get_item("Signal.quantity") == "εxx (%)"
        by_title = {str(s.metadata.General.title): s for s in tree.signals()}
        omega = by_title["Strain ω"]
        assert np.allclose(omega.data, 1.0)
        assert omega.metadata.get_item("Signal.quantity") == "ω (°)"

    def test_every_node_is_on_the_scan_scale(self, window):
        session = window["window"]
        tree = self._commit(session, _calibrated_scan(scale=2.5))
        for sig in tree.signals():
            for axis in sig.axes_manager.signal_axes:
                assert str(axis.units) == "nm"
                assert axis.scale == pytest.approx(2.5)

    def test_the_chip_views_draw_the_same_scale_and_colorbar(self, window):
        # The chips are figures of their own; a label that only the primary
        # plot carries would vanish the moment the user clicks εyy.
        from de_shell.actions.figure_registry import _FIGS
        from spyde.actions.views import build_tiled_figure
        session = window["window"]
        tree = self._commit(session, _calibrated_scan())
        wid = tree.signal_plots[0].window_id

        labels = {}
        for fig in _FIGS.get(int(wid), []):
            for p in fig._plots_map.values():
                st = p._state
                if st.get("show_colorbar"):
                    labels[st["colorbar_label"]] = st["units"]
        for component in ("eyy", "exy", "omega"):
            assert labels.get(strain_quantity(component)) == "nm"

        fig, _fig_id, _html, order = build_tiled_figure(
            wid, [STRAIN_TITLES["exx"], STRAIN_TITLES["omega"]])
        tiled = [p._state for p in fig._plots_map.values()]
        assert [s["colorbar_label"] for s in tiled] == ["εxx (%)", "ω (°)"]
        assert all(s["units"] == "nm" and s["has_axes"] for s in tiled)


class TestOrientationStrainCommit:
    def test_the_vom_strain_window_is_percent_and_calibrated(self, window):
        from spyde.actions.vector_orientation_om import _build_strain_window
        session = window["window"]
        scan = _calibrated_scan()
        scan.metadata.General.title = "Scan"
        strain = np.zeros((4, 5, 3), np.float32)
        strain[..., 0] = 0.02
        result = type("R", (), {"strain": strain})()

        _build_strain_window(session, scan, result)

        tree = session.signal_trees[-1]
        assert np.allclose(tree.root.data, 2.0)
        assert tree.root.metadata.get_item("Signal.quantity") == "εxx (%)"
        assert str(tree.root.axes_manager.signal_axes[0].units) == "nm"


class TestDpcColorbar:
    def test_a_scalar_view_is_labelled_and_the_rgb_view_is_not(self):
        import anyplotlib as apl
        from spyde.actions.dpc_display import _label_colorbar
        fig, ax = apl.subplots()
        p = ax.imshow(np.zeros((4, 5), "f4"))
        result = type("R", (), {"mode": "magnetic", "units": "mrad"})()

        _label_colorbar(p, result, "fx")
        assert p._state["show_colorbar"] is True
        assert p._state["colorbar_label"] == "Bx (mrad)"

        _label_colorbar(p, result, "rgb")
        assert p._state["show_colorbar"] is False
