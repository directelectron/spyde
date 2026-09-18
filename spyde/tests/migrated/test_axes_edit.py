"""
Axes table edits write back to the dataset AND reflect in the plot immediately.

`set_axis` mutates the real axes_manager, re-pushes every plot in the tree (so the
extent / scale bar update at once), and re-emits the axes table + metadata. The
dataset shape now lives in the Metadata panel (`Dataset` section), not the table.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, close_session, make_session


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


class TestAxesEdit:
    def test_set_axis_writes_back_and_re_emits(self):
        import spyde.backend.session as sess_mod
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((4, 5, 24, 24), np.float32))
            s.set_signal_type("electron_diffraction")
            session._add_signal(s)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            captured = []
            # _set_axis lives in the AxesEditorMixin (_session_axes) and emits via
            # ipc.emit; patch that single channel to capture its messages.
            import de_shell.ipc as ipc_mod
            orig = ipc_mod.emit
            ipc_mod.emit = lambda m: captured.append(m)
            try:
                # Edit signal-axis 3 (kx) scale → 0.25.
                session._set_axis(plot, {"index": 3, "field": "scale", "value": "0.25"})
            finally:
                ipc_mod.emit = orig

            # Written to the real axes_manager (immediate, in-memory).
            assert abs(float(tree.root.axes_manager._axes[3].scale) - 0.25) < 1e-9
            # Re-emitted the axes table with the new value …
            axes_msgs = [m for m in captured if m.get("type") == "axes_info"]
            assert axes_msgs
            scales = [a["scale"] for a in axes_msgs[-1]["axes"]]
            assert any(abs(sc - 0.25) < 1e-9 for sc in scales)
            # … and re-emitted metadata (Dataset shape lives there now).
            md = [m for m in captured if m.get("type") == "metadata"]
            assert md and "Dataset" in md[-1]["metadata"]
            assert "Shape" in md[-1]["metadata"]["Dataset"]
        finally:
            close_session(session)

    def test_scale_edit_preserves_origin_pixel(self):
        """Changing scale rescales the offset so the (0,0) data point stays on
        the SAME pixel — recalibrating pixel size must not move the marked
        origin / crosshair centre."""
        import hyperspy.api as hs
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((4, 5, 20, 20), np.float32))
            s.set_signal_type("electron_diffraction")
            # _axes order is nav-first; signal kx is index 3 (matches the test
            # above). Configure that axis so the edit targets it.
            s.axes_manager._axes[3].scale = 0.1
            s.axes_manager._axes[3].offset = -1.0   # origin pixel = 1.0/0.1 = 10
            session._add_signal(s)
            _settle(session)
            plot = _signal_plot(session)
            axx = plot.signal_tree.root.axes_manager._axes[3]
            origin_px = -float(axx.offset) / float(axx.scale)
            assert abs(origin_px - 10.0) < 1e-9

            # double the scale → offset should double so origin pixel stays 10
            session._set_axis(plot, {"index": 3, "field": "scale", "value": "0.2"})
            assert abs(float(axx.scale) - 0.2) < 1e-9
            assert abs(float(axx.offset) - (-2.0)) < 1e-9
            new_origin_px = -float(axx.offset) / float(axx.scale)
            assert abs(new_origin_px - origin_px) < 1e-9
        finally:
            close_session(session)


class TestDetectorUnitsToggle:
    """Re-expressing the detector is a DISPLAY choice, not a recalibration.

    ``set_reciprocal_units`` rescales the axes so the calibration and its label
    never disagree — and because the crystallographic paths ask what a pixel is
    worth in Å⁻¹ rather than reading the axis scale, the same scan indexes
    identically whichever unit is showing. Both halves are asserted: the axes
    really move, and ``reciprocal_radius`` really does not.
    """

    @staticmethod
    def _diffraction_session(scale, units):
        session = make_session()
        s = hs.signals.Signal2D(np.zeros((4, 5, 24, 24), np.float32))
        for ax in s.axes_manager.signal_axes:
            ax.scale, ax.offset, ax.units = scale, -scale * 12, units
        s.set_signal_type("electron_diffraction")
        # set_signal_type stamps units="px"; restate the real label.
        for ax in s.axes_manager.signal_axes:
            ax.units = units
        s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
        session._add_signal(s)
        _settle(session)
        return session

    def _capture(self, session, plot, units):
        import de_shell.ipc as ipc_mod
        captured = []
        orig = ipc_mod.emit
        ipc_mod.emit = lambda m: captured.append(m)
        try:
            session._set_reciprocal_units(plot, {"units": units})
        finally:
            ipc_mod.emit = orig
        return captured

    def test_toggle_rescales_and_leaves_the_indexing_alone(self):
        from spyde.actions._common import reciprocal_radius

        # Opened in nm⁻¹ — _add_signal re-expresses it in Å⁻¹ on the way in.
        session = self._diffraction_session(0.1, "nm^-1")
        try:
            plot = _signal_plot(session)
            tree = plot.signal_tree
            axis = tree.root.axes_manager.signal_axes[0]
            assert str(axis.units) == "A^-1"
            assert abs(float(axis.scale) - 0.01) < 1e-12
            before = reciprocal_radius(tree.root)

            for unit, expected in (("nm^-1", 0.1), ("mrad", None),
                                   ("px", 1.0), ("A^-1", 0.01)):
                self._capture(session, plot, unit)
                axis = tree.root.axes_manager.signal_axes[0]
                assert str(axis.units) == unit, unit
                if expected is not None:
                    assert abs(float(axis.scale) - expected) < 1e-9, unit
                # The whole point: the number the template library is sized by
                # does not move when the display unit does.
                assert abs(reciprocal_radius(tree.root) - before) < 1e-9, unit
        finally:
            close_session(session)

    def test_the_dock_is_told_what_is_on_offer(self):
        session = self._diffraction_session(0.01, "A^-1")
        try:
            plot = _signal_plot(session)
            captured = self._capture(session, plot, "nm^-1")
            axes_msgs = [m for m in captured if m.get("type") == "axes_info"]
            assert axes_msgs
            toggle = axes_msgs[-1]["units_toggle"]
            assert toggle["current"] == "nm^-1"
            assert toggle["order"] == ["px", "mrad", "nm^-1", "A^-1"]
            # Everything reachable: this signal has a scale AND a beam energy.
            assert set(toggle["reasons"].values()) == {""}
        finally:
            close_session(session)

    def test_milliradian_says_what_it_needs_instead_of_failing_quietly(self):
        session = self._diffraction_session(0.01, "A^-1")
        try:
            plot = _signal_plot(session)
            tree = plot.signal_tree
            tree.root.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 0)
            captured = self._capture(session, plot, "mrad")
            assert str(tree.root.axes_manager.signal_axes[0].units) == "A^-1"
            errors = [m for m in captured if m.get("type") == "error"]
            assert errors and "beam energy" in str(errors[-1]).lower()
        finally:
            close_session(session)

    def test_a_scan_space_result_map_is_offered_no_detector_units(self):
        from spyde.metadata_extract import build_units_toggle

        session = make_session()
        try:
            # A strain map: 2 signal axes, calibrated in nm. Not a detector.
            s = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
            for ax in s.axes_manager.signal_axes:
                ax.scale, ax.units = 0.9, "nm"
            session._add_signal(s)
            _settle(session)
            tree = session.signal_trees[-1]
            assert build_units_toggle(tree) is None
        finally:
            close_session(session)

    def test_typing_a_beam_energy_re_offers_milliradian(self):
        """Availability depends on METADATA, so a metadata edit has to re-offer.

        Without this the dock keeps the reasons it was last sent: mrad stays
        greyed out on a signal that can now reach it, and the only way to
        discover otherwise is to click something else and come back.
        """
        session = self._diffraction_session(0.01, "A^-1")
        try:
            plot = _signal_plot(session)
            tree = plot.signal_tree
            tree.root.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 0)

            import de_shell.ipc as ipc_mod
            captured = []
            orig = ipc_mod.emit
            ipc_mod.emit = lambda m: captured.append(m)
            try:
                session._set_metadata(plot, {"group": "Instrument Metadata",
                                             "prop": "Acc. Volt.", "value": "200"})
            finally:
                ipc_mod.emit = orig

            axes_msgs = [m for m in captured if m.get("type") == "axes_info"]
            assert axes_msgs, "a metadata edit did not re-emit the axes payload"
            assert axes_msgs[-1]["units_toggle"]["reasons"]["mrad"] == ""
        finally:
            close_session(session)
