"""
Instrument-metadata panel edits write back to the dataset AND persist across
save/reload — mirrors test_axes_edit.py's `set_axis` contract for `set_metadata`.

`set_metadata` resolves the writable `key`/`type` for a `(group, prop)` cell
from METADATA_WIDGET_CONFIG (the same table `build_metadata_dict` reads),
coerces the typed-in string per the YAML `type` (float/int -> number, else
stays a string), writes it onto `tree.root.metadata` via `set_item` (real
hyperspy metadata, so it saves like any other field), and re-emits the
metadata dict so the dock reflects the committed value.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, close_session, make_session


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


class TestSetMetadataAction:
    def test_set_metadata_float_writes_back_and_re_emits(self):
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
            import de_shell.ipc as ipc_mod
            orig = ipc_mod.emit
            ipc_mod.emit = lambda m: captured.append(m)
            try:
                session._set_metadata(
                    plot, {"group": "Instrument Metadata", "prop": "Mag", "value": "12000.5"}
                )
            finally:
                ipc_mod.emit = orig

            # Written to the real metadata tree (immediate, in-memory) as a float.
            written = tree.root.metadata.get_item("Acquisition_instrument.TEM.magnification")
            assert isinstance(written, float)
            assert abs(written - 12000.5) < 1e-9

            # Re-emitted the resolved metadata dict with the new value.
            md_msgs = [m for m in captured if m.get("type") == "metadata"]
            assert md_msgs
            assert md_msgs[-1]["metadata"]["Instrument Metadata"]["Mag"] == "12000.5 x"
        finally:
            close_session(session)

    def test_set_metadata_string_field(self):
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
            session._add_signal(s)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            session._set_metadata(
                plot, {"group": "Root Experiment Details", "prop": "Name",
                       "value": "My Experiment"}
            )
            assert tree.root.metadata.get_item("General.title") == "My Experiment"
        finally:
            close_session(session)

    def test_set_metadata_invalid_number_ignored(self):
        """Unparsable numeric input (mid-typing junk) must not write NaN/0 —
        it's silently ignored so the field reverts on the next re-render,
        matching _set_axis's non-numeric-input guard."""
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
            s.metadata.set_item("Acquisition_instrument.TEM.magnification", 5000.0)
            session._add_signal(s)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            session._set_metadata(
                plot, {"group": "Instrument Metadata", "prop": "Mag", "value": "not-a-number"}
            )
            # Untouched — the bad input was rejected, not coerced to 0/NaN.
            assert tree.root.metadata.get_item(
                "Acquisition_instrument.TEM.magnification") == 5000.0
        finally:
            close_session(session)

    def test_units_suffixed_display_string_rejected(self):
        """Regression lock for the units-in-pre-fill bug: the panel DISPLAYS
        "12000.5 x" (units baked in). If the editor ever pre-filled that
        display string again, a natural edit would commit "13000 x" — which
        must NOT write (float() fails → silent revert), so the raw/display
        split in the renderer is the only way a numeric edit can land."""
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
            s.metadata.set_item("Acquisition_instrument.TEM.magnification", 5000.0)
            session._add_signal(s)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            session._set_metadata(
                plot, {"group": "Instrument Metadata", "prop": "Mag", "value": "13000 x"}
            )
            assert tree.root.metadata.get_item(
                "Acquisition_instrument.TEM.magnification") == 5000.0
            # …while the raw (unit-free) value the UI actually pre-fills with
            # commits fine — the shape the renderer now sends.
            session._set_metadata(
                plot, {"group": "Instrument Metadata", "prop": "Mag", "value": "13000"}
            )
            assert tree.root.metadata.get_item(
                "Acquisition_instrument.TEM.magnification") == 13000.0
        finally:
            close_session(session)

    def test_non_finite_numbers_rejected(self):
        """float() happily parses 'nan'/'inf' (and '1e400' overflows to inf)
        without raising — without an explicit isfinite guard they'd write and
        even round-trip to .zspy as garbage calibration."""
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
            s.metadata.set_item("Acquisition_instrument.TEM.magnification", 5000.0)
            session._add_signal(s)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            for bad in ("nan", "NaN", "inf", "-inf", "1e400"):
                session._set_metadata(
                    plot, {"group": "Instrument Metadata", "prop": "Mag", "value": bad}
                )
                assert tree.root.metadata.get_item(
                    "Acquisition_instrument.TEM.magnification") == 5000.0, bad
        finally:
            close_session(session)

    def test_set_metadata_readonly_derived_field_ignored(self):
        """Dtype/Dim. are `attr`/`function` derived config entries with no
        writable `key` — editing them must be a no-op: no raise, and no
        metadata mutation at all (no bogus key created anywhere)."""
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
            session._add_signal(s)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            before = tree.root.metadata.as_dictionary()
            session._set_metadata(
                plot, {"group": "Root Experiment Details", "prop": "Dtype", "value": "int64"}
            )
            # The whole metadata tree is bit-identical — nothing was written.
            assert tree.root.metadata.as_dictionary() == before
        finally:
            close_session(session)

    def test_set_metadata_unknown_cell_ignored(self):
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
            session._add_signal(s)
            _settle(session)
            plot = _signal_plot(session)
            session._set_metadata(
                plot, {"group": "Nonexistent Group", "prop": "Nope", "value": "1"}
            )
        finally:
            close_session(session)

    def test_set_metadata_no_plot_is_noop(self, window):
        session = window["window"]
        session._set_metadata(None, {"group": "Instrument Metadata", "prop": "Mag",
                                      "value": "1"})  # must not raise


class TestMetadataEditPersistence:
    """PERSISTENCE IS PART OF THE TASK: an edited value must round-trip through
    a real save-to-disk + fresh reload, not just live in the in-memory tree."""

    def test_edit_survives_zspy_save_reload(self, stem_4d_dataset, tmp_path):
        session = stem_4d_dataset["window"]
        plot = _signal_plot(session)
        tree = plot.signal_tree

        # Edit a float instrument field and a string field.
        session._set_metadata(
            plot, {"group": "Instrument Metadata", "prop": "Acc. Volt.", "value": "200.0"}
        )
        session._set_metadata(
            plot, {"group": "Root Experiment Details", "prop": "Name",
                   "value": "Round Trip Sample"}
        )
        assert tree.root.metadata.get_item(
            "Acquisition_instrument.TEM.beam_energy") == 200.0
        assert tree.root.metadata.get_item("General.title") == "Round Trip Sample"

        # Save the edited signal to a real .zspy Zarr store (tmp_path) via the
        # same synchronous writer thread body the app uses (_save_signal_thread).
        out = str(tmp_path / "edited.zspy")
        session._save_signal_thread(tree.root, out, "edited.zspy")

        # Reload FRESH — a brand-new hs.load, not the in-memory tree — proving
        # the edit was actually persisted to disk, not just held in RAM.
        reloaded = hs.load(out)
        assert reloaded.metadata.get_item(
            "Acquisition_instrument.TEM.beam_energy") == 200.0
        assert reloaded.metadata.get_item("General.title") == "Round Trip Sample"

        # And build_metadata_dict on a tree wrapping the reloaded signal shows
        # the persisted value too (what the sidebar would render after reload).
        from spyde.metadata_extract import build_metadata_dict

        class _Tree:
            def __init__(self, root):
                self.root = root
                self.signal_plots = []

            def get_nested_attr(self, attr_path):
                obj = self
                for attr in (p for p in attr_path.split(".") if p):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        return None
                return obj

        md = build_metadata_dict(_Tree(reloaded))
        assert md["Instrument Metadata"]["Acc. Volt."] == "200.0 kV"
        assert md["Root Experiment Details"]["Name"] == "Round Trip Sample"

    def test_metadata_reemit_reaches_captured_messages(self, stem_4d_dataset):
        """The metadata re-emit after an edit is on the SAME `ipc.emit` channel
        the `captured_messages` fixture monkeypatches — the renderer-facing
        assertion, complementing the direct-patch capture used above."""
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        plot = _signal_plot(session)

        before = len(messages)
        session._set_metadata(
            plot, {"group": "Instrument Metadata", "prop": "Cam. Len.", "value": "150.0"}
        )
        new_msgs = messages[before:]
        md_msgs = [m for m in new_msgs if isinstance(m, dict) and m.get("type") == "metadata"]
        assert md_msgs, "expected a metadata re-emit after set_metadata"
        assert md_msgs[-1]["metadata"]["Instrument Metadata"]["Cam. Len."] == "150.0 mm"

    def test_emit_carries_raw_editable_map(self, stem_4d_dataset):
        """The `editable` field of the metadata emit is {group: {prop: raw}} —
        the UNIT-FREE value the renderer pre-fills the inline editor with.
        This is what makes numeric edits work from the real UI: the `metadata`
        display strings have units baked in ("150.0 mm") and would fail the
        backend float() if committed back."""
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        plot = _signal_plot(session)

        before = len(messages)
        session._set_metadata(
            plot, {"group": "Instrument Metadata", "prop": "Cam. Len.", "value": "150.0"}
        )
        md_msgs = [m for m in messages[before:]
                   if isinstance(m, dict) and m.get("type") == "metadata"]
        assert md_msgs
        ed = md_msgs[-1]["editable"]

        # Raw value is unit-free (display shows "150.0 mm", raw is "150.0"),
        # so pre-fill → tweak → commit round-trips through float().
        assert ed["Instrument Metadata"]["Cam. Len."] == "150.0"
        float(ed["Instrument Metadata"]["Cam. Len."])   # parses clean
        # An unset editable cell pre-fills empty (not the "--" placeholder).
        assert ed["Instrument Metadata"]["Mag"] == ""
        # Derived read-only cells and the synthetic Dataset group are absent.
        assert "Dtype" not in ed["Root Experiment Details"]
        assert "Dim." not in ed["Root Experiment Details"]
        assert "Dataset" not in ed
        # And a string field carries its raw text verbatim.
        before2 = len(messages)
        session._set_metadata(
            plot, {"group": "Root Experiment Details", "prop": "Name", "value": "RawCheck"}
        )
        md2 = [m for m in messages[before2:]
               if isinstance(m, dict) and m.get("type") == "metadata"]
        assert md2[-1]["editable"]["Root Experiment Details"]["Name"] == "RawCheck"
