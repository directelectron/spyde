"""
Signal-type gating of toolbar actions + the Signal Type dock panel.

Diffraction-only actions (Virtual Imaging, Orientation Mapping) must be
hidden for plain Signal2D data and appear after the signal is retyped to a
Diffraction2D subclass via the Plot Control dock's Signal Type panel.
"""

DIFFRACTION_ONLY_ACTIONS = ("Virtual Imaging", "Orientation Mapping")


def _bottom_action_names(plot):
    toolbar = plot.plot_state.toolbar_bottom
    return [a.text() for a in toolbar.actions()]


class TestSignalTypeGating:
    def test_hidden_for_plain_signal2d(self, tem_2d_dataset):
        window = tem_2d_dataset["subwindows"][0]
        names = _bottom_action_names(window.current_plot_item)
        for action in DIFFRACTION_ONLY_ACTIONS:
            assert action not in names

    def test_shown_for_4d_stem_diffraction(self, stem_4d_dataset):
        # the 4D STEM signal plot is typed electron_diffraction at creation
        windows = stem_4d_dataset["subwindows"]
        signal_plot = next(
            w.current_plot_item
            for w in windows
            if not w.current_plot_item.is_navigator
        )
        names = _bottom_action_names(signal_plot)
        for action in DIFFRACTION_ONLY_ACTIONS:
            assert action in names


class TestSignalTypePanel:
    def test_panel_populates_on_activation(self, tem_2d_dataset):
        win = tem_2d_dataset["window"]
        window = tem_2d_dataset["subwindows"][0]
        dm = win.dock_manager
        dm.on_active_plot_changed(window)

        assert dm._signal_type_group.isEnabled()
        assert "Class:" in dm.signal_class_label.text()
        items = [
            dm.signal_type_combo.itemText(i)
            for i in range(dm.signal_type_combo.count())
        ]
        assert "diffraction" in items
        assert "(generic)" in items
        assert dm.signal_type_combo.currentText() == "(generic)"

    def test_set_type_retypes_signal_and_rebuilds_toolbar(self, tem_2d_dataset):
        from pyxem.signals import Diffraction2D

        win = tem_2d_dataset["window"]
        window = tem_2d_dataset["subwindows"][0]
        plot = window.current_plot_item
        dm = win.dock_manager
        dm.on_active_plot_changed(window)

        names_before = _bottom_action_names(plot)
        for action in DIFFRACTION_ONLY_ACTIONS:
            assert action not in names_before

        dm.signal_type_combo.setCurrentText("diffraction")
        dm._on_set_signal_type()

        signal = plot.plot_state.current_signal
        assert isinstance(signal, Diffraction2D)
        # toolbars were rebuilt with the diffraction-gated actions present
        names_after = _bottom_action_names(plot)
        for action in DIFFRACTION_ONLY_ACTIONS:
            assert action in names_after
        # panel reflects the new class
        assert "Diffraction2D" in dm.signal_class_label.text()

        # retype back to generic: gated actions disappear again
        dm.signal_type_combo.setCurrentText("(generic)")
        dm._on_set_signal_type()
        names_reverted = _bottom_action_names(plot)
        for action in DIFFRACTION_ONLY_ACTIONS:
            assert action not in names_reverted
