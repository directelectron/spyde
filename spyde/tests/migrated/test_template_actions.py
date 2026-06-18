"""Toolbar actions on the host-agnostic template must run end-to-end."""
import time


def _signal_plot(session):
    for p in session._plots:
        if not p.is_navigator and p.plot_state is not None:
            return p
    return None


def _run(session, messages, name, params):
    messages.clear()
    plot = _signal_plot(session)
    assert plot is not None, "no signal plot"
    session._dispatch_toolbar_action(plot, name, params)
    time.sleep(0.5)
    opened = [m for m in messages if m.get("type") == "window_opened"]
    errors = [m.get("text") for m in messages if m.get("type") == "error"]
    return opened, errors


class TestTemplateActions:
    def test_virtual_imaging_opens_output_window(self, stem_4d_dataset):
        # Virtual Imaging is now a submenu; the VI is created by "add_virtual_image".
        opened, errors = _run(
            stem_4d_dataset["window"], stem_4d_dataset["messages"],
            "add_virtual_image", {"type": "disk", "calculation": "mean"},
        )
        assert not errors, errors
        assert len(opened) == 1

    def test_fft_opens_output_window(self, stem_4d_dataset):
        opened, errors = _run(
            stem_4d_dataset["window"], stem_4d_dataset["messages"], "FFT", {},
        )
        assert not errors, errors
        assert len(opened) == 1

    def test_line_profile_opens_output_window(self, stem_4d_dataset):
        opened, errors = _run(
            stem_4d_dataset["window"], stem_4d_dataset["messages"], "Line Profile", {},
        )
        assert not errors, errors
        assert len(opened) == 1

    def test_unknown_action_reports_error(self, stem_4d_dataset):
        opened, errors = _run(
            stem_4d_dataset["window"], stem_4d_dataset["messages"], "Nope", {},
        )
        assert errors and "Unknown toolbar action" in errors[0]
