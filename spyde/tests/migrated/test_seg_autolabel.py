"""seg_autolabel + the REAL seg_train must produce a usable head."""
import time
import numpy as np
from spyde.actions import particles_action as pa


def _wait(pred, timeout=120.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


class TestAutolabelDoor:
    def test_autolabel_then_train_gives_a_trained_head(self, window):
        session = window["window"]
        session._load_test_data_particles({"frames": 6})
        plot = None
        assert _wait(lambda: any(
            not p.is_navigator and p.plot_state is not None
            for p in session._plots))
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        tree = plot.signal_tree
        pa.seg_open(session, plot, {})
        assert _wait(lambda: getattr(tree, "_seg_wizard", None) is not None)
        wiz = tree._seg_wizard

        pa.seg_autolabel(session, plot, {})
        counts = wiz.label_store().counts()
        assert counts.get(0, 0) > 0, f"no particle pixels painted: {counts}"
        assert counts.get(1, 0) > 0, f"no film pixels painted: {counts}"

        pa.seg_train(session, plot, {})
        assert _wait(lambda: wiz.classifier is not None
                     and wiz.classifier.is_trained), "training never finished"
        assert _wait(lambda: wiz.preview is not None
                     and wiz.preview["count"] > 0, timeout=60), \
            "the trained head found nothing on the frame"

    def test_it_refuses_a_signal_with_no_stamped_truth(self, tem_2d_dataset):
        w = tem_2d_dataset
        session = w["window"]
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        pa.seg_open(session, plot, {})
        assert _wait(lambda: getattr(plot.signal_tree, "_seg_wizard", None))
        pa.seg_autolabel(session, plot, {})
        assert any("test door" in str(m.get("text", ""))
                   for m in w["messages"] if isinstance(m, dict)), \
            "autolabel silently accepted a non-synthetic signal"
