"""GUI-wiring test for the neural (SpotUNet) find-vectors method.

Exercises the full dispatch + chunk + CSR-packing path with the neural detector,
forcing the CPU branch (``torch_gpu_device`` → None) so it's deterministic and
avoids the torch-CUDA-under-pytest segfault on Windows (see CLAUDE.md). Confirms:

  - the wizard default method is now ``neural`` and the model registry resolves
    the bundled model,
  - ``fv_models`` emits the available-models payload for the Model dropdown,
  - ``fv_run`` with ``method="neural"`` produces a vectors window with
    ``tree.diffraction_vectors`` attached.

Detection *quality* is not asserted (the network's accuracy is benchmarked
separately on real data) — only that the wiring produces a valid vectors image.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _diffraction_4d(nav=(4, 5), sig=(32, 32), scale=0.1):
    """A small 4D-STEM stack with a few bright Gaussian disks per pattern (so the
    neural detector reliably fires)."""
    from scipy.ndimage import gaussian_filter
    data = np.zeros(nav + sig, dtype=np.float32)
    base = np.zeros(sig, np.float32)
    for (cy, cx) in [(16, 16), (8, 22), (24, 9)]:
        base[cy, cx] = 400.0
    base = gaussian_filter(base, 2.0)
    for idx in np.ndindex(*nav):
        data[idx] = base
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = 0.0
        ax.units = "1/nm"
    return s


class TestFindVectorsNeural:
    def test_neural_default_and_models_payload(self):
        from spyde.actions.find_vectors_action import _coerce, fv_models
        from spyde.models import default_model_id

        # Neural is the default method out of the box.
        assert _coerce({})["method"] == "neural"
        assert abs(_coerce({})["threshold"] - 0.3) < 1e-9

        # fv_models emits the registry's available models (for the Model dropdown).
        import spyde.actions.find_vectors_action as fva
        captured = []
        orig = fva.emit
        fva.emit = lambda msg: captured.append(msg)
        try:
            class _P:
                window_id = 3
            fv_models(None, _P(), {"window_id": 3})
        finally:
            fva.emit = orig
        assert captured and captured[0]["type"] == "fv_models"
        assert captured[0]["default"] == default_model_id()
        assert any(m["id"] == default_model_id() for m in captured[0]["models"])

    def test_neural_run_cpu(self, monkeypatch):
        # Force the CPU branch — deterministic, and no torch-CUDA under pytest.
        import spyde.actions.find_vectors_torch as fvt
        monkeypatch.setattr(fvt, "torch_gpu_device", lambda: None)

        from spyde.backend.session import Session
        from spyde.actions.find_vectors_action import fv_run

        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_diffraction_4d())
            time.sleep(0.4)
            src = _signal_plot(session)
            assert src is not None
            before = len(session.signal_trees)

            fv_run(session, src, {
                "method": "neural", "sigma": 1.0, "threshold": 0.3,
                "min_distance": 4, "subpixel": True, "model_id": "",
            })
            assert _wait(lambda: len(session.signal_trees) == before + 1), \
                "neural vectors window never opened"
            vtree = session.signal_trees[-1]
            assert _wait(lambda: getattr(vtree, "diffraction_vectors", None) is not None), \
                "diffraction_vectors never attached"
            # A valid (possibly empty) count map of the right nav shape.
            cm = vtree.diffraction_vectors.count_map()
            assert cm.shape == (4, 5)
        finally:
            session.shutdown()
