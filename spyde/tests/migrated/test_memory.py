"""Memory regression: the per-plot shared buffer must be sized to the displayed
frame, not a fixed 256 MB (8192x8192x4) max-image buffer that spiked RAM and
slowed subwindow spawn."""
import numpy as np
import hyperspy.api as hs

from spyde.drawing.plots.plot import Plot


def _plot_with_signal(signal):
    p = Plot.__new__(Plot)
    p._shared_memory = None

    class _PS:
        current_signal = signal
    p.plot_state = _PS()
    return p


class TestSharedBufferSize:
    def test_256_dp_buffer_is_about_1mb(self):
        sig = hs.signals.Signal2D(np.zeros((256, 256), dtype=np.float32))
        p = _plot_with_signal(sig)
        mb = p._buffer_nbytes() / 1024 / 1024
        assert 0.5 < mb < 4, f"buffer {mb:.2f} MB (should be ~1 MB, was 256 MB)"

    def test_buffer_never_256mb(self):
        # Even a large 1024x1024 frame must be far below the old 256 MB.
        sig = hs.signals.Signal2D(np.zeros((1024, 1024), dtype=np.float32))
        p = _plot_with_signal(sig)
        mb = p._buffer_nbytes() / 1024 / 1024
        assert mb < 32, f"buffer {mb:.2f} MB too large"

    def test_fallback_is_modest(self):
        p = Plot.__new__(Plot)
        p._shared_memory = None
        p.plot_state = None
        mb = p._buffer_nbytes() / 1024 / 1024
        assert mb <= 8, f"fallback buffer {mb:.2f} MB too large"
