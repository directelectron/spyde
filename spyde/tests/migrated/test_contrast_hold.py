"""
Contrast behaviour:

* A NAVIGATED frame (diffraction pattern under the navigator) HOLDS contrast as
  you drag — adjacent frames share a range, so re-auto-leveling each one flashed
  ("lazy navigator flashes / inconsistent").
* An OUTPUT plot (virtual image / FFT / line profile) re-auto-levels on every
  recompute — each is a brand-new image whose range can differ a lot; holding
  made the VI look wrong as the detector ROI moved.
"""
from __future__ import annotations

import numpy as np

from spyde.drawing.plots.plot import Plot


class _AM:
    def __init__(self, nav_dim): self.navigation_dimension = nav_dim
class _Sig:
    def __init__(self, nav_dim): self.axes_manager = _AM(nav_dim)
class _PS:
    def __init__(self, nav_dim): self.current_signal = _Sig(nav_dim)


def _bare_plot(nav_dim: int):
    p = Plot.__new__(Plot)
    p._plot2d = None
    p._plot1d = None
    p._fig = None
    p.fig_id = None
    p.window_id = None
    p.is_navigator = False
    p.needs_auto_level = True
    p._last_levels = None
    p.plot_state = _PS(nav_dim)        # nav_dim>0 → navigated (DP); 0 → output
    clims: list[tuple[float, float]] = []

    class _FakeP2D:
        def set_data(self, data, **k): pass
        def set_clim(self, lo, hi): clims.append((float(lo), float(hi)))

    p._ensure_figure = lambda dims: setattr(p, "_plot2d", _FakeP2D())
    return p, clims


_BRIGHT = np.zeros((16, 16), np.float32); _BRIGHT[4:12, 4:12] = 100.0
_DARK = np.zeros((16, 16), np.float32);  _DARK[6:10, 6:10] = 5.0


class TestContrastHold:
    def test_navigated_frames_hold_contrast(self):
        p, clims = _bare_plot(nav_dim=2)        # diffraction pattern
        Plot._set_array(p, _BRIGHT)
        first = clims[-1]
        assert first[1] > first[0]
        # A subsequent navigator frame must REUSE the contrast (no flash).
        Plot._set_array(p, _DARK)
        assert clims[-1] == first, "DP contrast rescaled on a navigator frame (flash)"

    def test_output_plot_relevels_each_recompute(self):
        # Virtual image / FFT output (nav_dim 0) must re-level so it isn't shown
        # with stale contrast as the detector ROI changes the image.
        p, clims = _bare_plot(nav_dim=0)
        Plot._set_array(p, _BRIGHT)
        first = clims[-1]
        Plot._set_array(p, _DARK)
        assert clims[-1] != first, "output plot held stale contrast (VI looked wrong)"

    def test_explicit_autolevel_recomputes(self):
        p, clims = _bare_plot(nav_dim=2)
        Plot._set_array(p, _BRIGHT)
        first = clims[-1]
        p.needs_auto_level = True            # new data / Integrate toggle
        Plot._set_array(p, _DARK)
        assert clims[-1] != first

    def test_user_clim_is_held_across_navigated_frames(self):
        p, clims = _bare_plot(nav_dim=2)
        Plot._set_array(p, _BRIGHT)
        Plot.set_clim(p, 10.0, 50.0)
        assert p._last_levels == (10.0, 50.0)
        Plot._set_array(p, _BRIGHT)
        assert clims[-1] == (10.0, 50.0)
