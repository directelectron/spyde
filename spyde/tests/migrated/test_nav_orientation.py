"""
Navigator → diffraction-pattern coordinate mapping (Qt→anyplotlib regression).

pyqtgraph displayed the navigator image TRANSPOSED, so the Qt index math
``data[(cx, cy)]`` happened to be correct. anyplotlib's imshow displays it
un-transposed (axis 0 = rows = iy, axis 1 = cols = ix), so the selector's
widget coords (cx = column = ix, cy = row = iy) must be swapped to ``data[iy, ix]``.

A NON-SQUARE scan with each pattern positionally encoded makes the swap
unambiguous: a wrong mapping returns the wrong frame (or clamps on the
mismatched axis).
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs

from spyde.drawing.update_functions import update_from_navigation_selection


def _encoded_4d(nav=(3, 5), sig=(4, 4)):
    """data[iy, ix, 0, 0] == 100*iy + ix — so the returned DP names its own
    scan position. nav is non-square (3 != 5) to expose any axis swap."""
    data = np.zeros(nav + sig, dtype=np.float32)
    for iy in range(nav[0]):
        for ix in range(nav[1]):
            data[iy, ix, 0, 0] = 100 * iy + ix
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    return s


class TestNavOrientation:
    def test_crosshair_selects_data_iy_ix(self):
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_encoded_4d(), source_path=None)
            time.sleep(0.4)
            tree = session.signal_trees[0]
            sel = tree.navigator_plot_manager.all_navigation_selectors[0]
            child = next(p for p in session._plots
                         if not p.is_navigator and p.plot_state is not None)

            # Crosshair at column ix=4, row iy=1 → expect DP at data[1, 4] = 104.
            sel._widget._data["cx"] = 4   # column = ix
            sel._widget._data["cy"] = 1   # row = iy
            idx = sel.get_selected_indices()
            img = np.asarray(update_from_navigation_selection(sel, child, idx))
            assert float(img[0, 0]) == 104.0, \
                f"expected DP at iy=1,ix=4 (=104), got {img[0, 0]}"

            # A second, asymmetric position confirms it isn't a lucky symmetry.
            sel._widget._data["cx"] = 0   # ix=0
            sel._widget._data["cy"] = 2   # iy=2
            idx = sel.get_selected_indices()
            img = np.asarray(update_from_navigation_selection(sel, child, idx))
            assert float(img[0, 0]) == 200.0, \
                f"expected DP at iy=2,ix=0 (=200), got {img[0, 0]}"
        finally:
            session.shutdown()
