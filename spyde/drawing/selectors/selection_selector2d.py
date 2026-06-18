"""Static selection display — no-op stubs (anyplotlib widgets are self-contained)."""
from __future__ import annotations
from typing import List


class SelectionSelector2D:
    """Stub for static selection display.

    In the anyplotlib architecture, widgets are self-contained overlays and
    don't need a separate static-copy mechanism.  This class is kept for
    import compatibility.
    """

    def __init__(self, rois: List = None):
        self.rois = rois or []
        self._last_clicked_index = None

    def _on_roi_clicked(self, index: int) -> None:
        self._last_clicked_index = index
