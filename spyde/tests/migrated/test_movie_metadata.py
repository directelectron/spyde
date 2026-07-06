"""
Movie / in-situ metadata (Phase 8).

The metadata panel gains a "Movie / In-Situ" group with FPS + Frame time. The
value comes from the explicit metadata key when present, else is DERIVED from a
calibrated leading TIME axis (fps = 1/scale) so an in-situ movie shows real
numbers instead of "--".
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.metadata_extract import build_metadata_dict


class _Tree:
    """Minimal stand-in exposing what build_metadata_dict reads: .root,
    .signal_plots (empty → falls back to root), and get_nested_attr (used by the
    existing Dtype/Dim. config props)."""
    def __init__(self, root):
        self.root = root
        self.signal_plots = []

    def get_nested_attr(self, attr_path: str):
        obj = self
        for attr in (p for p in attr_path.split(".") if p):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj


def _movie(n=30, frame=(64, 64), name="time", units="sec", scale=0.1):
    s = hs.signals.Signal2D(
        da.zeros((n,) + frame, dtype=np.float32, chunks=(1,) + frame)).as_lazy()
    ax = s.axes_manager.navigation_axes[0]
    ax.name, ax.units, ax.scale = name, units, scale
    return s


class TestMovieMetadata:
    def test_group_present(self):
        md = build_metadata_dict(_Tree(_movie()))
        assert "Movie / In-Situ" in md
        assert "FPS" in md["Movie / In-Situ"]
        assert "Frame time" in md["Movie / In-Situ"]

    def test_fps_derived_from_time_axis(self):
        # scale 0.1 s/frame → 10 fps.
        md = build_metadata_dict(_Tree(_movie(scale=0.1)))
        movie = md["Movie / In-Situ"]
        assert "10" in movie["FPS"], movie["FPS"]
        assert "0.1" in movie["Frame time"], movie["Frame time"]

    def test_ms_units_converted(self):
        # 50 ms/frame → 20 fps.
        md = build_metadata_dict(_Tree(_movie(units="ms", scale=50.0)))
        assert "20" in md["Movie / In-Situ"]["FPS"]

    def test_explicit_fps_key_preferred(self):
        s = _movie(name="z", units="<undefined>", scale=1.0)   # not a time axis
        s.metadata.set_item("Acquisition_instrument.TEM.frames_per_second", 25.0)
        md = build_metadata_dict(_Tree(s))
        assert "25" in md["Movie / In-Situ"]["FPS"]

    def test_non_time_axis_shows_placeholder(self):
        # A 'z' stack with no fps metadata and no time calibration → "--".
        md = build_metadata_dict(_Tree(_movie(name="z", units="<undefined>", scale=1.0)))
        assert md["Movie / In-Situ"]["FPS"].startswith("--")
