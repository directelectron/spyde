"""Selector utility functions."""
from __future__ import annotations

import numpy as np


def broadcast_rows_cartesian(*arrays: np.ndarray) -> np.ndarray:
    """
    Cartesian product over *rows* of multiple index arrays, keeping
    the columns of each array together.

    Each input is treated as shape (Ni, Ci): Ni rows, Ci columns.
    The output has shape (N_total, sum(Ci)), where N_total is the
    product of all Ni.
    """
    if len(arrays) == 0:
        return np.empty((0, 0), dtype=int)

    mats = [np.atleast_2d(a) for a in arrays]
    n_rows = [m.shape[0] for m in mats]

    grids = np.meshgrid(*[np.arange(n) for n in n_rows], indexing="ij")

    parts = []
    for m, g in zip(mats, grids):
        chosen_rows = m[g.ravel()]
        parts.append(chosen_rows)

    return np.concatenate(parts, axis=1)


def no_return_update_function(selector, child_plot, indices: np.ndarray):
    """Placeholder update function that does nothing."""
    return None


# Stubs kept for import compatibility — linked ROI logic is now handled by
# anyplotlib widget state synchronization, not pyqtgraph signal chaining.

def create_linked_rect_roi(core_roi):
    return core_roi


def create_linked_linear_region(core_roi, pen=None, hover_pen=None):
    return core_roi


def create_linked_infinite_line(core_roi, pen=None, hover_pen=None):
    return core_roi
