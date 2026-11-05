"""
Regions — the instances a trainable segmentation found, one row per instance.

A :class:`~spyde.signals.ragged_store.RaggedStore` whose navigation grid is
the list of FIELDS the segmentation ran over: one field for a single image or
a 4-D STEM navigator, one per frame for a movie. The ``field`` column indexes
it; the other columns are :data:`spyde.segmentation.measure.COLUMNS`, already
calibrated in ``units``.

``space`` records what a pixel of the field was: ``"signal"`` (an image
pixel) or ``"navigation"`` (a scan position of a 4-D dataset, so a region is
a set of scan positions and its diffraction pattern is well-defined).

The label images themselves are not stored here: they are recomputed from the
classifier that produced them, the way a drift-corrected node recomputes its
frames (see ``segment_action``).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from spyde.segmentation.measure import COLUMNS, empty_table
from spyde.signals.ragged_store import RaggedStore, build_leaf_offsets

SIGNAL_SPACE = "signal"
NAVIGATION_SPACE = "navigation"


class Regions(RaggedStore):
    columns_schema = (("field", "i4"),) + COLUMNS
    format_version = 1

    #: ``(height, width)`` of every field.
    field_shape: Tuple[int, int] = (0, 0)
    #: Pixel size the length and area columns were calibrated with, and its unit.
    scale: float = 1.0
    units: str = "px"
    #: What a field pixel is: :data:`SIGNAL_SPACE` or :data:`NAVIGATION_SPACE`.
    space: str = SIGNAL_SPACE

    @classmethod
    def from_tables(cls, tables: Sequence[Dict[str, np.ndarray]], *,
                    field_shape: Tuple[int, int], scale: float = 1.0,
                    units: str = "px", space: str = SIGNAL_SPACE,
                    nav_axes: Sequence[object] = (), params: dict | None = None,
                    provenance: dict | None = None) -> "Regions":
        """Build from one measured table per field, in field order.

        *tables* are the dicts :func:`~spyde.segmentation.measure_instances`
        returns; an empty dict or ``None`` stands for a field with no instances.
        """
        n_fields = len(tables)
        filled = [table if table else empty_table() for table in tables]
        columns = {name: np.concatenate([np.asarray(table[name], dtype)
                                         for table in filled])
                   for name, dtype in COLUMNS}
        columns["field"] = np.concatenate(
            [np.full(len(table["label"]), index, np.int32)
             for index, table in enumerate(filled)])
        offsets = build_leaf_offsets([columns["field"]], (n_fields,))
        regions = cls(columns, offsets, (n_fields,), index_columns=("field",),
                      nav_axes=nav_axes, params=params, provenance=provenance)
        regions.field_shape = (int(field_shape[0]), int(field_shape[1]))
        regions.scale = float(scale)
        regions.units = str(units)
        regions.space = str(space)
        return regions

    @property
    def n_fields(self) -> int:
        return int(self.full_nav_shape[0])

    @property
    def n_regions(self) -> int:
        return int(self.offsets[-1])

    def count_series(self) -> np.ndarray:
        """Instances per field, ``(n_fields,)``."""
        return self.counts()

    def table_at(self, field: int) -> Dict[str, np.ndarray]:
        """The measured columns of one field, as views."""
        return self.at(int(field))

    def _save_extra(self):
        return {}, {"field_shape": list(self.field_shape), "scale": self.scale,
                    "units": self.units, "space": self.space}

    @classmethod
    def _load_extra(cls, arrays, meta):
        return {"field_shape": tuple(int(size) for size in meta.get("field_shape", (0, 0))),
                "scale": float(meta.get("scale", 1.0)),
                "units": str(meta.get("units", "px")),
                "space": str(meta.get("space", SIGNAL_SPACE))}

    def __repr__(self) -> str:
        return (f"Regions({self.n_regions} regions in {self.n_fields} fields of "
                f"{self.field_shape[0]}x{self.field_shape[1]} {self.space} px, "
                f"{self.units})")
