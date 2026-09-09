"""
labels.py — the scribbles a user paints to train the segmentation.

Three classes, fixed: ``PARTICLE`` (what to find), ``BACKGROUND`` (everything
else) and ``BOUNDARY`` (the seam between two touching particles — the ilastik
convention). A boundary stroke is optional; when one is painted the instance
split becomes a connected-components pass instead of a watershed, which is
both faster and more accurate on touching particles.

Painted pixels are stored per field as flat indices with a class and a stroke
id, not as a dense label image: a stroke is a few hundred pixels of a
16-million-pixel frame. Repainting a pixel replaces its class (last write
wins, as a brush does); erasing removes it.

Stroke ids survive because the classifier weights every STROKE equally rather
than every pixel — otherwise a long stroke outvotes a short one and painting
more of one particle silently reweights the model toward it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

PARTICLE, BACKGROUND, BOUNDARY = 0, 1, 2


@dataclass(frozen=True)
class LabelClass:
    id: int
    name: str
    colour: str


CLASSES: tuple[LabelClass, ...] = (
    LabelClass(PARTICLE, "particle", "#f9a03f"),
    LabelClass(BACKGROUND, "background", "#89b4fa"),
    LabelClass(BOUNDARY, "boundary", "#f38ba8"),
)

CLASS_IDS: tuple[int, ...] = tuple(label_class.id for label_class in CLASSES)


def disc_indices(shape: tuple[int, int], y: float, x: float, radius: float) -> np.ndarray:
    """Flat indices of a filled disc, clipped to the field.

    Built from the bounding box rather than a full-field distance map: a brush
    dab is a handful of pixels, and a full ``mgrid`` per dab makes painting a
    4096² field unusable.
    """
    height, width = int(shape[0]), int(shape[1])
    radius = max(0.5, float(radius))
    y0, y1 = max(0, math.floor(y - radius)), min(height - 1, math.ceil(y + radius))
    x0, x1 = max(0, math.floor(x - radius)), min(width - 1, math.ceil(x + radius))
    if y1 < y0 or x1 < x0:
        return np.zeros(0, np.int64)
    ys = np.arange(y0, y1 + 1)[:, None]
    xs = np.arange(x0, x1 + 1)[None, :]
    inside = (ys - y) ** 2 + (xs - x) ** 2 <= radius * radius
    yy, xx = np.nonzero(inside)
    return (yy + y0).astype(np.int64) * width + (xx + x0).astype(np.int64)


def stroke_indices(shape: tuple[int, int], points: Sequence[Sequence[float]],
                   radius: float) -> np.ndarray:
    """Flat indices under a polyline of ``(y, x)`` points stroked ``radius`` wide.

    The polyline is densified to half-pixel steps first: the widget reports a
    pointer sample per animation frame, and a fast stroke jumps many pixels
    between samples, so dabbing only at the samples leaves a dotted line.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return np.zeros(0, np.int64)
    dense = [points[0]]
    for start, end in zip(points, points[1:]):
        steps = int(math.ceil(np.hypot(*(end - start)) * 2.0))
        if steps > 1:
            dense.extend(start + (end - start) * (np.arange(1, steps + 1) / steps)[:, None])
        else:
            dense.append(end)
    discs = [disc_indices(shape, y, x, radius) for y, x in dense]
    return np.unique(np.concatenate(discs))


class Labels:
    """Scribbles keyed by field index, accumulating across fields.

    ``shape`` is the ``(height, width)`` of every field. It is fixed for the
    life of the store because a flat index means nothing without it.
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        # field index -> (flat indices int64, class ids int16, stroke ids int32)
        self._fields: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._next_stroke = 0

    # -- painting ------------------------------------------------------------

    def paint(self, field: int, points: Sequence[Sequence[float]], class_id: int,
              *, radius: float) -> int:
        """Paint one stroke of ``(y, x)`` image-pixel points; returns pixels painted."""
        if int(class_id) not in CLASS_IDS:
            raise ValueError(f"unknown class {class_id}; classes are {CLASS_IDS}")
        indices = stroke_indices(self.shape, points, radius)
        if not indices.size:
            return 0
        stroke = self._next_stroke
        self._next_stroke += 1
        current = self._fields.get(int(field), self._empty())
        merged = (np.concatenate([current[0], indices]),
                  np.concatenate([current[1], np.full(indices.size, int(class_id), np.int16)]),
                  np.concatenate([current[2], np.full(indices.size, stroke, np.int32)]))
        self._fields[int(field)] = _last_write_wins(*merged)
        return int(indices.size)

    def erase(self, field: int, points: Sequence[Sequence[float]], *, radius: float) -> int:
        """Unlabel the pixels under a stroke; returns pixels removed."""
        current = self._fields.get(int(field))
        if current is None:
            return 0
        drop = np.isin(current[0], stroke_indices(self.shape, points, radius))
        if drop.all():
            del self._fields[int(field)]
        else:
            self._fields[int(field)] = tuple(array[~drop] for array in current)
        return int(drop.sum())

    def clear(self) -> None:
        self._fields.clear()

    # -- inspection ----------------------------------------------------------

    def at(self, field: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(flat_indices, class_ids, stroke_ids)`` for one field; empty if unpainted."""
        return self._fields.get(int(field), self._empty())

    def painted_fields(self) -> list[int]:
        return sorted(self._fields)

    def counts(self) -> dict[int, int]:
        """Painted pixels per class id, across every field, zero included."""
        totals = {class_id: 0 for class_id in CLASS_IDS}
        for _indices, class_ids, _strokes in self._fields.values():
            for class_id, count in zip(*np.unique(class_ids, return_counts=True)):
                totals[int(class_id)] += int(count)
        return totals

    def __len__(self) -> int:
        return int(sum(indices.size for indices, _, _ in self._fields.values()))

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe, so the scribbles that trained a result travel with it."""
        return {
            "shape": list(self.shape),
            "fields": {str(field): {"index": indices.tolist(),
                                    "class": class_ids.tolist(),
                                    "stroke": strokes.tolist()}
                       for field, (indices, class_ids, strokes)
                       in sorted(self._fields.items())},
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "Labels":
        labels = cls(tuple(state["shape"]))
        for field, block in (state.get("fields") or {}).items():
            strokes = np.asarray(block["stroke"], dtype=np.int32)
            labels._fields[int(field)] = (np.asarray(block["index"], dtype=np.int64),
                                          np.asarray(block["class"], dtype=np.int16),
                                          strokes)
            if strokes.size:
                labels._next_stroke = max(labels._next_stroke, int(strokes.max()) + 1)
        return labels

    @staticmethod
    def _empty() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.zeros(0, np.int64), np.zeros(0, np.int16), np.zeros(0, np.int32))


def _last_write_wins(indices: np.ndarray, class_ids: np.ndarray,
                     strokes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated indices keeping the LAST entry written to each.

    ``np.unique`` keeps the first occurrence, so the arrays are reversed going
    in: a repaint over an existing stroke has to change its class, not be
    ignored.
    """
    unique, first = np.unique(indices[::-1], return_index=True)
    return unique, class_ids[::-1][first], strokes[::-1][first]
