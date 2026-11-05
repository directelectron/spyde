"""
Regions — the ragged per-field store a segmentation commits.
"""
from __future__ import annotations

import numpy as np

from spyde.segmentation.measure import COLUMNS, measure_instances
from spyde.signals import Regions
from spyde.signals.regions import NAVIGATION_SPACE


def _table(n: int, seed: int) -> dict:
    labels = np.zeros((40, 40), np.int32)
    rng = np.random.default_rng(seed)
    for index in range(n):
        y, x = rng.integers(4, 34, size=2)
        labels[y - 3:y + 3, x - 3:x + 3] = index + 1
    return measure_instances(labels, rng.random((40, 40)), scale=2.0)


class TestRegions:
    def test_from_tables_indexes_rows_by_field(self):
        tables = [_table(2, 1), None, _table(3, 2), {}]
        regions = Regions.from_tables(tables, field_shape=(40, 40), scale=2.0, units="nm")
        assert regions.n_fields == 4 and regions.n_regions == 5
        np.testing.assert_array_equal(regions.count_series(), [2, 0, 3, 0])
        assert regions.table_at(2)["label"].tolist() == [1, 2, 3]
        assert regions.table_at(1)["area"].size == 0
        assert list(regions.table_at(0)) == ["field"] + [name for name, _ in COLUMNS]

    def test_save_and_load_keep_the_extras(self, tmp_path):
        regions = Regions.from_tables([_table(2, 3)], field_shape=(40, 40), scale=0.25,
                                      units="µm", space=NAVIGATION_SPACE,
                                      params={"min_size": 20},
                                      provenance={"action": "Segment"})
        path = str(tmp_path / "regions.npz")
        regions.save(path)
        loaded = Regions.load(path)
        assert (loaded.field_shape, loaded.scale, loaded.units, loaded.space) == \
            ((40, 40), 0.25, "µm", NAVIGATION_SPACE)
        assert loaded.params == {"min_size": 20} and loaded.provenance["action"] == "Segment"
        np.testing.assert_array_equal(loaded.column("area"), regions.column("area"))
        assert "2 regions in 1 fields" in repr(loaded)
