"""
RaggedStore — the shared ragged per-nav-position column store.

Pins the base contract SpyDEDiffractionVectors (and later the particle store)
is wired over: streaming out-of-order fill == from_packed of the sorted input,
derived outer offset levels are zero-copy strided views matching the historical
``_build_nav_offsets`` output, the ``count_map() == map(None, 'count')`` law,
the versioned save/load format (append-only column padding included), rank-1
and rank-3 grids, and the structure-frozen lifecycle.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.signals.ragged_store import (
    RaggedStore, _AxisLite, build_leaf_offsets, derive_levels,
)


class _EventStore(RaggedStore):
    """Rank-2 store with a heterogeneous (dict-of-columns) schema."""
    columns_schema = (("iy", "i8"), ("ix", "i8"), ("val", "f4"))


class _SeriesStore(RaggedStore):
    """Rank-1 store (a time series of ragged events)."""
    columns_schema = (("t", "i8"), ("val", "f4"))


class _StackStore(RaggedStore):
    """Rank-3 packed store mirroring the 5-D vectors layout (single dtype)."""
    columns_schema = (("t", "f4"), ("iy", "f4"), ("ix", "f4"), ("val", "f4"))


def _random_packed(rng, full_nav_shape, max_per_pos=5):
    """A SORTED packed (N, 4) f4 buffer for _StackStore with random ragged
    counts (some positions empty)."""
    rows = []
    grid = np.ndindex(*full_nav_shape)
    for pos in grid:
        n = int(rng.integers(0, max_per_pos + 1))
        for _ in range(n):
            rows.append([*pos, rng.uniform(0, 100)])
    if not rows:
        return np.zeros((0, len(_StackStore.columns_schema)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


class TestStreamingEqualsFromPacked:
    def test_out_of_order_packed_batches(self):
        rng = np.random.default_rng(7)
        shape = (3, 4, 5)
        packed = _random_packed(rng, shape)
        # Shuffle rows, then split into ragged batches — the out-of-order
        # chunk-arrival case.
        shuffled = packed[rng.permutation(len(packed))]
        splits = np.array_split(shuffled, 4)

        st = _StackStore.streaming(shape, index_columns=("t", "iy", "ix"))
        total = 0
        for b in splits:
            total = st.append_batch(b)
        assert total == len(packed)
        st.finalize()

        # Reference: from_packed of the stable-sorted concatenation (same
        # arrival order finalize saw).
        flat = (shuffled[:, 0].astype(np.int64) * 20
                + shuffled[:, 1].astype(np.int64) * 5
                + shuffled[:, 2].astype(np.int64))
        ref_sorted = shuffled[np.argsort(flat, kind="stable")]
        ref = _StackStore.from_packed(ref_sorted, shape,
                                      index_columns=("t", "iy", "ix"))

        np.testing.assert_array_equal(st.offset_levels()[-1],
                                      ref.offset_levels()[-1])
        for name in ("t", "iy", "ix", "val"):
            np.testing.assert_array_equal(st.column(name), ref.column(name))
        np.testing.assert_array_equal(st.count_map(), ref.count_map())

    def test_dict_batches_match_packed(self):
        rng = np.random.default_rng(11)
        shape = (4, 3)
        iy = rng.integers(0, 4, 40)
        ix = rng.integers(0, 3, 40)
        val = rng.uniform(0, 1, 40).astype(np.float32)

        st = _EventStore.streaming(shape, index_columns=("iy", "ix"))
        st.append_batch({"iy": iy[:25], "ix": ix[:25], "val": val[:25]})
        st.append_batch({"iy": iy[25:], "ix": ix[25:], "val": val[25:]})
        st.finalize()

        order = np.argsort(iy * 3 + ix, kind="stable")
        np.testing.assert_array_equal(st.column("iy"), iy[order])
        np.testing.assert_array_equal(st.column("ix"), ix[order])
        np.testing.assert_array_equal(st.column("val"), val[order])
        counts = np.bincount(iy * 3 + ix, minlength=12).reshape(4, 3)
        np.testing.assert_array_equal(st.count_map(), counts)

    def test_single_sorted_packed_batch_is_adopted_not_copied(self):
        """The already-sorted fast path keeps the block object-identical —
        what keeps the from_arrays/orchestrate path byte-for-byte as it was."""
        rng = np.random.default_rng(3)
        shape = (2, 3, 4)
        packed = _random_packed(rng, shape)
        st = _StackStore.streaming(shape, index_columns=("t", "iy", "ix"))
        st.append_batch(packed)
        st.finalize()
        assert st.flatten() is packed

    def test_mixing_batch_kinds_raises(self):
        st = _EventStore.streaming((2, 2), index_columns=("iy", "ix"))
        st.append_batch({"iy": [0], "ix": [1], "val": [2.0]})
        with pytest.raises(TypeError):
            st.append_batch(np.zeros((1, 3)))

    def test_zero_batches_finalizes_empty(self):
        st = _EventStore.streaming((2, 3), index_columns=("iy", "ix"))
        st.finalize()
        assert st.count_map().shape == (2, 3)
        assert st.count_map().sum() == 0
        assert st.column("val").shape == (0,)
        assert all(v.shape == (0,) for v in st.at(1, 2).values())


class TestDerivedLevels:
    def test_levels_match_build_nav_offsets_and_are_views(self):
        """The rectangular-grid stride identity: every outer level equals the
        historically materialised ``_build_nav_offsets`` output, but as a
        ZERO-COPY strided view of the leaf."""
        from spyde.signals.diffraction_vectors import (
            _build_nav_offsets, N_COLS, COL_NAV_X, COL_NAV_Y, COL_TIME,
        )
        rng = np.random.default_rng(5)
        shape = (3, 4, 5)
        packed = _random_packed(rng, shape)                       # (t, iy, ix, val)

        # The same rows in the vectors (N, 6) layout for the historical builder.
        flat6 = np.zeros((len(packed), N_COLS), dtype=np.float32)
        flat6[:, COL_TIME] = packed[:, 0]
        flat6[:, COL_NAV_Y] = packed[:, 1]
        flat6[:, COL_NAV_X] = packed[:, 2]
        legacy = _build_nav_offsets(flat6, shape)

        st = _StackStore.from_packed(packed, shape,
                                     index_columns=("t", "iy", "ix"))
        levels = st.offset_levels()
        assert len(levels) == len(legacy) == 3
        for got, want in zip(levels, legacy):
            np.testing.assert_array_equal(got, want)
        leaf = levels[-1]
        for outer in levels[:-1]:
            assert np.shares_memory(outer, leaf)

    def test_rank1_single_level(self):
        st = _SeriesStore.streaming((4,), index_columns=("t",))
        st.append_batch({"t": [2, 0, 2, 3], "val": [1.0, 2.0, 3.0, 4.0]})
        st.finalize()
        levels = st.offset_levels()
        assert len(levels) == 1
        np.testing.assert_array_equal(levels[0], [0, 1, 1, 3, 4])

    def test_columns_are_zero_copy_views_of_packed(self):
        rng = np.random.default_rng(2)
        shape = (2, 2, 2)
        packed = _random_packed(rng, shape)
        st = _StackStore.from_packed(packed, shape,
                                     index_columns=("t", "iy", "ix"))
        assert np.shares_memory(st.column("val"), packed)
        if len(packed):
            row = st.at(0, 0, 0) if st.counts()[0] else st.at(*np.unravel_index(
                int(np.argmax(st.counts())), shape))
            assert all(np.shares_memory(v, packed) for v in row.values()
                       if v.size)


class TestCountMapMapLaw:
    def _store(self, seed=9, shape=(4, 6)):
        rng = np.random.default_rng(seed)
        n = 60
        iy = rng.integers(0, shape[0], n)
        # Leave column 0 empty so the law covers empty positions too.
        ix = rng.integers(1, shape[1], n)
        val = rng.uniform(-5, 5, n).astype(np.float32)
        st = _EventStore.streaming(shape, index_columns=("iy", "ix"))
        st.append_batch({"iy": iy, "ix": ix, "val": val})
        return st.finalize(), iy, ix, val

    def test_count_map_equals_map_count(self):
        st, *_ = self._store()
        cm = st.count_map()
        mapped = st.map(None, "count")
        assert cm.dtype == mapped.dtype == np.int64
        np.testing.assert_array_equal(cm, mapped)
        assert (cm[:, 0] == 0).all()          # the law holds over empties

    def test_map_reducers_match_manual_loop(self):
        st, iy, ix, val = self._store()
        shape = st.full_nav_shape
        for reducer, fn in [("sum", np.sum), ("mean", np.mean),
                            ("max", np.max), ("min", np.min),
                            ("median", np.median), ("std", np.std)]:
            got = st.map("val", reducer)
            for y in range(shape[0]):
                for x in range(shape[1]):
                    rows = val[(iy == y) & (ix == x)]
                    if not len(rows):
                        want = 0.0 if reducer == "sum" else np.nan
                    else:
                        want = fn(rows.astype(np.float64))
                    np.testing.assert_allclose(
                        got[y, x], want, atol=1e-6, err_msg=f"{reducer} ({y},{x})")

    def test_map_callable_and_fill(self):
        st, *_ = self._store()
        got = st.map("val", lambda rows: float(len(rows)), fill=-1.0)
        counts = st.count_map()
        np.testing.assert_array_equal(got[counts > 0],
                                      counts[counts > 0].astype(np.float64))
        assert (got[counts == 0] == -1.0).all()


class TestSaveLoad:
    def _store(self):
        st = _EventStore.streaming(
            (2, 3), index_columns=("iy", "ix"),
            nav_axes=[_AxisLite(scale=2.0, offset=1.0, size=2, units="nm", name="y"),
                      _AxisLite(scale=3.0, offset=-1.0, size=3, units="nm", name="x")],
            params={"threshold": 0.5},
            provenance={"action": "test", "spyde_version": "0"})
        st.append_batch({"iy": [1, 0, 1], "ix": [2, 0, 2], "val": [7.0, 8.0, 9.0]})
        return st.finalize()

    def test_round_trip(self, tmp_path):
        st = self._store()
        path = str(tmp_path / "store.npz")
        st.save(path)
        back = _EventStore.load(path)
        assert back.full_nav_shape == (2, 3)
        np.testing.assert_array_equal(back.offset_levels()[-1],
                                      st.offset_levels()[-1])
        for name in ("iy", "ix", "val"):
            np.testing.assert_array_equal(back.column(name), st.column(name))
        assert back.params == {"threshold": 0.5}
        assert back.provenance["action"] == "test"
        assert len(back.nav_axes) == 2
        assert back.nav_axes[0].scale == 2.0 and back.nav_axes[1].name == "x"
        np.testing.assert_array_equal(back.count_map(), st.count_map())

    def test_older_file_prefix_columns_are_padded(self, tmp_path):
        """Append-only rule: a file written before a column existed loads with
        that column zero-filled at the schema dtype."""

        class _V1(RaggedStore):
            columns_schema = (("iy", "i8"), ("ix", "i8"), ("val", "f4"))

        class _V2(RaggedStore):
            columns_schema = (("iy", "i8"), ("ix", "i8"), ("val", "f4"),
                              ("weight", "f8"))
            format_version = 2

        old = _V1.streaming((2, 2), index_columns=("iy", "ix"))
        old.append_batch({"iy": [0, 1], "ix": [1, 0], "val": [3.0, 4.0]})
        old.finalize()
        path = str(tmp_path / "old.npz")
        old.save(path)

        new = _V2.load(path)
        np.testing.assert_array_equal(new.column("val"), old.column("val"))
        pad = new.column("weight")
        assert pad.dtype == np.float64 and pad.shape == (2,)
        assert (pad == 0).all()

    def test_newer_format_version_is_rejected(self, tmp_path):
        class _New(RaggedStore):
            columns_schema = (("iy", "i8"), ("ix", "i8"), ("val", "f4"))
            format_version = 99

        st = _New.streaming((1, 1), index_columns=("iy", "ix"))
        st.append_batch({"iy": [0], "ix": [0], "val": [1.0]})
        st.finalize()
        path = str(tmp_path / "new.npz")
        st.save(path)
        with pytest.raises(ValueError, match="format_version"):
            _EventStore.load(path)

    def test_non_prefix_columns_are_rejected(self, tmp_path):
        class _Other(RaggedStore):
            columns_schema = (("a", "f4"), ("b", "f4"))

        st = self._store()
        path = str(tmp_path / "store.npz")
        st.save(path)
        with pytest.raises(ValueError, match="prefix"):
            _Other.load(path)


class TestRanks:
    def test_rank1_round_trip_access(self):
        st = _SeriesStore.streaming((5,), index_columns=("t",))
        st.append_batch({"t": [4, 1, 1], "val": [9.0, 5.0, 6.0]})
        st.finalize()
        assert st.count_map().shape == (5,)
        np.testing.assert_array_equal(st.count_map(), [0, 2, 0, 0, 1])
        np.testing.assert_array_equal(st.at(1)["val"], [5.0, 6.0])
        np.testing.assert_array_equal(st.at(4)["val"], [9.0])
        assert st.at(0)["val"].shape == (0,)

    def test_rank3_full_and_prefix_slicing(self):
        rng = np.random.default_rng(13)
        shape = (2, 3, 4)
        packed = _random_packed(rng, shape)
        st = _StackStore.from_packed(packed, shape,
                                     index_columns=("t", "iy", "ix"))
        # Full index: every position's rows match a mask over the raw buffer.
        for t in range(2):
            for iy in range(3):
                for ix in range(4):
                    mask = ((packed[:, 0] == t) & (packed[:, 1] == iy)
                            & (packed[:, 2] == ix))
                    np.testing.assert_array_equal(
                        st.at(t, iy, ix)["val"], packed[mask, 3])
        # Prefix: slice_at(t) and slice_at(t, iy) are O(1) outer-level reads.
        for t in range(2):
            np.testing.assert_array_equal(
                st.slice_at(t)["val"], packed[packed[:, 0] == t, 3])
            for iy in range(3):
                mask = (packed[:, 0] == t) & (packed[:, 1] == iy)
                np.testing.assert_array_equal(
                    st.slice_at(t, iy)["val"], packed[mask, 3])


class TestStructureFrozen:
    def _store(self):
        st = _EventStore.streaming((2, 2), index_columns=("iy", "ix"))
        st.append_batch({"iy": [0, 1], "ix": [0, 1], "val": [1.0, 2.0]})
        return st.finalize()

    def test_append_after_finalize_raises(self):
        st = self._store()
        with pytest.raises(RuntimeError, match="frozen"):
            st.append_batch({"iy": [0], "ix": [0], "val": [3.0]})

    def test_reads_before_finalize_raise(self):
        st = _EventStore.streaming((2, 2), index_columns=("iy", "ix"))
        for read in (st.count_map, st.offset_levels,
                     lambda: st.column("val"), lambda: st.at(0, 0),
                     lambda: st.map("val")):
            with pytest.raises(RuntimeError, match="finalized"):
                read()

    def test_finalize_is_idempotent(self):
        st = self._store()
        leaf = st.offset_levels()[-1]
        assert st.finalize() is st
        assert st.offset_levels()[-1] is leaf

    def test_cell_values_stay_writable_in_place(self):
        """Structure-frozen, not value-frozen: the owning subclass may
        overwrite cell VALUES through the column views."""
        st = self._store()
        st.column("val")[0] = 42.0
        np.testing.assert_array_equal(st.at(0, 0)["val"], [42.0])

    def test_packed_values_writable_through_backing(self):
        packed = np.array([[0, 0, 1.0], [1, 1, 2.0]], dtype=np.float32)

        class _P(RaggedStore):
            columns_schema = (("iy", "f4"), ("ix", "f4"), ("val", "f4"))

        st = _P.from_packed(packed, (2, 2), index_columns=("iy", "ix"))
        packed[0, 2] = 5.0
        np.testing.assert_array_equal(st.at(0, 0)["val"], [5.0])


class TestBuilders:
    def test_build_leaf_offsets_out_of_grid_raises(self):
        with pytest.raises(ValueError, match="outside"):
            build_leaf_offsets([np.array([5]), np.array([0])], (2, 2))

    def test_derive_levels_shapes(self):
        leaf = np.arange(0, 25, dtype=np.int64)      # (3*2*4)+1 = 25 entries
        levels = derive_levels(leaf, (3, 2, 4))
        assert [len(v) for v in levels] == [4, 7, 25]
        assert levels[-1] is leaf
        np.testing.assert_array_equal(levels[0], leaf[::8])
        np.testing.assert_array_equal(levels[1], leaf[::4])
