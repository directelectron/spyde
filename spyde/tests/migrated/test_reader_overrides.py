"""Reads answered by a reader pinned on the tree instead of by the node's array.

Three windows have no array to slice: a Find Vectors result (frames drawn from
the vector store), a progressively-filled result while its batch runs (only the
blocks that have landed), and a CSB movie in raw-frame mode (one camera frame
cut from the event stream). Each pins a reader on its tree, and the navigator
read asks the tree first.

Every fixture here is lazy and chunked, which is the shape these windows have in
the app: the arrays behind them are placeholders a plain slice would paint black.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.actions.csb_raw_frame import RawFrameReader, install
from spyde.actions.find_vectors_action import (
    RenderedVectorsReader, build_vectors_result_tree,
)
from spyde.actions.live_signal import attach_signal_preview
from spyde.drawing.update_functions import update_from_navigation_selection
from spyde.signals.diffraction_vectors import (
    COL_INTENSITY, COL_KX, COL_KY, COL_TIME, N_COLS,
    SpyDEDiffractionVectors, _AxisLite,
)
from spyde.tests.migrated.conftest import _settle, close_session, make_session
from spyde.tests.migrated._async import wait_until


class _Pointer:
    """A selector stand-in: the navigator read only asks it whether the position
    it is being given integrates."""

    def __init__(self, integrating: bool = False):
        self.is_integrating = integrating


def _vectors(nav=(4, 4), sig=32):
    """One spot per navigation position, drifting with the column so different
    positions render different frames."""
    ny, nx = nav
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            row = np.zeros(N_COLS, np.float32)
            row[0], row[1] = ix, iy
            row[COL_TIME] = -1.0
            row[COL_KX] = -0.5 + 0.2 * ix
            row[COL_KY] = 0.1 * iy
            row[COL_INTENSITY] = 100.0 + ix + iy
            rows.append(row)
    axis = _AxisLite(scale=2.0 / (sig - 1), offset=-1.0, size=sig,
                     units="1/A", name="k")
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=np.stack(rows).astype(np.float32), full_nav_shape=(ny, nx),
        sig_shape=(sig, sig), sig_axes=[axis, axis], kernel_radius_px=3.0,
        kernel_radius_data=0.0, params={}, nav_axes=None,
    )


def _lazy_movie(planes=6, sig=(8, 8)):
    """A lazy movie, one frame per chunk, standing in for a loaded event stream."""
    frames = np.stack([np.full(sig, i, np.float32) for i in range(planes)])
    signal = hs.signals.Signal2D(da.from_array(frames, chunks=(1,) + sig)).as_lazy()
    return signal


class TestVectorsWindowReader:
    def test_a_point_reads_the_rendered_frame(self):
        session = make_session()
        try:
            vecs = _vectors()
            tree = build_vectors_result_tree(session, vecs)
            _settle(session)
            plot = tree.signal_plots[0]
            assert isinstance(tree.reader_override_for(tree.root),
                              RenderedVectorsReader)
            # A crosshair reports [[ix, iy]]; the frame is the store's own.
            frame = update_from_navigation_selection(
                _Pointer(), plot, np.array([[2, 1]]))
            np.testing.assert_array_equal(frame, vecs.render_frame(1, 2))
        finally:
            close_session(session)

    def test_a_region_reads_the_summed_render(self):
        """The store owns the rule, so the region IS ``render_region``: each
        position's disks at their intra-frame maximum, summed across
        positions, used exactly as it comes back."""
        session = make_session()
        try:
            vecs = _vectors()
            tree = build_vectors_result_tree(session, vecs)
            _settle(session)
            plot = tree.signal_plots[0]
            # A region selector reports every [ix, iy] it covers.
            grid = np.array([[ix, iy] for iy in (0, 1) for ix in (0, 1)])
            frame = update_from_navigation_selection(
                _Pointer(integrating=True), plot, grid)
            np.testing.assert_array_equal(frame, vecs.render_region(0, 2, 0, 2))
        finally:
            close_session(session)


class TestProgressivePreviewReader:
    def _result_tree(self, session, nav=(4, 5), sig=(16, 16)):
        from spyde.actions.commit import open_result_tree
        from spyde.drawing.selectors import CrosshairSelector

        shape = tuple(nav) + tuple(sig)
        placeholder = hs.signals.Signal2D(
            da.zeros(shape, chunks=(2, 2) + tuple(sig), dtype=np.float32)).as_lazy()
        navigator = hs.signals.BaseSignal(np.zeros(nav, dtype=np.float32)).T
        return open_result_tree(session, title="Preview", signal=placeholder,
                                navigator_override=navigator,
                                selector_type=CrosshairSelector)

    def test_an_unlanded_position_reads_nothing_and_keeps_the_frame(self):
        session = make_session()
        try:
            tree = self._result_tree(session)
            _settle(session)
            plot = tree.signal_plots[0]
            landed = {(1, 2)}

            def render(index):
                return (np.full((16, 16), 7.0, np.float32)
                        if tuple(index) in landed else None)

            preview = attach_signal_preview(session, tree, render=render,
                                            nav_shape=(4, 5))
            assert preview is not None
            preview.note_block((slice(1, 2), slice(2, 3)))

            # A landed position reads its frame.
            frame = update_from_navigation_selection(
                _Pointer(), plot, np.array([[2, 1]]))
            assert frame is not None and float(frame.max()) == 7.0
            plot.enqueue_paint(frame)
            assert wait_until(lambda: plot.current_data is frame, 5)

            # A position the batch has not reached reads nothing, and the
            # navigator leaves the last frame up rather than painting zeros.
            assert update_from_navigation_selection(
                _Pointer(), plot, np.array([[4, 3]])) is None
            assert plot.current_data is frame
        finally:
            close_session(session)


    def test_a_region_shows_its_centre_position(self):
        """Integrating over a half-finished result would have to wait for every
        position in it, so a region shows the one position at its centre, and
        nothing at all when that position has not landed."""
        session = make_session()
        try:
            tree = self._result_tree(session)
            _settle(session)
            plot = tree.signal_plots[0]
            landed = {(1, 2)}

            def render(index):
                return (np.full((16, 16), 7.0, np.float32)
                        if tuple(index) in landed else None)

            preview = attach_signal_preview(session, tree, render=render,
                                            nav_shape=(4, 5))
            preview.note_block((slice(1, 2), slice(2, 3)))

            # A region centred on the landed position (widget [ix, iy] pairs).
            around = np.array([[1, 1], [2, 1], [3, 1]])
            frame = update_from_navigation_selection(
                _Pointer(integrating=True), plot, around)
            assert frame is not None and float(frame.max()) == 7.0

            elsewhere = np.array([[3, 3], [4, 3]])
            assert update_from_navigation_selection(
                _Pointer(integrating=True), plot, elsewhere) is None
        finally:
            close_session(session)


class TestRawFrameToggle:
    def test_toggling_raw_leaves_another_trees_override_alone(self):
        session = make_session()
        try:
            vecs = _vectors()
            vectors_tree = build_vectors_result_tree(session, vecs)
            _settle(session)
            rendered = vectors_tree.reader_override_for(vectors_tree.root)
            assert isinstance(rendered, RenderedVectorsReader)

            session._add_signal(_lazy_movie(), source_path=None)
            _settle(session)
            movie_tree = session.signal_trees[-1]
            movie_plot = movie_tree.signal_plots[0]
            selector = next(
                s for s in movie_tree.navigator_plot_manager.all_navigation_selectors
                if movie_plot in s.children)

            assert install(selector, True) is True
            assert isinstance(
                movie_tree.reader_override_for(
                    movie_plot.plot_state.current_signal, movie_plot),
                RawFrameReader)
            assert vectors_tree.reader_override_for(vectors_tree.root) is rendered

            assert install(selector, False) is False
            assert movie_tree.reader_override_for(
                movie_plot.plot_state.current_signal, movie_plot) is None
            assert vectors_tree.reader_override_for(vectors_tree.root) is rendered
        finally:
            close_session(session)

    def test_two_windows_on_one_signal_toggle_raw_independently(self):
        """Raw is a way of looking at a movie, not a property of it. A second
        window on the same signal keeps showing integrated planes while the
        first shows raw frames, and turning the first off leaves the second
        alone."""
        session = make_session()
        try:
            session._add_signal(_lazy_movie(), source_path=None)
            _settle(session)
            tree = session.signal_trees[-1]
            navigator = [p for p in session._plots if p.is_navigator][0]
            navigator.multiplot_manager.add_navigation_selector_and_signal_plot(
                navigator.plot_window)
            _settle(session)

            first, second = tree.signal_plots[0], tree.signal_plots[1]
            assert first is not second
            signal = first.plot_state.current_signal
            selectors = {
                id(plot): next(
                    s for s in
                    tree.navigator_plot_manager.all_navigation_selectors
                    if plot in s.children)
                for plot in (first, second)
            }

            install(selectors[id(first)], True)
            install(selectors[id(second)], True)
            install(selectors[id(first)], False)

            assert tree.reader_override_for(signal, first) is None
            assert isinstance(tree.reader_override_for(signal, second),
                              RawFrameReader)
        finally:
            close_session(session)

    def test_closing_a_window_takes_its_pin_with_it(self):
        """A pin is keyed by the window, so one left behind outlives the thing
        it describes and would answer for a plot object nobody holds."""
        session = make_session()
        try:
            session._add_signal(_lazy_movie(), source_path=None)
            _settle(session)
            tree = session.signal_trees[-1]
            plot = tree.signal_plots[0]
            signal = plot.plot_state.current_signal
            selector = next(
                s for s in tree.navigator_plot_manager.all_navigation_selectors
                if plot in s.children)

            install(selector, True)
            assert tree.reader_override_for(signal, plot) is not None
            plot.close()
            assert tree._reader_overrides == {}
        finally:
            close_session(session)

    def test_raw_mode_on_a_signal_that_cannot_serve_it_reads_nothing(self):
        """The reader answers None rather than raising when the signal is not an
        event stream, so the navigator keeps the last frame."""
        session = make_session()
        try:
            session._add_signal(_lazy_movie(), source_path=None)
            _settle(session)
            tree = session.signal_trees[-1]
            plot = tree.signal_plots[0]
            reader = RawFrameReader(_Pointer(), plot.plot_state.current_signal)
            assert reader.read_frame((2,)) is None
            assert reader.frame_bytes == 8 * 8 * 4
        finally:
            close_session(session)
