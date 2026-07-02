"""
vector_virtual_imaging.py — Electron-native Vector Virtual Imaging.

Builds virtual images from a ``SpyDEDiffractionVectors`` result tree (the output
of Find Diffraction Vectors). REUSES the raw Virtual-Imaging machinery: it is the
same multi-VI sub-toolbar UX (a "＋" adds a colour-cycled detector ROI on the
vectors diffraction pattern → its own output window that recomputes live), built
on the host-agnostic :class:`~spyde.actions.virtual_image.VirtualImageAction`
template. Only :meth:`reduce` differs — instead of a Dask reduction over the raw
4D dataset, each image is built in-memory from the CSR flat buffer via
``vecs.virtual_image_from_roi_gpu`` (O(N_frame), recomputes live on every drag).

Each ROI is intensity-weighted (sums NXCORR peak scores) by default, matching how
the vectors were detected; a "count" weighting tallies vectors instead.

No Qt — mirrors :mod:`spyde.actions.virtual_image` so it runs in the Electron
backend (and a notebook) unchanged.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.virtual_image import VirtualImageAction, VI_COLORS

log = logging.getLogger(__name__)


class VectorVirtualImageAction(VirtualImageAction):
    """A virtual image computed from the tree's diffraction vectors (not the raw
    4D data). Inherits the nav-shaped placeholder + live-recompute flow from
    :class:`VirtualImageAction`; overrides the selector mapping and the reduce."""

    name = "Vector Virtual Image"
    output_node_name = "Vector Virtual Image"

    # Same param KEYS as the raw VI (``type``/``calculation``) so the multi-VI
    # sub-toolbar UI (shape icon + per-VI caret) is reused verbatim — only the
    # option lists/meaning differ (calculation = vector weighting here).
    parameters = {
        "type": {
            "name": "Detector shape",
            "type": "enum",
            "default": "disk",
            "options": ["disk", "annular", "rectangle"],
        },
        "calculation": {
            "name": "Weighting",
            "type": "enum",
            "default": "intensity",
            "options": ["intensity", "count"],
        },
    }

    def selector_for_params(self, **params):
        from spyde.drawing.selectors import (
            CircleSelector, AnnularSelector, RectangleSelector,
        )
        return {
            "disk": CircleSelector,
            "annular": AnnularSelector,
            "rectangle": RectangleSelector,
        }.get(params.get("type", "disk"), CircleSelector)

    #: When set (5-D only), the VI output is a SEPARATE navigable 3-D result tree
    #: (navigator = stack/time line, signal = the per-slice VI map). The detector
    #: ROI lives on the source vectors DP; every move recomputes the FULL
    #: (n_t, nav_y, nav_x) stack and replaces this tree's root data. See
    #: :func:`add_vector_virtual_image`.
    vi_tree = None

    def _current_t(self):
        """The stack/time index the SOURCE navigator is parked on (5-D only), so
        the VI reflects the slice the user is viewing. None for 4-D."""
        vecs = getattr(self.signal_tree, "diffraction_vectors", None)
        if vecs is None or getattr(vecs, "n_time", 0) <= 0:
            return None
        try:
            return int(self.signal_tree.root.axes_manager.indices[0])
        except Exception:
            return None

    def _roi_geom(self, signal, widget):
        """Parse the anyplotlib ROI widget into calibrated detector geometry.

        Widgets report geometry in *image-pixel* coords; the vectors store kx,ky
        in *calibrated* units, so convert pixel → calibrated (``k = px*scale +
        offset``). Returns ``("rect", (x0, y0, x1, y1))`` or
        ``("disk", (cx, cy, r_outer, r_inner))``."""
        sig_axes = signal.axes_manager.signal_axes
        sx, ox = float(sig_axes[0].scale), float(sig_axes[0].offset)
        sy, oy = float(sig_axes[1].scale), float(sig_axes[1].offset)
        data = getattr(widget, "_data", {})
        if "w" in data and "h" in data:                      # rectangle
            x0 = float(widget.x) * sx + ox
            y0 = float(widget.y) * sy + oy
            x1 = (float(widget.x) + float(widget.w)) * sx + ox
            y1 = (float(widget.y) + float(widget.h)) * sy + oy
            return "rect", (x0, y0, x1, y1)
        cx = float(widget.cx) * sx + ox                      # disk / annulus
        cy = float(widget.cy) * sy + oy
        if "r_outer" in data:
            r_out = float(widget.r_outer) * sx
            r_in = float(widget.r_inner) * sx
        else:
            r_out = float(widget.r) * sx
            r_in = 0.0
        return "disk", (cx, cy, r_out, r_in)

    def reduce(self, signal, selector, indices, **params):
        """Build the virtual image for the current detector ROI from the vectors.

        Returns a 2-D nav-space numpy image (in-memory, synchronous — sub-ms for
        typical data) for the CURRENT stack slice (``t=``; None for 4-D). For the
        5-D navigable output (``self.vi_tree`` set), :meth:`reduce_to` recomputes
        the whole stack and pushes it into the result tree instead — see there."""
        vecs = getattr(self.signal_tree, "diffraction_vectors", None)
        widget = getattr(selector, "roi", None)
        if vecs is None or widget is None:
            return None
        weighted = params.get("calculation", "intensity") != "count"
        t = self._current_t()
        try:
            kind, geom = self._roi_geom(signal, widget)
            if kind == "rect":
                x0, y0, x1, y1 = geom
                img = vecs.virtual_image_from_rect(
                    x0, y0, x1, y1, t=t, intensity_weighted=weighted)
            else:
                cx, cy, r_out, r_in = geom
                img = vecs.virtual_image_from_roi_gpu(
                    cx, cy, r_out, r_in, t=t, intensity_weighted=weighted)
        except Exception:
            return None
        return np.asarray(img, dtype=np.float32)

    def _series_array(self, signal, selector, **params):
        """Compute the FULL (n_t, nav_y, nav_x) virtual-image stack for the
        current detector ROI — every stack slice in one O(N_total) CSR pass via
        :meth:`SpyDEDiffractionVectors.virtual_image_series`. Returns None if
        there's no ROI yet."""
        vecs = getattr(self.signal_tree, "diffraction_vectors", None)
        widget = getattr(selector, "roi", None)
        if vecs is None or widget is None:
            return None
        weighted = params.get("calculation", "intensity") != "count"
        try:
            kind, geom = self._roi_geom(signal, widget)
            if kind == "rect":
                x0, y0, x1, y1 = geom
                stack = vecs.virtual_image_series_rect(
                    x0, y0, x1, y1, intensity_weighted=weighted)
            else:
                cx, cy, r_out, r_in = geom
                stack = vecs.virtual_image_series(
                    cx, cy, r_out, r_in, intensity_weighted=weighted)
        except Exception as e:
            log.debug("vector VI series compute failed: %s", e)
            return None
        return np.asarray(stack, dtype=np.float32)

    def run_into_tree(self, vi_tree, **params):
        """5-D variant of :meth:`RegionAction.run`: place the detector ROI on the
        SOURCE vectors DP, but route its live recompute into ``vi_tree`` (a
        separate navigable 3-D result window) instead of a single output plot.

        The result tree owns the stack/time navigator; this just keeps replacing
        its root data as the ROI moves. ``children`` is the tree's signal plot so
        the RegionAction machinery has a valid child to push the current slice to
        (the tree's own navigator drives which slice is shown)."""
        self.vi_tree = vi_tree
        resolved = self._resolved_params(params)
        self._live_params = dict(resolved)

        sig_plots = list(getattr(vi_tree, "signal_plots", []))
        if not sig_plots:
            log.debug("VI tree has no signal plot; falling back to single-plot VI")
            return self.run(**params)
        self._out_plot = sig_plots[0]
        selector = self._make_selector(self._out_plot)
        self._selector = selector
        try:
            selector.delayed_update_data(force=True)
        except Exception as e:
            log.debug("initial 5-D vector VI compute failed: %s", e)
        return selector

    def reduce_to(self, signal, selector, child, indices, **params):
        """4-D: single VI image (delegates to :meth:`reduce`). 5-D
        (``self.vi_tree`` set): recompute the FULL 3-D stack and replace the
        result tree's root data, forcing its navigator to re-slice so both the
        navigator summary and the currently-viewed slice refresh. Returns None —
        the tree's OWN navigator owns the signal plot, so the selector must NOT
        push a slice directly (that would race the navigator re-slice)."""
        if self.vi_tree is None:
            return self.reduce(signal, selector, indices, **params)

        stack = self._series_array(signal, selector, **params)
        if stack is None:
            return None
        self._push_stack_to_tree(stack)
        return None

    def _push_stack_to_tree(self, stack):
        """Replace the 5-D VI result tree's root data with the freshly-computed
        stack and force a navigator re-slice. Marshalled onto the asyncio main
        thread (the update hook runs on the _NavDispatcher thread; figure/IPC
        touches must happen on the main loop)."""
        tree = self.vi_tree

        def _apply():
            try:
                tree.root.data = stack
                # Drop the cached dask view captured at first render so the
                # navigator slices the NEW data, not the stale placeholder.
                try:
                    tree.root.cached_dask_array = None
                    tree.root._clear_cache_dask_data()
                except Exception as e:
                    log.debug("clearing VI tree cache failed: %s", e)
                npm = getattr(tree, "navigator_plot_manager", None)
                if npm is not None:
                    for sels in getattr(npm, "navigation_selectors", {}).values():
                        for sel in sels:
                            try:
                                sel.delayed_update_data(force=True)
                            except Exception as e:
                                log.debug("VI tree nav re-slice failed: %s", e)
            except Exception as e:
                log.debug("pushing VI stack to tree failed: %s", e)

        session = getattr(self, "session", None)
        if session is not None and hasattr(session, "_dispatch_to_main"):
            session._dispatch_to_main(_apply)
        else:
            _apply()


def vector_virtual_imaging(ctx, action_name: str = "Vector Virtual Imaging", **kwargs):
    """Parent submenu action — a no-op; the toolbar opens its sub-toolbar
    ("Add Vector Virtual Image") instead of dispatching this."""
    return None


def _spawn_navigable_vi(ctx, act, vtype, calc, color):
    """5-D only: build a navigable 3-D virtual-image result window from the
    vectors and wire the source detector ROI to recompute its full stack live.

    The result is a HyperSpy ``Signal2D`` of shape (n_t, nav_y, nav_x) → nav_dim
    1 (stack/time navigator) + signal_dim 2 (the VI map). We open it as its own
    signal tree (reusing the navigator/signal/selector machinery), then place the
    detector ROI on the SOURCE vectors DP via :meth:`run_into_tree`, which keeps
    replacing the tree's root data as the ROI moves."""
    import hyperspy.api as hs

    session = ctx.session
    src_tree = ctx.plot.signal_tree
    vecs = src_tree.diffraction_vectors

    # Initial stack from a default detector ROI (centred disk, kernel-sized) so
    # the window isn't blank before the first drag. run_into_tree's forced compute
    # immediately overwrites this with the actual placed-ROI image.
    n_t = max(1, vecs.n_time)
    nav_y, nav_x = vecs.nav_shape
    stack = np.zeros((n_t, nav_y, nav_x), dtype=np.float32)

    vi_sig = hs.signals.Signal2D(stack)
    base_title = src_tree.root.metadata.get_item("General.title", "Vectors")
    vi_sig.metadata.General.title = f"{base_title} — Vector Virtual Image ({color})"

    # Calibrate: the VI nav axis is the stack/time axis; its SIGNAL axes are the
    # real-space scan grid (the source's spatial navigation axes — the last two
    # nav axes of a 5-D dataset).
    try:
        src_nav = src_tree.root.axes_manager.navigation_axes
        # src_nav = (t, scan_y, scan_x); time → VI nav axis, scan_y/x → VI signal.
        if len(src_nav) >= 1:
            vi_sig.axes_manager.navigation_axes[0].scale = src_nav[0].scale
            vi_sig.axes_manager.navigation_axes[0].offset = src_nav[0].offset
            vi_sig.axes_manager.navigation_axes[0].units = src_nav[0].units
            vi_sig.axes_manager.navigation_axes[0].name = src_nav[0].name or "stack"
        for i, src_ax in enumerate(src_nav[1:]):
            if i < len(vi_sig.axes_manager.signal_axes):
                tgt = vi_sig.axes_manager.signal_axes[i]
                tgt.scale, tgt.offset = src_ax.scale, src_ax.offset
                tgt.units, tgt.name = src_ax.units, src_ax.name
    except Exception as e:
        log.debug("calibrating VI result axes failed: %s", e)

    vi_tree = session._add_signal(vi_sig)

    # Tag the result window(s) with the ROI colour so the renderer can match it.
    try:
        npm = getattr(vi_tree, "navigator_plot_manager", None)
        for pw in (list(npm.all_plot_windows) if npm else []):
            pw.vi_color = color
    except Exception as e:
        log.debug("tagging VI result window colour failed: %s", e)

    # Park the result navigator on the slice the source is currently viewing.
    try:
        t0 = int(src_tree.root.axes_manager.indices[0])
        t0 = int(np.clip(t0, 0, n_t - 1))
        vi_tree.root.axes_manager.indices = (t0,)
    except Exception as e:
        log.debug("parking VI navigator on source slice failed: %s", e)

    return act.run_into_tree(vi_tree, type=vtype, calculation=calc)


def add_vector_virtual_image(ctx, action_name: str = "Add Vector Virtual Image", **params):
    """Add ONE more vector virtual image (multi-VI sub-toolbar, Qt parity): a
    colour-cycled detector ROI on the vectors diffraction pattern → its own
    output window that recomputes live from the vectors as the ROI is dragged.
    Cycles red→…→magenta and is listed as a removable chip in the Vector Virtual
    Imaging sub-toolbar."""
    from spyde.backend.ipc import emit

    plot = ctx.plot
    session = ctx.session
    if getattr(getattr(plot, "signal_tree", None), "diffraction_vectors", None) is None:
        # Find Vectors may still be attaching (its batch finalizes on a worker
        # thread) — wait it out and re-dispatch instead of erroring in the gap.
        from spyde.backend.ipc import emit_error
        from spyde.actions.lifecycle import wait_for_vectors
        if wait_for_vectors(session, plot,
                            lambda: add_vector_virtual_image(
                                ctx, action_name=action_name, **params),
                            what="Vector Virtual Imaging", strict=True):
            return None
        emit_error("Vector Virtual Imaging: this signal has no diffraction vectors")
        return None

    items = getattr(plot, "_vi_items", [])
    n = len(items)
    color = VI_COLORS[n % len(VI_COLORS)]

    vtype = params.get("type", "disk")
    calc = params.get("calculation", "intensity")
    act = VectorVirtualImageAction(ctx)
    act.roi_color = color

    vecs = plot.signal_tree.diffraction_vectors
    if getattr(vecs, "n_time", 0) > 0:
        # 5-D: spawn a navigable 3-D VI result window (navigator = stack/time,
        # signal = the per-slice VI map), then drive it from the source ROI.
        selector = _spawn_navigable_vi(ctx, act, vtype, calc, color)
    else:
        selector = act.run(type=vtype, calculation=calc)

    vi_name = f"Vector Image {n + 1} ({color})"
    out_wids = sorted({
        c.window_id for c in getattr(selector, "active_children", [])
        if getattr(c, "window_id", None) is not None
    })
    # Same chip schema as raw VI (`type`/`calculation`) so the SubToolbar shape
    # icon + per-VI caret are reused verbatim; `parent_action` routes caret edits
    # (update_vi) to the Vector Virtual Imaging bar, not the raw one.
    items.append({"name": vi_name, "color": color, "type": vtype,
                  "calculation": calc, "out_wids": out_wids,
                  "parent_action": "Vector Virtual Imaging"})
    plot._vi_items = items

    src_wid = getattr(plot, "window_id", None)
    if src_wid is not None and session is not None:
        session._action_artifacts[(src_wid, vi_name)] = {
            "selector": selector, "out_wids": out_wids, "vi_source": src_wid,
            "action": act,
        }
        emit({
            "type": "sub_item", "window_id": src_wid, "action": "Vector Virtual Imaging",
            "name": vi_name, "color": color, "vtype": vtype,
            "calculation": calc, "active": True,
        })
    return None
