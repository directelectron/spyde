"""
virtual_image.py — virtual imaging as a RegionAction (the template reference).

Demonstrates how a previously ~470-line Qt action collapses into a small
declaration on top of :class:`~spyde.actions.action.RegionAction`.  Works
unchanged in the Electron app and in a Jupyter notebook::

    from spyde.actions.virtual_image import VirtualImageAction
    VirtualImageAction.for_plot(plot).run(type="disk", calculation="mean")

The interactive flow (place ROI → live recompute → display) is provided by the
RegionAction template + anyplotlib selector; only :meth:`reduce` and the
selector choice are action-specific.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.action import RegionAction
from spyde.actions.masks import widget_to_mask

log = logging.getLogger(__name__)


class VirtualImageAction(RegionAction):
    """Integrate a 4D-STEM dataset over a detector region → a navigation image."""

    name = "Virtual Image"
    output_dims = 2
    output_node_name = "Virtual Image"

    parameters = {
        "type": {
            "name": "Detector Type",
            "type": "enum",
            "default": "disk",
            "options": ["annular", "disk", "rectangle"],
        },
        "calculation": {
            "name": "Calculation",
            "type": "enum",
            "default": "mean",
            "options": ["mean", "sum"],
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

    def placeholder_signal(self):
        """The VI output is a NAVIGATION-space image, so the placeholder must
        already be nav-shaped — otherwise the figure is built for a 10x10 dummy
        and the (differently-shaped) result never replaces it on screen (the
        "VI is just black" placeholder, axes stuck at 0-9)."""
        import hyperspy.api as hs
        try:
            nav_shape = tuple(int(n) for n in self.signal.axes_manager.navigation_shape)
            shape = tuple(reversed(nav_shape))[-2:]   # numpy order, last 2 nav dims
            if len(shape) == 2 and all(s > 0 for s in shape):
                return hs.signals.Signal2D(np.zeros(shape, dtype=np.float32))
        except Exception as e:
            log.debug("building VI placeholder from nav shape failed: %s", e)
        return super().placeholder_signal()

    def _virtual_image_array(self, signal, selector, **params):
        """Return the (uncomputed) virtual-image array for the current ROI, or
        None if there's no ROI yet."""
        widget = getattr(selector, "roi", None)
        if widget is None:
            return None
        mask = np.ascontiguousarray(widget_to_mask(widget, signal), dtype=np.float32)

        # Contract the detector mask per chunk WITHOUT materialising the
        # ``data * mask`` product: einsum accumulates in place, so a task's
        # only intermediates are the source chunk + the tiny nav output. The
        # old ``(data * mask).sum(...)`` allocated a full float copy of every
        # chunk (2-4x the chunk bytes) — ~36 of those in flight was the "VI
        # spills GiBs to disk" pathology on uint16 data.
        def _masked_sum(block):
            return np.einsum("...ij,ij->...", block, mask).astype(
                np.float32, copy=False)

        data = signal.data
        if hasattr(data, "chunks"):
            import dask.array as da
            if len(data.chunks[-1]) == 1 and len(data.chunks[-2]) == 1:
                # Storage-aligned chunking (whole frames per chunk — the app's
                # loading contract): one einsum per chunk.
                vi = da.map_blocks(
                    _masked_sum, data, dtype=np.float32,
                    drop_axis=(data.ndim - 2, data.ndim - 1),
                )
            else:
                # Signal axes are split across chunks (foreign/odd data): the
                # per-block mask slice bookkeeping isn't worth it — fall back
                # to the broadcast product and let dask handle alignment.
                vi = (data * mask).sum(axis=(-2, -1))
        else:
            vi = _masked_sum(data)

        if params.get("calculation", "mean") == "mean":
            norm = float(mask.sum())
            if norm > 0:
                vi = vi / norm       # nav-sized — the cheap truediv
        return vi

    def reduce(self, signal, selector, indices, **params):
        """Compute the virtual image for the current detector ROI.

        ``(data * mask).sum(over signal axes)`` — optionally normalised by the
        mask area for a mean. Numpy fast-path; for lazy signals this returns a
        Future (the progressive streaming lives in :meth:`reduce_to`).
        """
        vi = self._virtual_image_array(signal, selector, **params)
        if vi is None:
            return None
        client = getattr(self.signal_tree, "client", None)
        if getattr(signal, "_lazy", False) and client is not None:
            # A lazy VI is a FULL-dataset reduction (sum over the detector for
            # every nav position). Dragging the ROI must NOT pile these up — each
            # reads the whole dataset. Cancel the previous (now superseded) compute
            # before submitting the new one so the cluster isn't clogged.
            prev = getattr(self, "_prev_vi_future", None)
            tree = self.signal_tree
            if prev is not None:
                try:
                    if not prev.done():
                        prev.cancel()
                except Exception as e:
                    log.debug("cancelling prior VI future failed: %s", e)
                if tree is not None and hasattr(tree, "unregister_cancel"):
                    tree.unregister_cancel(future=prev)
            fut = client.compute(vi)
            self._prev_vi_future = fut
            # Register on the tree so closing it mid-fill cancels this VI compute
            # (dragging the ROI supersedes+unregisters the prior one above).
            if tree is not None and hasattr(tree, "register_cancel"):
                tree.register_cancel(future=fut)
            return fut
        if hasattr(vi, "compute"):
            return vi.compute()
        return np.asarray(vi)

    def reduce_to(self, signal, selector, child, indices, **params):
        """Compute the virtual image for the live selector flow.

        Lazy data with a client STREAMS through the windowed progressive
        compute (bounded in-flight chunks, ROI-move cancellable — the old
        monolithic ``client.compute`` let the scheduler load the entire source
        dataset and spill GiBs, and painted nothing until the very end). The
        stream OWNS the child display and this returns ``None``, which the
        selector skips (``base_selector``: ``if new_data is None: continue``)
        — that is what fixes the historical clobbered-back-to-blank bug that
        forced the earlier per-chunk stream to be reverted: nothing else
        pushes to the child while the stream paints.

        Numpy data keeps the synchronous :meth:`reduce` path.
        """
        client = getattr(self.signal_tree, "client", None)
        if getattr(signal, "_lazy", False) and client is not None:
            vi = self._virtual_image_array(signal, selector, **params)
            if vi is None:
                return None
            from spyde.drawing.update_functions import stream_progressive_to_plot
            stream_progressive_to_plot(child, vi, client, name="vi")
            return None
        return self.reduce(signal, selector, indices, **params)


# Multi virtual-image colour cycle — SAME palette/order as the Qt toolbar.
VI_COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta"]


def virtual_imaging(ctx, action_name: str = "Virtual Imaging", **kwargs):
    """Parent submenu action — a no-op; the toolbar opens its sub-toolbar
    ("Add Virtual Image") instead of dispatching this."""
    return None


def add_virtual_image(ctx, action_name: str = "Add Virtual Image", **params):
    """Add ONE more virtual image (Qt-parity multi-VI sub-toolbar): a colour-cycled
    ROI on the diffraction pattern → its own output window that recomputes live.
    Each call cycles red→green→blue→yellow→cyan→magenta and is listed as a chip in
    the Virtual Imaging sub-toolbar (so you can add several and remove them)."""
    from spyde.backend.ipc import emit

    plot = ctx.plot
    session = ctx.session
    items = getattr(plot, "_vi_items", [])
    n = len(items)
    color = VI_COLORS[n % len(VI_COLORS)]

    vtype = params.get("type", "disk")
    calc = params.get("calculation", "mean")
    act = VirtualImageAction(ctx)
    act.roi_color = color
    selector = act.run(type=vtype, calculation=calc)

    vi_name = f"Virtual Image {n + 1} ({color})"
    out_wids = sorted({
        c.window_id for c in getattr(selector, "active_children", [])
        if getattr(c, "window_id", None) is not None
    })
    items.append({"name": vi_name, "color": color, "type": vtype, "calculation": calc,
                  "out_wids": out_wids, "parent_action": "Virtual Imaging"})
    plot._vi_items = items

    # Track (incl. the action for caret edits) + list it in the sub-toolbar.
    src_wid = getattr(plot, "window_id", None)
    if src_wid is not None and session is not None:
        session._action_artifacts[(src_wid, vi_name)] = {
            "selector": selector, "out_wids": out_wids, "vi_source": src_wid,
            "action": act,
        }
        emit({
            "type": "sub_item", "window_id": src_wid, "action": "Virtual Imaging",
            "name": vi_name, "color": color, "vtype": vtype, "calculation": calc,
            "active": True,
        })
    return None   # tracked manually; don't let _track_action_artifacts re-track


def vi_commit(session, plot, payload) -> None:
    """Commit a live virtual image (raw OR vector) to its own SignalTree — the
    standard Commit door (same pattern as strain). The live VI window + ROI
    stay open for further tuning; the committed tree is an independent
    snapshot that survives deselecting the action."""
    from spyde.backend.ipc import emit_error, emit_status

    name = (payload or {}).get("name")
    src_wid = (payload or {}).get("window_id")
    art = session._action_artifacts.get((src_wid, name)) if name else None
    if not art:
        emit_error(f"Commit: no live virtual image named {name!r}")
        return
    out_plot = None
    for owid in art.get("out_wids", []):
        p = session._plot_by_window_id(owid)
        if p is not None:
            out_plot = p
            break
    data = getattr(out_plot, "current_data", None) if out_plot is not None else None
    if data is None or not hasattr(data, "__array__"):
        emit_error("Commit: the virtual image hasn't finished computing yet")
        return
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        emit_error(f"Commit: unexpected virtual-image shape {data.shape}")
        return

    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    src_sig = getattr(tree, "root", None) if tree is not None else None
    src_title = ""
    if src_sig is not None:
        src_title = src_sig.metadata.get_item("General.title", "") or ""

    def _calibrate(new_tree):
        """Copy the SOURCE's spatial nav-axis calibration onto the committed
        image's signal axes (the VI output lives in navigation space)."""
        if src_sig is None:
            return
        try:
            nav_axes = list(src_sig.axes_manager.navigation_axes)[-2:]
            sig_axes = list(new_tree.root.axes_manager.signal_axes)
            for ax, ref in zip(sig_axes, nav_axes):
                ax.scale, ax.offset = ref.scale, ref.offset
                ax.units, ax.name = ref.units, ref.name
        except Exception as e:
            log.debug("calibrating committed VI axes failed: %s", e)

    from spyde.actions.commit import commit_result_tree
    commit_result_tree(
        session, title=name or "Virtual Image", primary=data, levels=None,
        provenance={"action": "Virtual Imaging", "item": name,
                    "source_title": src_title},
        on_tree=_calibrate,
    )
    emit_status(f"Committed {name} to a new signal tree")
