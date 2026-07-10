"""
_session_axes.py — AxesEditorMixin extracted from session.py.

Holds the axis-editing surface: emitting the axes table to the sidebar,
per-axis property edits, and the draggable "set origin" crosshair tool.

These methods reference ``self._plots``, ``self.signal_trees`` etc. which are
initialised in ``Session.__init__`` — the mixin only USES ``self.<attr>``.
"""
from __future__ import annotations

import logging

from spyde.backend import ipc

log = logging.getLogger(__name__)


class AxesEditorMixin:
    def _emit_axes(self, tree) -> None:
        try:
            from spyde.metadata_extract import build_axes_list
            ipc.emit({
                "type": "axes_info",
                "window_ids": self._tree_window_ids(tree),
                "axes": build_axes_list(tree),
            })
        except Exception as e:
            log.warning("axes emit failed: %s", e)

    def _set_axis(self, plot, payload: dict) -> None:
        """Edit one axis property of the active window's root signal and
        recalibrate every plot in its tree. Writes back to the real
        axes_manager so the change is reflected in the dataset."""
        if plot is None:
            return
        tree = getattr(plot, "signal_tree", None)
        if tree is None:
            return
        index = payload.get("index")
        field = payload.get("field")
        value = payload.get("value")
        if index is None or field not in ("name", "units", "scale", "offset"):
            return
        try:
            axes = tree.root.axes_manager._axes
            if not (0 <= int(index) < len(axes)):
                return
            ax = axes[int(index)]
            if field == "scale":
                try:
                    new_scale = float(value)
                except (TypeError, ValueError):
                    return  # ignore non-numeric input mid-typing
                # Keep the ORIGIN PIXEL fixed when the scale changes: the pixel
                # where data == 0 is pixel0 = -offset/scale; to pin that same
                # pixel under the new scale, offset must scale with it:
                #   offset_new = offset_old * (scale_new / scale_old).
                # So the (0,0) point (e.g. the crosshair-marked centre) does not
                # drift when the user recalibrates the pixel size.
                old_scale = float(ax.scale)
                old_offset = float(ax.offset)
                ax.scale = new_scale
                if old_scale != 0.0:
                    ax.offset = old_offset * (new_scale / old_scale)
            elif field == "offset":
                try:
                    ax.offset = float(value)
                except (TypeError, ValueError):
                    return  # ignore non-numeric input mid-typing
            else:
                setattr(ax, field, str(value))
        except Exception as e:
            log.warning("set_axis failed: %s", e)
            return

        # Recalibrate: re-push every plot in the tree (re-reads the axes →
        # updated scale bar / extent) and re-emit the table + metadata. A
        # navigator plot reads the ROOT's navigation axes directly on repaint
        # (Plot._axes_info / _axes_info_1d branch on is_navigator), so editing a
        # navigation axis here reaches the navigator panel via this same re-push
        # — no separate mirror onto the derived navigator signal is needed.
        for p in list(self._plots):
            if getattr(p, "signal_tree", None) is tree:
                try:
                    p.update()
                except Exception as e:
                    log.debug("re-emitting plot update failed: %s", e)
        self._emit_axes(tree)
        try:
            from spyde.metadata_extract import build_metadata_dict
            ipc.emit({
                "type": "metadata",
                "window_ids": self._tree_window_ids(tree),
                "metadata": build_metadata_dict(tree),
            })
        except Exception as e:
            log.debug("re-emitting metadata failed: %s", e)

    def _set_title(self, plot, payload: dict) -> None:
        """Rename the dataset (the breadcrumb's [Name] segment). Writes the root
        signal's ``General.title`` — shared by the signal AND navigator windows of
        the tree — then re-applies the in-panel title strip and emits a
        lightweight ``window_title`` update to every window of the tree (no figure
        re-emit, so the iframe doesn't reload)."""
        if plot is None:
            return
        tree = getattr(plot, "signal_tree", None)
        if tree is None:
            return
        title = str(payload.get("title", "")).strip()
        if not title:
            return
        try:
            tree.root.metadata.set_item("General.title", title)
        except Exception as e:
            log.warning("set_title failed: %s", e)
            return
        # Re-apply the in-panel title strip on every plot of the tree.
        for p in list(self._plots):
            if getattr(p, "signal_tree", None) is tree:
                try:
                    p._apply_plot_title()
                except Exception as e:
                    log.debug("re-applying plot title failed: %s", e)
        ipc.emit({
            "type": "window_title",
            "window_ids": self._tree_window_ids(tree),
            "title": title,
        })

    def _set_offset_crosshair(self, plot, payload: dict) -> None:
        """Toggle a draggable "set origin" crosshair on the ACTIVE plot.

        The crosshair edits the offsets of the axes the active plot is drawn
        against, so it reads (0, 0) at the crosshair position:
          • signal plot    → the two SIGNAL axes' offsets
          • navigator plot → the two NAVIGATION axes' offsets
        Offsets are in real (calibrated) units; the tool starts at the current
        origin so it begins at the existing offset.

        payload {"on": True}  → drop the crosshair and update offsets as it moves.
                {"on": False} → remove the crosshair.
        """
        if plot is None:
            return
        tree = getattr(plot, "signal_tree", None)
        if tree is None:
            return
        on = bool(payload.get("on", False))
        plot2d = getattr(plot, "_plot2d", None)

        # always clear any existing crosshair first (idempotent). Keyed per-plot
        # so a signal-plot tool and a navigator-plot tool don't clobber each
        # other; store on the plot, not the shared tree.
        old = getattr(plot, "_offset_cross", None)
        if old is not None:
            # remove_widget() deletes the widget AND re-pushes the panel, so the
            # crosshair disappears on the FIRST toggle-off. A bare widget.hide()
            # only emits a targeted event that a later repaint overwrites, so the
            # ROI lingered until a second click (the reported "needs 2x").
            try:
                if plot2d is not None and hasattr(plot2d, "remove_widget"):
                    plot2d.remove_widget(old)
                else:
                    old.hide()
            except Exception as e:
                log.debug("removing offset crosshair failed: %s", e)
                try:
                    old.hide()
                except Exception:
                    pass
            plot._offset_cross = None
        if not on:
            return
        if plot2d is None:
            return

        # The axes the ACTIVE plot is drawn against: navigation axes for a
        # navigator, signal axes otherwise (mirrors Plot._axes_info / scale bar).
        try:
            if getattr(plot, "is_navigator", False):
                edit_ax = tree.root.axes_manager.navigation_axes
            else:
                edit_ax = plot.plot_state.current_signal.axes_manager.signal_axes
        except Exception as e:
            log.debug("offset crosshair axes lookup failed: %s", e)
            return
        if len(edit_ax) < 2:
            return
        ax_x, ax_y = edit_ax[0], edit_ax[1]
        w, h = int(ax_x.size), int(ax_y.size)
        # Start the crosshair on the PIXEL that is currently the origin
        # (data == 0): pixel = -offset/scale. anyplotlib's 2-D widget takes
        # pixel coordinates directly (it does NOT apply the axis scale/offset),
        # so we hand it the pixel, not a data coord. If the origin is off-image,
        # fall back to the image centre.
        def _origin_pixel():
            sx, ox = float(ax_x.scale), float(ax_x.offset)
            sy, oy = float(ax_y.scale), float(ax_y.offset)
            pxi = (-ox / sx) if sx else w / 2.0
            pyi = (-oy / sy) if sy else h / 2.0
            if not (0 <= pxi <= w and 0 <= pyi <= h):
                pxi, pyi = w / 2.0, h / 2.0
            return pxi, pyi
        cx0, cy0 = _origin_pixel()
        try:
            cross = plot2d.add_crosshair_widget(cx=cx0, cy=cy0, color="#ffae57")
        except Exception as e:
            log.debug("offset crosshair add failed: %s", e)
            return
        plot._offset_cross = cross

        # Capture the scale at toggle-on time as the FIXED reference. The widget
        # reports its position in PIXELS (unaffected by the offset we mutate each
        # move), so there's no offset feedback to anchor against — only the scale
        # matters for offset_new = -pixel * scale. Kept as a dict for symmetry
        # with the per-move apply below.
        ref = {"sx": float(ax_x.scale), "sy": float(ax_y.scale)}

        def _apply(final: bool):
            # The crosshair's cx/cy ARE the pixel it sits on (image-pixel coords);
            # set each offset so that pixel maps to data 0: offset_new =
            # -pixel * scale. Stable across repeated move events.
            try:
                sx, sy = ref["sx"], ref["sy"]
                px = float(cross.cx)
                py = float(cross.cy)
                ax_x.offset = -px * sx
                ax_y.offset = -py * sy
            except Exception as e:
                log.debug("offset crosshair update failed: %s", e)
                return
            # Live: re-emit the axes table so the dock shows the new offsets as
            # the user drags.  Defer the HOST-plot re-push (which rewrites the
            # displayed extent, and would shift the widget under the cursor) to
            # pointer-up so dragging stays smooth.  Only the host plot is
            # re-pushed — NOT every plot in the tree: re-pushing a navigator that
            # is progressively filling clobbers its live buffer, and editing one
            # plot's axes doesn't change the other's calibration.
            self._emit_axes(tree)
            if final:
                try:
                    plot.update()
                except Exception as e:
                    log.debug("re-pushing host plot after offset set failed: %s", e)
                # No reference re-anchor needed: the widget reports ABSOLUTE pixel
                # coordinates, which the host-plot re-push (a relabel of the
                # axes) leaves unchanged. A subsequent drag's offset is computed
                # from the new absolute pixel directly.

        def _on_event(event=None):
            etype = getattr(event, "type", None) or getattr(event, "name", None)
            _apply(final=(etype == "pointer_up"))

        try:
            cross.add_event_handler(_on_event, "pointer_move", "pointer_up")
        except Exception as e:
            log.debug("offset crosshair handler bind failed: %s", e)
        # Emit the current axes once so the dock reflects the starting state, but
        # do NOT mutate the offset at toggle-on: the crosshair already starts at
        # the existing origin, so the offset is unchanged until the user drags.
        self._emit_axes(tree)
