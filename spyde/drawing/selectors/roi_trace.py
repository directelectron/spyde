"""Diagnostics for an integrating ROI that MOVES ON ITS OWN.

The symptom is "the ROI jumps": the box shifts or resizes to somewhere the
pointer never went. It is hard to chase because it is intermittent and the
interesting state (what the renderer sent, what the clamp did to it, what
indices came out) is gone by the time anyone looks.

Two known mechanisms, and this distinguishes them:

* ``_clamp_extent``'s FALLBACK rewriting geometry mid-drag. It anchors on the
  lower edge, so it can move the edge the user is holding. Since anyplotlib
  >=0.4.1 the widget enforces the cap itself (``max_extent``) and the fallback
  is supposed to be dead code during a drag — if it fires, that assumption
  broke. (Measured 2026-07-26 on a calibrated 0.05 s/frame movie axis: cap
  correct at construction, widget clamps the drag itself, fallback never
  fires. So a hit here is genuinely new information.)
* The geometry arriving from the renderer already wrong — a pointer->data
  mapping problem, which shows up as the ROI landing somewhere unrelated to
  the pointer (observed once: a drag starting mid-plot left the span pinned at
  the last frame).

Design notes:

* **Always on, silent until something is actually wrong.** Requiring an env var
  means the user has to reproduce it twice — once to notice, once to capture.
  The anomaly rules below are deterministic (no thresholds on "how far is too
  far"), so they can run on every pointer_move without crying wolf. A fast mouse
  move legitimately translates the ROI a long way in one event, which is exactly
  why distance is NOT one of the rules.
* **The ring buffer is the point.** A single anomalous event says little; the
  handful of events LEADING UP to it say what the user was doing. On a hit the
  whole ring is dumped at WARNING, so the Log panel has the sequence.
* ``SPYDE_ROI_TRACE=1`` logs every observation at INFO for a deliberate
  reproduction session.

Costs a few float comparisons and a deque append per pointer event.
"""
from __future__ import annotations

import logging
import os
from collections import deque

logger = logging.getLogger(__name__)

RING = 8                 # events kept for context when an anomaly fires
_REL_TOL = 1e-3          # fraction of the span treated as "no change"


def _trace_all() -> bool:
    return os.environ.get("SPYDE_ROI_TRACE", "0").lower() in ("1", "true", "yes")


def _fmt(spans) -> str:
    return " ".join(f"[{lo:.4g},{hi:.4g}]" for lo, hi in spans)


class RoiTrace:
    """One per selector. Feed it every pointer event via :meth:`observe`.

    ``spans`` is a list of ``(lo, hi)`` per navigation axis — the 1-D span is
    ``[(x0, x1)]``, the 2-D rectangle is ``[(x, x+w), (y, y+h)]`` — so both
    selectors share one set of rules.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self._prev = None                       # spans AFTER the previous event
        self._ring: deque = deque(maxlen=RING)

    def observe(self, before, after, n_indices=None, n_unique=None,
                extra: str = "") -> None:
        """``before`` = geometry as the RENDERER just set it; ``after`` = the same
        geometry once ``_clamp_extent`` has had its say. Logs nothing unless a
        rule below fires (or SPYDE_ROI_TRACE=1)."""
        try:
            anomalies = self._check(before, after, n_indices, n_unique)
            entry = (f"{_fmt(before)}"
                     f"{'' if before == after else ' -> clamp ' + _fmt(after)}"
                     f" n={n_indices} uniq={n_unique}"
                     f"{(' ' + extra) if extra else ''}")
            self._ring.append(entry)
            self._prev = after
            if anomalies:
                logger.warning(
                    "[ROI-JUMP] %s: %s\n  %s", self.label, ", ".join(anomalies),
                    "\n  ".join(self._ring))
            elif _trace_all():
                logger.info("[ROI-TRACE] %s: %s", self.label, entry)
        except Exception as e:            # diagnostics must never break a drag
            logger.debug("roi trace failed: %s", e)

    @staticmethod
    def _moved(a, b) -> bool:
        """Did the geometry change by more than float noise? Sized relative to the
        span, so it scales with a calibrated axis (0.05 s/frame) as well as with
        image pixels."""
        if len(a) != len(b):
            return True
        for (alo, ahi), (blo, bhi) in zip(a, b):
            tol = max(abs(ahi - alo), abs(bhi - blo), 1e-12) * _REL_TOL
            if abs(blo - alo) > tol or abs(bhi - ahi) > tol:
                return True
        return False

    # ── rules ────────────────────────────────────────────────────────────────
    def _check(self, before, after, n_indices, n_unique):
        out = []
        # 1. The fallback clamp rewrote geometry. With the widget-side cap this
        #    should be unreachable during a drag, so it is always worth a line.
        #    Compared with a tolerance, not exactly: a float round-trip through
        #    the widget can shift an edge by ~1e-16, which is not a jump and would
        #    otherwise fire on every event of a capped drag.
        if self._moved(before, after):
            out.append("clamp rewrote geometry (widget cap did not hold)")

        # 2. The ROI both MOVED and RESIZED in one event. Dragging an edge moves
        #    one end; dragging the body moves both by the SAME amount. Both ends
        #    moving by DIFFERENT amounts is neither, so the geometry did not come
        #    from a single pointer gesture.
        prev = self._prev
        if prev is not None and len(prev) == len(before):
            for ax, ((plo, phi), (blo, bhi)) in enumerate(zip(prev, before)):
                span = max(abs(phi - plo), abs(bhi - blo), 1e-12)
                tol = span * _REL_TOL
                dlo, dhi = blo - plo, bhi - phi
                if abs(dlo) > tol and abs(dhi) > tol and abs(dlo - dhi) > tol:
                    out.append(
                        f"axis{ax} moved AND resized in one event "
                        f"(dlo={dlo:+.4g} dhi={dhi:+.4g})")

        # 3. The whole ROI collapsed onto one nav position while still covering
        #    more than one — i.e. every index clamped to a data edge. This is the
        #    "span pinned at the last frame" shape, and it also makes the read do
        #    N times the work for one frame.
        if (n_indices is not None and n_unique is not None
                and n_indices > 1 and n_unique == 1):
            out.append(f"all {n_indices} indices collapsed onto one position "
                       f"(ROI ran off the end of the data?)")
        return out
