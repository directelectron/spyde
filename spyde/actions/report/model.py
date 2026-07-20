"""
model.py — the Report data model and the ``.spyde-report`` container format.

A report is a document of ordered **cells**: markdown text interleaved with
**figure** snapshots (and, in a template, empty figure **placeholders**). The
on-disk form is a single portable zip whose contents are deliberately
human-readable — plain markdown + YAML, **no JSON anywhere**:

    report.md            # THE document: YAML front-matter + markdown body
    figures/<id>.yaml    # a FigureSpec (recipe) per figure cell
    assets/<id>.png      # the baked WYSIWYG snapshot per figure cell

``report.md`` is valid standalone markdown (pandoc-ready when unzipped):

* A **figure cell** is a standalone-paragraph image ref
  ``![<caption>](assets/<id>.png)`` — the basename IS the cell id, the alt text
  IS the caption, and the live recipe lives at ``figures/<id>.yaml``.
* A **template placeholder** is an HTML comment
  ``<!-- spyde:placeholder <id> <caption> -->`` (invisible in any external
  markdown renderer; SpyDE draws it as a dashed drop zone).
* Everything else is markdown cells.

Parsing and serialization ROUND-TRIP: user markdown containing inline images,
non-spyde HTML comments, and code fences with image-ref lookalikes must survive
unchanged. Only a *standalone-paragraph* image whose target is
``assets/<id>.png`` counts as a figure cell.

The schema is the FULL one (single-panel/single-layer is Phase 1's subset) so
Phase 2 (combined figures, MDI layering) reuses it without migration.
"""
from __future__ import annotations

import base64
import binascii
import io
import os
import re
import logging
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# ── cell-id + marker helpers ──────────────────────────────────────────────────


def new_cell_id() -> str:
    """A short, unique, filesystem-safe cell id (``c`` + 8 hex chars)."""
    return "c" + uuid.uuid4().hex[:8]


def _tree_uid(tree, create: bool = True) -> "str | None":
    """A per-tree stable id used to rebind a figure to its source WITHIN a
    session (survives a save→reload while the tree is still open — the "match
    open trees first" key, robust when the signal has no file / title). Stored on
    the tree as ``_spyde_report_uid``. ``create=False`` reads without assigning
    (so ``resolve`` never mints ids on unrelated trees)."""
    if tree is None:
        return None
    uid = getattr(tree, "_spyde_report_uid", None)
    if uid is None and create:
        uid = "t" + uuid.uuid4().hex[:12]
        try:
            tree._spyde_report_uid = uid
        except Exception:
            return None
    return uid


def _signal_shape(sig) -> "list | None":
    """The signal's data shape as a plain list of ints (for rebind
    disambiguation), or None if it can't be read."""
    if sig is None:
        return None
    try:
        shp = getattr(getattr(sig, "data", None), "shape", None)
        if shp is None:
            return None
        return [int(x) for x in shp]
    except Exception:
        return None


def _plot_root_shape(plot) -> "list | None":
    """The shape of the ROOT signal of a plot's tree (matches the shape stored on
    a SignalRef by :meth:`SignalRef.from_plot`, which reads the current signal at
    snapshot time). Used only to disambiguate same-title trees during rebind."""
    try:
        sig = plot.plot_state.current_signal
        return _signal_shape(sig)
    except Exception:
        return None


# A standalone-paragraph image ref whose target is ``assets/<id>.png``. The
# basename (without extension) is the cell id; the alt text is the caption.
_FIG_LINE_RE = re.compile(
    r"^!\[(?P<caption>.*?)\]\(assets/(?P<cid>[^)/]+)\.png\)\s*$")
# The image-file extensions a PHOTO/IMAGE cell may use. A figure cell is ALWAYS
# ``.png`` (with a sibling ``figures/<id>.yaml``); an IMAGE cell serializes as an
# ``assets/<id>.<ext>`` ref with NO figures yaml. A NON-png image ext parses
# straight to an image cell (a figure never uses jpg/gif/webp); a ``.png`` image
# is disambiguated from a figure by the yaml-presence check in :func:`read_report`
# (a bare ``.png`` ref defaults to a figure cell here, back-compat).
IMAGE_EXTS = ("png", "jpg", "jpeg", "gif", "webp")
_NONPNG_EXT_ALT = "|".join(e for e in IMAGE_EXTS if e != "png")
# A standalone-paragraph image ref whose target is ``assets/<id>.<non-png-ext>``
# — an IMAGE cell (a photo dropped/pasted/browsed in). ``.png`` refs are handled
# by ``_FIG_LINE_RE`` above (default figure; promoted to image on missing yaml).
_IMAGE_LINE_RE = re.compile(
    r"^!\[(?P<caption>.*?)\]\(assets/(?P<cid>[^)/]+)\."
    r"(?P<ext>" + _NONPNG_EXT_ALT + r")\)\s*$")
# A template placeholder comment: ``<!-- spyde:placeholder <id> [caption] -->``.
_PLACEHOLDER_RE = re.compile(
    r"^<!--\s*spyde:placeholder\s+(?P<cid>\S+)(?:\s+(?P<caption>.*?))?\s*-->\s*$")
# A movie-cell marker: ``<!-- spyde:movie <id> -->`` — an invisible comment BEFORE
# a standalone poster image ref (``assets/<id>.png``). The marker CREATES the movie
# cell; ONLY an immediately-following ``.png`` ref whose id MATCHES fills its poster
# (mirrors the split marker). A movie marker with no matching poster is a
# placeholder movie (no source assigned yet). The ``movies/<id>.yaml`` sibling
# (attached by read_report) supplies the MovieSpec.
_MOVIE_RE = re.compile(r"^<!--\s*spyde:movie\s+(?P<cid>\S+)\s*-->\s*$")
# A slide-break marker (Present mode): ``<!-- spyde:slide-break -->`` — an
# invisible comment BEFORE the cell that starts a new slide.
_SLIDE_BREAK_RE = re.compile(r"^<!--\s*spyde:slide-break\s*-->\s*$")
# A "go live" excursion marker: ``<!-- spyde:live-action <yaml-flow> -->`` — a
# small YAML flow mapping (e.g. ``{tutorial: strain, guide: strain}``) BEFORE
# the cell it applies to.
_LIVE_ACTION_RE = re.compile(
    r"^<!--\s*spyde:live-action\s+(?P<payload>.*?)\s*-->\s*$")
# A per-slide KIND marker (Present mode / slides): ``<!-- spyde:slide-kind title -->``
# — an invisible comment BEFORE the slide's FIRST cell (the slide-break cell)
# declaring the WHOLE slide a title/section slide (rendered as a large centered
# title block). ``content`` / absence = a normal slide. Anything else → ""
# (content). Mirrors the slide-break marker exactly.
_SLIDE_KIND_RE = re.compile(r"^<!--\s*spyde:slide-kind\s+(?P<val>\S+)\s*-->\s*$")
# A per-slide STYLE marker (Present mode / slides): ``<!-- spyde:slide-style plain -->``
# — an invisible comment BEFORE the slide's FIRST cell picking a background/heading
# preset for the WHOLE slide. ``default`` / absence = the standard dark stage;
# ``plain`` = a flat darker stage; ``accent`` = a subtle accent-tinted gradient.
# Anything else → "" (default). Mirrors the slide-kind marker.
_SLIDE_STYLE_RE = re.compile(r"^<!--\s*spyde:slide-style\s+(?P<val>\S+)\s*-->\s*$")
# A per-slide SPEAKER-NOTES marker (Present mode presenter view):
# ``<!-- spyde:notes <base64> -->`` — an invisible comment BEFORE the slide's FIRST
# cell carrying the speaker's private notes for the WHOLE slide (shown only in the
# presenter view / never to the audience). Notes are free multi-line text with
# markdown + unicode, so they are base64-encoded (utf-8 → standard base64) inside
# the single-line comment: this survives newlines / ``-->`` / special chars while
# staying a valid, invisible HTML comment. Absent → "" (older files). ``_encode_notes``
# / ``_decode_notes`` own the round trip.
_SLIDE_NOTES_RE = re.compile(r"^<!--\s*spyde:notes\s+(?P<b64>\S+)\s*-->\s*$")
# The accepted slide kinds/styles (anything else → "" == default). "content" and
# "" are equivalent (a normal slide); "default" and "" pick the standard stage.
_SLIDE_KINDS = ("title",)
_SLIDE_STYLES = ("plain", "accent")
# A SPLIT-cell marker (the split-block primitive): ``<!-- spyde:split <layout>
# <id> <text-b64> -->`` — an invisible comment that IS a split cell (one atomic
# block: a text side + a figure/photo side, side by side). It carries the layout,
# the cell id, and the TEXT side base64-encoded INTO the marker (``"="`` sentinel
# for empty text — always a 4th token). Encoding the text keeps it fully OPAQUE:
# a standalone-image-ref lookalike inside the text can never fragment the cell or
# be mis-bound as the figure side. The ONLY thing that may follow the marker is
# the split's OWN figure/image ref (id == ``<id>``), which fills the figure side.
# ``<layout>`` ∈ {"text-left", "text-right"}. Absent marker → not a split cell.
_SPLIT_RE = re.compile(
    r"^<!--\s*spyde:split\s+(?P<layout>\S+)\s+(?P<cid>\S+)\s+(?P<text>\S+)\s*-->\s*$")
# The accepted split layouts (anything else → "text-left", the default). Which
# side the TEXT sits on relative to the figure/photo: left / right (side by side)
# or top / bottom (stacked).
_SPLIT_LAYOUTS = ("text-left", "text-right", "text-top", "text-bottom")


def _normalize_split_layout(val) -> str:
    """Normalise a raw split layout to one of :data:`_SPLIT_LAYOUTS`. Any
    unknown/absent value collapses to ``"text-left"`` (the default — text on the
    left, figure on the right) so a malformed marker still renders sanely."""
    s = str(val or "").strip().lower()
    return s if s in _SPLIT_LAYOUTS else "text-left"


# The accepted document types (anything else / absent → "report"). "movie" is
# reserved (not built in Wave A) but tolerated so a hand-authored future file
# doesn't fail to load.
_DOC_TYPES = ("report", "presentation", "movie")


def _normalize_doc_type(val) -> str:
    """Normalise a raw document type to one of ``{"report", "presentation",
    "movie"}``. Any unknown/absent value collapses to ``"report"`` (a scrolling
    article — the default and every legacy document) so an older report — which
    has no ``type:`` front-matter — loads exactly as before."""
    s = str(val or "").strip().lower()
    return s if s in _DOC_TYPES else "report"


def _normalize_slide_kind(val) -> str:
    """Normalise a raw slide-kind to one of ``{"", "title"}``. ``content`` and any
    unknown/absent value collapse to ``""`` (a normal content slide) so an older
    report — which has no slide-kind markers — renders exactly as before."""
    s = str(val or "").strip().lower()
    return s if s in _SLIDE_KINDS else ""


def _normalize_slide_style(val) -> str:
    """Normalise a raw slide-style to one of ``{"", "plain", "accent"}``.
    ``default`` and any unknown/absent value collapse to ``""`` (the standard dark
    stage) so an older report loads exactly as before."""
    s = str(val or "").strip().lower()
    return s if s in _SLIDE_STYLES else ""


def _encode_notes(notes) -> str:
    """utf-8 speaker notes → a single-line, comment-safe base64 token (standard
    base64, no newlines). Empty / whitespace-only notes → ``""`` (the marker is
    then omitted entirely). Base64 is bulletproof for the free-text notes: it
    survives newlines, markdown syntax, ``-->`` sequences and unicode inside an
    HTML comment."""
    s = str(notes or "")
    if not s.strip():
        return ""
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _decode_notes(token) -> str:
    """A base64 notes token (from :func:`_encode_notes`) → the utf-8 notes string.
    Tolerant: a malformed / non-base64 token decodes to ``""`` rather than raising,
    so a hand-edited or corrupt marker can never break a report load."""
    t = str(token or "").strip()
    if not t:
        return ""
    try:
        return base64.b64decode(t, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return ""


# The split marker's TEXT token uses base64 with a single-char sentinel for the
# empty string, so the token is ALWAYS present + non-blank (a bare "=" is not a
# valid base64 payload, so it can never collide with real encoded content).
_SPLIT_EMPTY = "="


def _b64_text(text) -> str:
    """utf-8 → a single-line, comment-safe base64 token; the empty string →
    ``"="`` (a sentinel) so the split marker always carries a 4th token."""
    s = str(text or "")
    if s == "":
        return _SPLIT_EMPTY
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _unb64_text(token) -> str:
    """Inverse of :func:`_b64_text`. The ``"="`` sentinel and any malformed token
    → ``""`` (tolerant)."""
    t = str(token or "").strip()
    if t == "" or t == _SPLIT_EMPTY:
        return ""
    try:
        return base64.b64decode(t, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return ""


# ── the spec dataclasses (full schema; Phase 1 uses single-panel/single-layer) ─


@dataclass
class SignalRef:
    """Provenance for rebinding a figure to a live signal on report open.

    ``file_path`` + ``fingerprint`` (size/mtime) identify the source file;
    ``tree_uid`` is a per-tree stable id that survives a save/reload WITHIN a
    session (the "match open trees first" key — works even when the signal has
    no file / title, e.g. synthetic data); ``tree_node`` names the node within
    the tree; ``view`` / ``title`` are the displayed view label + panel title at
    snapshot time. Any field may be None (an in-memory / test signal has no
    file_path)."""
    file_path: str | None = None
    fingerprint: dict | None = None            # {"size": int, "mtime": float}
    tree_uid: str | None = None
    tree_node: str | None = None
    view: str | None = None
    title: str | None = None
    shape: list | None = None                  # signal data shape (disambiguates
                                               # same-title trees on rebind)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "fingerprint": (dict(self.fingerprint)
                            if self.fingerprint is not None else None),
            "tree_uid": self.tree_uid,
            "tree_node": self.tree_node,
            "view": self.view,
            "title": self.title,
            "shape": (list(self.shape) if self.shape is not None else None),
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "SignalRef":
        d = d or {}
        fp = d.get("fingerprint")
        shp = d.get("shape")                   # tolerate absent on older files
        return cls(
            file_path=d.get("file_path"),
            fingerprint=(dict(fp) if isinstance(fp, dict) else None),
            tree_uid=d.get("tree_uid"),
            tree_node=d.get("tree_node"),
            view=d.get("view"),
            title=d.get("title"),
            shape=(list(shp) if isinstance(shp, (list, tuple)) else None),
        )

    @classmethod
    def from_plot(cls, plot) -> "SignalRef":
        """Build a SignalRef from a live ``Plot`` (its tree + current signal)."""
        tree = getattr(plot, "signal_tree", None)
        file_path = getattr(tree, "source_path", None) if tree is not None else None
        fingerprint = fingerprint_file(file_path)
        tree_uid = _tree_uid(tree)
        # The displayed node name: prefer the current signal's General.title, then
        # the plot's view label / navigator flag.
        tree_node = None
        title = None
        shape = None
        try:
            sig = plot.plot_state.current_signal
            title = str(sig.metadata.get_item("General.title", default="") or "")
            shape = _signal_shape(sig)
        except Exception:
            title = None
        if tree is not None:
            try:
                root_title = str(tree.root.metadata.get_item(
                    "General.title", default="") or "")
                tree_node = root_title or None
            except Exception:
                tree_node = None
        view = getattr(plot, "view_label", None)
        return cls(file_path=file_path, fingerprint=fingerprint,
                   tree_uid=tree_uid, tree_node=tree_node, view=view,
                   title=title or tree_node, shape=shape)

    def resolve(self, session) -> "Any | None":
        """Find the live plot this ref points at in *session*, or None.

        Match strategy (per the plan): open trees first — by ``tree_uid`` (stable
        within a session), then by root/node title — then by file fingerprint.
        Returns a ``Plot`` (whose ``current_data`` the caller re-snapshots) or
        None when the source is offline."""
        if session is None:
            return None
        plots = list(getattr(session, "_plots", []) or [])
        candidates: list = []
        # 1a) open trees by stable tree_uid (survives an in-session save/reload).
        if self.tree_uid:
            for p in plots:
                tree = getattr(p, "signal_tree", None)
                if tree is not None and _tree_uid(tree, create=False) == self.tree_uid:
                    candidates.append(p)
        # 1b) open trees by root/node title (with shape disambiguation) OR by
        # source file path. Title matching is fuzzy — two DIFFERENT datasets can
        # share a title — so we require a shape match when the ref recorded one,
        # and if title matching still lands on more than one DISTINCT tree we
        # treat the source as unresolved (offline) rather than guessing wrong.
        if not candidates:
            title_candidates: list = []
            title_trees: list = []             # distinct trees matched by title
            for p in plots:
                tree = getattr(p, "signal_tree", None)
                if tree is None:
                    continue
                try:
                    root_title = str(tree.root.metadata.get_item(
                        "General.title", default="") or "")
                except Exception:
                    root_title = ""
                if self.tree_node and root_title and root_title == self.tree_node:
                    # Require a shape match when both sides carry a shape; if the
                    # ref has a shape but the plot can't report one, don't match on
                    # title alone (fall through to fingerprint / offline).
                    if self.shape is not None:
                        cand_shape = _plot_root_shape(p)
                        if cand_shape is None or list(cand_shape) != list(self.shape):
                            continue
                    title_candidates.append(p)
                    if tree not in title_trees:
                        title_trees.append(tree)
                elif (self.file_path is not None
                      and getattr(tree, "source_path", None) == self.file_path):
                    candidates.append(p)
            # Ambiguous: title (+shape) matched more than one distinct tree → the
            # ref can't be pinned to a single source, so stay offline.
            if len(title_trees) <= 1:
                candidates.extend(title_candidates)
        # 2) file fingerprint fallback (a report reopened in a fresh session).
        if not candidates and self.file_path is not None:
            for p in plots:
                tree = getattr(p, "signal_tree", None)
                sp = getattr(tree, "source_path", None) if tree is not None else None
                if sp is not None and sp == self.file_path:
                    fp = fingerprint_file(sp)
                    if self.fingerprint is None or fp == self.fingerprint:
                        candidates.append(p)
        if not candidates:
            return None
        # Prefer a non-navigator plot (the diffraction pattern / image itself).
        for p in candidates:
            if not getattr(p, "is_navigator", False):
                return p
        return candidates[0]


def new_layer_id() -> str:
    """A short, unique layer id (``l`` + 6 hex chars) — addresses a LayerSpec in a
    panel for the Phase-2 compose edit handlers (repfig_set_layer / _remove_layer)."""
    return "l" + uuid.uuid4().hex[:6]


@dataclass
class LayerSpec:
    """One image (or line) layer within a panel. ``source`` is a SignalRef;
    ``>1`` layer per panel = an overlay (Phase 2). ``id`` addresses the layer for
    the compose edit handlers.

    ``tint`` is a ``#rgb``/``#rrggbb`` hex string that renders the layer as a
    clear→colour intensity ramp (anyplotlib's tint LUT) instead of a named
    colormap; ``None`` keeps colormap display. ``cmap`` is ALWAYS stored (even
    while tinted) — it's the revert value when the tint is cleared. Tolerant
    round-trip: ``to_dict`` emits the ``tint`` key ONLY when set and
    ``from_dict`` reads it with ``.get()``, so an older file without the key
    loads as ``None`` and renders exactly as before (SCHEMA_VERSION stays 1).

    ``color`` / ``linewidth`` / ``label`` are LINE-PANEL curve styling
    (``PanelSpec.kind == "line"``): ``color`` a CSS colour string for the
    curve, ``linewidth`` the stroke width in px, ``label`` the legend entry
    text (anyplotlib draws a legend automatically once any line carries a
    non-empty label). Unused / ignored on an image layer. Tolerant round-trip
    like ``tint``: each is emitted in ``to_dict`` ONLY when set (non-None) and
    read with ``.get()`` in ``from_dict``, so an older file without the keys
    loads every one as ``None`` (figure_builder falls back to anyplotlib's own
    defaults) — SCHEMA_VERSION stays 1."""
    source: SignalRef = field(default_factory=SignalRef)
    cmap: str = "viridis"
    clim: list | None = None                   # [lo, hi] or None (auto)
    alpha: float = 1.0
    visible: bool = True
    tint: str | None = None                    # "#rrggbb" ramp, None = cmap mode
    color: str | None = None                   # line-panel curve colour
    linewidth: float | None = None             # line-panel stroke width (px)
    label: str | None = None                   # line-panel legend label
    id: str = field(default_factory=new_layer_id)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "source": self.source.to_dict(),
            "cmap": self.cmap,
            "clim": (list(self.clim) if self.clim is not None else None),
            "alpha": float(self.alpha),
            "visible": bool(self.visible),
        }
        # Emitted only when set — old files/readers never see the key.
        if self.tint:
            d["tint"] = str(self.tint)
        if self.color is not None:
            d["color"] = str(self.color)
        if self.linewidth is not None:
            d["linewidth"] = float(self.linewidth)
        if self.label is not None:
            d["label"] = str(self.label)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayerSpec":
        d = d or {}
        clim = d.get("clim")
        tint = d.get("tint")                   # tolerate absent on older files
        color = d.get("color")
        linewidth = d.get("linewidth")
        label = d.get("label")
        return cls(
            source=SignalRef.from_dict(d.get("source")),
            cmap=d.get("cmap", "viridis"),
            clim=(list(clim) if clim is not None else None),
            alpha=float(d.get("alpha", 1.0)),
            visible=bool(d.get("visible", True)),
            tint=(str(tint) if tint else None),
            color=(str(color) if color is not None else None),
            linewidth=(float(linewidth) if linewidth is not None else None),
            label=(str(label) if label is not None else None),
            id=d.get("id") or new_layer_id(),
        )


@dataclass
class PanelSpec:
    """One panel (axes cell) of a figure: ≥1 layer, calibrated axes, annotations,
    and decorations. Phase 1 uses a single panel with a single image layer.

    ``kind`` is ``"image"`` | ``"line"`` | ``"scene3d"``. A ``scene3d`` panel is
    a 3-D scatter scene (the IPF sphere explorer): its pixels are NOT a
    LayerSpec image — the point cloud lives in the backend snapshot map keyed
    ``(panel_id, "xyz")`` / ``(panel_id, "rgb")`` (float/uint8 arrays, never in
    this spec / YAML / report_state). ``scene`` holds the SMALL recompute
    parameters only:

        {"kind": "ipf3d", "direction": "x|y|z", "point_size": float,
         "bounds": [[lo,hi],[lo,hi],[lo,hi]], "camera"?: {...}}

    A scene3d panel still carries ONE LayerSpec whose ``source`` SignalRef
    points at the orientation-result tree — that's the rebind/refresh handle
    (the layer itself paints nothing). Tolerant round-trip like ``tint``:
    ``to_dict`` emits ``scene`` only when set, ``from_dict`` reads it with
    ``.get()`` — older files without the key load as ``None`` and render
    exactly as before (SCHEMA_VERSION stays 1).

    ``text_sizes`` holds per-element font-size overrides in CSS pixels, keys
    among ``{"title","x_label","y_label","ticks","legend","colorbar"}``
    mapping to an int size. Absent/``None`` keeps anyplotlib's own defaults.
    Tolerant round-trip like ``scene``: ``to_dict`` emits ``text_sizes`` only
    when set, ``from_dict`` reads it with ``.get()``."""
    id: str = "p1"
    grid_pos: list = field(default_factory=lambda: [0, 0])
    kind: str = "image"                        # "image" | "line" | "scene3d"
    layers: list = field(default_factory=list)         # [LayerSpec]
    axes: dict | None = None                   # {units, scale:[sy,sx], offset:[oy,ox]}
    annotations: list = field(default_factory=list)    # [dict] (marker kwargs)
    scalebar: bool = False
    colorbar: bool = False
    title: str = ""
    insets: list = field(default_factory=list)         # [dict] (Phase 2)
    scene: dict | None = None                  # scene3d recompute params (small)
    text_sizes: dict | None = None             # {title|x_label|y_label|ticks|legend|colorbar: int}

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "grid_pos": list(self.grid_pos),
            "kind": self.kind,
            "layers": [ly.to_dict() for ly in self.layers],
            "axes": (dict(self.axes) if self.axes is not None else None),
            "annotations": [dict(a) for a in self.annotations],
            "scalebar": bool(self.scalebar),
            "colorbar": bool(self.colorbar),
            "title": self.title,
            "insets": [dict(i) for i in self.insets],
        }
        # Emitted only when set — old files/readers never see the key.
        if self.scene:
            d["scene"] = dict(self.scene)
        if self.text_sizes:
            d["text_sizes"] = dict(self.text_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PanelSpec":
        d = d or {}
        scene = d.get("scene")                 # tolerate absent on older files
        text_sizes = d.get("text_sizes")        # tolerate absent on older files
        return cls(
            id=d.get("id", "p1"),
            grid_pos=list(d.get("grid_pos", [0, 0])),
            kind=d.get("kind", "image"),
            layers=[LayerSpec.from_dict(x) for x in (d.get("layers") or [])],
            axes=(dict(d["axes"]) if d.get("axes") is not None else None),
            annotations=[dict(a) for a in (d.get("annotations") or [])],
            scalebar=bool(d.get("scalebar", False)),
            colorbar=bool(d.get("colorbar", False)),
            title=d.get("title", ""),
            insets=[dict(i) for i in (d.get("insets") or [])],
            scene=(dict(scene) if isinstance(scene, dict) else None),
            text_sizes=(dict(text_sizes) if isinstance(text_sizes, dict) else None),
        )


@dataclass
class FigureSpec:
    """The full recipe for a figure cell — ONE schema shared by report cells,
    the combined-figure editor, and MDI layering. Phase 1 emits
    ``layout={kind:single}`` with one panel / one layer.

    ``annotations`` are FIGURE-LEVEL markers (distinct from a panel's
    ``PanelSpec.annotations``): each dict is in the EXACT anyplotlib
    figure-marker schema — positions/sizes in FIGURE FRACTIONS (0..1, top-left
    origin), NO calibration/data-coord conversion — so they ride straight into
    ``Figure.set_figure_markers``. Absent on older files → ``[]``.

    ``vectors_mode`` records the user's drop-time choice for a source tree that
    carries diffraction vectors: ``"viewer"`` embeds the interactive explorer in
    HTML exports, ``"image"`` forces the static snapshot. ``""`` (older files /
    non-vectors sources) keeps the viewer-when-available default."""
    layout: dict = field(default_factory=lambda: {"kind": "single"})
    panels: list = field(default_factory=list)          # [PanelSpec]
    nav_context: dict | None = None            # {"indices": [iy, ix]}
    annotations: list = field(default_factory=list)     # [dict] figure-fraction markers
    vectors_mode: str = ""                     # "" | "viewer" | "image"

    def to_dict(self) -> dict:
        d = {
            "layout": dict(self.layout),
            "panels": [p.to_dict() for p in self.panels],
            "nav_context": (dict(self.nav_context)
                            if self.nav_context is not None else None),
            "annotations": [dict(a) for a in self.annotations],
        }
        if self.vectors_mode:
            d["vectors_mode"] = self.vectors_mode
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FigureSpec":
        d = d or {}
        return cls(
            layout=dict(d.get("layout") or {"kind": "single"}),
            panels=[PanelSpec.from_dict(x) for x in (d.get("panels") or [])],
            nav_context=(dict(d["nav_context"])
                         if d.get("nav_context") is not None else None),
            annotations=[dict(a) for a in (d.get("annotations") or [])],
            vectors_mode=str(d.get("vectors_mode", "") or ""),
        )

    def to_yaml(self) -> str:
        return _dump_yaml(self.to_dict())

    @classmethod
    def from_yaml(cls, text: str) -> "FigureSpec":
        return cls.from_dict(yaml.safe_load(text) or {})

    # convenience for the single-panel/single-layer Phase-1 path
    @property
    def primary_layer(self) -> "LayerSpec | None":
        for p in self.panels:
            if p.layers:
                return p.layers[0]
        return None


@dataclass
class MovieSpec:
    """The recipe for a MOVIE cell — an editable, persistent in-situ movie block.

    A movie cell references a live in-situ signal (``source``, a
    :class:`SignalRef` rebound on report open exactly like a figure layer) and
    carries the render/edit state the full-screen Movie editor mutates: the base
    render ``params`` (fps / spatial downsample / temporal stride / cmap / clim /
    timestamp / scalebar / time range), a list of time-gated ``annotations``
    (text / rect / circle / arrow — ROIs are just persistent rect/circle
    annotations, so there is NO separate rois list; the pipeline's
    ``_draw_annotations`` is the single draw path), ``text_overlays`` (a 1-D
    signal's live value painted as text, e.g. ``"T = 812.3 °C"``), ``freezes``
    (hold on a frame for a duration), an optional ``overlay_image`` (a 2nd image
    composited over the base — Phase 3), a ``crop`` rect (source signal px), and
    an ``out_size`` (final output px).

    Serialized to ``movies/<id>.yaml`` in the ``.spyde-report`` zip (the
    yaml-presence at ``movies/`` — vs ``figures/`` — is what marks a cell a movie
    on reload), alongside a baked poster ``assets/<id>.png``. All positions/sizes
    are in the ORIGINAL source frame's pixel space (the pipeline divides by the
    downsample factor at draw time). Every field tolerates absence (older /
    hand-authored files) so a partial spec never fails to load."""
    source: SignalRef | None = None
    params: dict = field(default_factory=dict)          # fps/downsample/stride/cmap/
    #                                                     clim/timestamp/scalebar/t_start/t_end
    annotations: list = field(default_factory=list)     # [dict] time-gated markers (incl. ROIs)
    text_overlays: list = field(default_factory=list)   # [dict] 1-D-signal-as-text overlays
    freezes: list = field(default_factory=list)         # [{"t":int,"hold_s":float}] (legacy)
    speed_segments: list = field(default_factory=list)  # [{"time_range":[s0,s1],"speed":float}]
    overlay_image: dict | None = None                   # 2nd-image composite (Phase 3)
    crop: list | None = None                            # [x0,y0,x1,y1] source px, or None
    out_size: list | None = None                        # [w,h] output px, or None

    def to_dict(self) -> dict:
        d: dict = {
            "source": (self.source.to_dict() if self.source is not None else None),
            "params": dict(self.params),
            "annotations": [dict(a) for a in self.annotations],
            "text_overlays": [dict(t) for t in self.text_overlays],
            "freezes": [dict(f) for f in self.freezes],
            "speed_segments": [dict(s) for s in self.speed_segments],
        }
        if self.overlay_image is not None:
            d["overlay_image"] = dict(self.overlay_image)
        if self.crop is not None:
            d["crop"] = [int(v) for v in self.crop]
        if self.out_size is not None:
            d["out_size"] = [int(v) for v in self.out_size]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MovieSpec":
        d = d or {}
        src = d.get("source")
        crop = d.get("crop")
        out_size = d.get("out_size")
        oi = d.get("overlay_image")
        return cls(
            source=(SignalRef.from_dict(src) if isinstance(src, dict) else None),
            params=dict(d.get("params") or {}),
            annotations=[dict(a) for a in (d.get("annotations") or [])],
            text_overlays=[dict(t) for t in (d.get("text_overlays") or [])],
            freezes=[dict(f) for f in (d.get("freezes") or [])],
            speed_segments=[dict(s) for s in (d.get("speed_segments") or [])],
            overlay_image=(dict(oi) if isinstance(oi, dict) else None),
            crop=([int(v) for v in crop] if crop else None),
            out_size=([int(v) for v in out_size] if out_size else None),
        )

    def to_yaml(self) -> str:
        return _dump_yaml(self.to_dict())

    @classmethod
    def from_yaml(cls, text: str) -> "MovieSpec":
        return cls.from_dict(yaml.safe_load(text) or {})


@dataclass
class Cell:
    """A document cell.

    ``cell_type`` is ``"markdown"``, ``"figure"``, ``"image"``, ``"split"``, or
    ``"movie"``. A figure cell carries a ``caption``, a ``fig_id`` (the FigureSpec
    / asset basename == this cell's id), a ``spec`` (FigureSpec, in memory), and —
    for a template — a ``placeholder`` flag when no figure has been dropped yet.

    A ``"movie"`` cell is an editable, persistent in-situ movie block: it carries
    a ``caption``, a ``movie`` (:class:`MovieSpec` — the source SignalRef + the
    render/edit state), a baked poster PNG asset (a representative frame, keyed by
    the cell id like a figure's baked snapshot), and — until a source signal is
    assigned — a ``placeholder`` flag (an empty "pick a signal" drop zone).
    Persisted in report.md as an invisible ``<!-- spyde:movie <id> -->`` marker
    before a standalone poster image ref; the live recipe lives at
    ``movies/<id>.yaml`` (that ``movies/`` sibling — vs ``figures/`` — is how a
    movie cell is told from a figure/photo on reload). Absent marker → NOT a movie
    cell (fully back-compatible; SCHEMA_VERSION stays 1).

    A ``"split"`` cell (Wave A — the split-block primitive) is ONE atomic block
    holding a TEXT side + a FIGURE/PHOTO side, side by side. It REUSES the existing
    fields: ``source`` is the text side's markdown; the figure/photo side is the
    SAME machinery a figure/image cell uses — a ``spec`` (FigureSpec) + its snapshot
    (like a figure cell) OR ``image_ext`` + held bytes (like an image cell). So all
    the existing snapshot / asset / export code works on a split cell unchanged; the
    split cell just ALSO renders its ``source`` text beside the figure/photo. Until
    a figure/photo is dropped the figure side is empty (``spec is None`` and no
    ``image_ext``) — an empty drop zone. ``split_layout`` (``"text-left"`` /
    ``"text-right"``) picks which side the TEXT sits on (the figure takes the
    other). Persisted in report.md as an invisible ``<!-- spyde:split <layout>
    <id> -->`` marker BEFORE the block: the text block THEN the figure/image ref
    both bind to the marker's cell id, so the two sides re-associate to the one
    split cell unambiguously (see :func:`serialize_report_md` /
    :func:`_parse_body_cells`). Absent marker → NOT a split cell (fully
    back-compatible — SCHEMA_VERSION stays 1).

    An ``"image"`` cell is a plain PHOTO the user dropped / pasted / browsed in:
    it carries a ``caption`` and its raw image bytes are stored as an asset like a
    figure, at ``assets/<id>.<image_ext>`` (``image_ext`` ∈ :data:`IMAGE_EXTS`).
    It has NO ``spec`` and NO sibling ``figures/<id>.yaml`` — that yaml-presence
    distinction (a ``.png`` ref WITH a figures yaml = figure; an image ref WITHOUT
    one = image) is how :func:`read_report` tells the two apart on load, so an
    older figure-only report parses unchanged.

    ``html`` is a DERIVED, NON-PERSISTED field: the renderer's own
    marked+DOMPurify-sanitized rendering of ``source`` (delivered on every
    markdown commit). It is used ONLY by HTML export — never written into
    report.md / the zip, and absent after a reload until the next edit. HTML
    export falls back to escaping the raw markdown when it's empty.

    ``slide_break`` (Phase 6 — Present mode) marks a cell as the START of a new
    slide when the report is presented / exported as slides: cells accumulate
    onto the current slide until the next cell with ``slide_break=True`` (see
    :meth:`ReportDoc.slides`). Persisted in report.md as an invisible HTML
    comment ``<!-- spyde:slide-break -->`` immediately BEFORE the cell's block
    (invisible in any external markdown renderer). Absent on older files → False
    (SCHEMA_VERSION stays 1). ``live_action`` (also Phase 6) is an OPTIONAL
    "go live" excursion handle — a small dict like
    ``{"tutorial": "<name>", "guide": "<id>"}`` that Present mode turns into a
    "Launch live ▶" button; persisted as ``<!-- spyde:live-action <yaml-flow> -->``
    before the cell. Absent → None.

    ``slide_kind`` / ``slide_style`` (Present mode / slides — presentation POLISH)
    are PER-SLIDE attributes carried on the slide's FIRST cell (the slide_break
    cell). ``slide_kind`` ``""`` (content, default) renders the slide as today;
    ``"title"`` makes the WHOLE slide a TITLE / SECTION slide — its markdown
    renders as a large vertically+horizontally centered title block (the first
    line big, the rest a muted subtitle). ``slide_style`` picks a per-slide
    background/heading preset: ``""``/``default`` the standard dark stage,
    ``"plain"`` a flat darker stage, ``"accent"`` a subtle accent-tinted gradient.
    Only the slide's FIRST cell's values matter (Present mode + export read them
    to style the whole slide); on a non-first cell they're harmless/ignored.
    Persisted as invisible ``<!-- spyde:slide-kind title -->`` /
    ``<!-- spyde:slide-style accent -->`` comments before the cell; absent on
    older files → ``""`` (SCHEMA_VERSION stays 1).

    ``notes`` (Present mode presenter view) are the slide's SPEAKER NOTES — free
    multi-line markdown text the speaker sees in the presenter view but the
    audience NEVER does. Like ``slide_kind`` / ``slide_style`` they are a PER-SLIDE
    attribute carried on the slide's FIRST cell (read via :func:`slide_meta` /
    :func:`slide_notes`); on a non-first cell they're harmless/ignored. Persisted
    as an invisible ``<!-- spyde:notes <base64> -->`` comment before the cell —
    base64 so the free text (newlines / markdown / ``-->`` / unicode) round-trips
    inside a single-line HTML comment and NEVER leaks into any external markdown
    renderer. Absent on older files → ``""`` (SCHEMA_VERSION stays 1)."""
    id: str = field(default_factory=new_cell_id)
    cell_type: str = "markdown"
    source: str = ""                           # markdown text (markdown cells)
    caption: str = ""                          # figure / image caption / alt text
    placeholder: bool = False
    spec: FigureSpec | None = None             # figure recipe (figure cells)
    movie: "MovieSpec | None" = None           # movie recipe (movie cells)
    image_ext: str = ""                        # image cells: "png"/"jpg"/… (asset ext)
    html: str = ""                             # derived, NON-persisted (export only)
    spec_error: str = ""                       # derived, NON-persisted: read_report
    #                                          # stamps the reason a figures/<id>.yaml
    #                                          # failed to parse (spec dropped to the
    #                                          # baked PNG); surfaced to the user, never
    #                                          # written back to the zip.
    slide_break: bool = False                  # Present mode: starts a new slide
    live_action: dict | None = None            # Present mode: "go live" excursion
    slide_kind: str = ""                       # Present mode: "" (content) | "title"
    slide_style: str = ""                      # Present mode: "" | "plain" | "accent"
    notes: str = ""                            # Present mode: per-slide speaker notes
    split_layout: str = "text-left"            # split cells: "text-left" | "text-right"


@dataclass
class ReportDoc:
    """The in-memory report: metadata + an ordered list of :class:`Cell`.

    ``doc_type`` (Wave A — the document TYPE field) is one of ``"report"``
    (a scrolling article — the default and every existing document),
    ``"presentation"`` (a slide deck), or ``"movie"`` (reserved, not built yet).
    It mirrors ``template``: serialized as a front-matter ``type:`` key, read back
    tolerantly (absent → ``"report"``), and never changes SCHEMA_VERSION."""
    title: str = "Untitled Report"
    template: bool = False
    doc_type: str = "report"
    version: int = SCHEMA_VERSION
    created: str = ""
    modified: str = ""
    cells: list = field(default_factory=list)  # [Cell]

    def __post_init__(self):
        now = _utcnow()
        if not self.created:
            self.created = now
        if not self.modified:
            self.modified = now

    # ── cell operations ────────────────────────────────────────────────────────

    def cell_by_id(self, cell_id: str) -> "Cell | None":
        for c in self.cells:
            if c.id == cell_id:
                return c
        return None

    def index_of(self, cell_id: str) -> int:
        for i, c in enumerate(self.cells):
            if c.id == cell_id:
                return i
        return -1

    def slides(self) -> "list[list[Cell]]":
        """Group ``cells`` into slides for Present mode / the slides export.

        A cell with ``slide_break=True`` STARTS a new slide; cells accumulate
        onto the current slide until the next break. The FIRST slide always
        begins with the first cell (a leading ``slide_break`` on cell 0 is a
        no-op — it can't split before the start). An empty document → ``[]``.
        Preserves cell order exactly; every cell lands in exactly one slide."""
        groups: list[list[Cell]] = []
        for c in self.cells:
            if c.slide_break and groups:
                groups.append([c])
            elif not groups:
                groups.append([c])
            else:
                groups[-1].append(c)
        return groups

    def touch(self) -> None:
        self.modified = _utcnow()


def slide_columns(cells: "list[Cell]") -> list:
    """Turn a SLIDE's cell list into an ordered list of ROWS for the slides
    layout, reused by the slides HTML export.

    Each returned row is one of:

    * ``{"kind": "full", "cell": <Cell>}`` — a full-width block, OR
    * ``{"kind": "split", "cell": <Cell>}`` — a self-contained SPLIT cell (Wave A):
      ONE cell that IS a 2-column block (text side + figure/photo side). The
      export renders its two sides side by side, ordered by the cell's
      ``split_layout`` (text-left / text-right).

    Rule: walk the cells in order; a ``cell_type=="split"`` cell emits a ``split``
    row; every other cell emits its own ``full`` row. Preserves order; every cell
    lands in exactly one row."""
    rows: list = []
    for c in cells:
        if c.cell_type == "split":
            rows.append({"kind": "split", "cell": c})
        else:
            rows.append({"kind": "full", "cell": c})
    return rows


def slide_meta(cells: "list[Cell]") -> dict:
    """The per-slide presentation attributes for a SLIDE's cell list — read off
    the slide's FIRST cell (the slide-break cell), which is where ``slide_kind`` /
    ``slide_style`` / ``notes`` are carried. Reused by Present mode + the slides
    HTML export (and mirrored in the renderer). An empty slide → the defaults.

    Returns ``{"kind": "" | "title", "style": "" | "plain" | "accent",
    "notes": "<str>"}`` — kind/style normalised (unknown / absent → ``""``),
    notes verbatim (absent → ``""``), so a legacy slide with no markers renders
    exactly as a content slide on the standard stage with no notes."""
    first = cells[0] if cells else None
    return {
        "kind": _normalize_slide_kind(first.slide_kind if first else ""),
        "style": _normalize_slide_style(first.slide_style if first else ""),
        "notes": first.notes if first else "",
    }


def slide_notes(cells: "list[Cell]") -> str:
    """The SPEAKER NOTES for a SLIDE's cell list — read off the slide's FIRST cell
    (where the per-slide ``notes`` are carried, like ``slide_kind`` / ``slide_style``).
    An empty slide / a slide with no notes → ``""``."""
    first = cells[0] if cells else None
    return first.notes if first else ""


def move_slide(cells: "list[Cell]", frm: int, to: int) -> "list[Cell]":
    """Reorder the flat ``cells`` list by moving WHOLE SLIDES (the block reorder
    behind ``report_move_slide``).

    A slide is a contiguous RUN of cells (see :meth:`ReportDoc.slides`): the run
    STARTS at a cell with ``slide_break=True`` and accumulates until the next
    break. ``move_slide`` extracts the whole cell-run for slide ``frm`` and
    re-inserts it so that, in the new order, it lands at slide POSITION ``to``.

    Returns a NEW list of the SAME Cell objects reordered (the cells themselves
    are mutated only to fix ``slide_break`` — see the invariant below); the input
    list is not modified. Out-of-range ``frm``/``to`` (or a no-op ``frm == to``)
    return a copy of ``cells`` unchanged.

    SLIDE-BREAK INVARIANT (kept correct across the move so ``slides()`` regroups
    into the SAME slides, just reordered):

    * ``slides()`` groups by ``slide_break`` but treats a leading break on cell 0
      as a no-op — the FIRST slide begins at cell 0 regardless. So after a move
      the deck could group WRONG unless we fix the two boundary cells:
      - The cell that becomes the NEW FIRST cell (index 0) does not NEED a
        ``slide_break`` (harmless if it has one), but the slide that USED to be
        first, once displaced, MUST gain ``slide_break=True`` on its first cell
        so it stays a distinct slide instead of merging into whatever precedes it.
      - Symmetrically, the moved slide's own first cell MUST carry
        ``slide_break=True`` unless it lands at index 0 (where it's the implicit
        first slide). We SET it on every non-index-0 slide-start so the grouping
        is unambiguous regardless of the source deck's leading-break state.
    * We therefore normalise ALL slide starts after the splice: every slide's
      first cell gets ``slide_break=True`` EXCEPT the slide at index 0 (whose
      leading break, if any, we leave untouched — it's a harmless no-op). This is
      idempotent and preserves the exact slide GROUPING while making the flags
      internally consistent."""
    # Group into slide blocks (list-of-lists), preserving object identity.
    blocks: list[list] = []
    for c in cells:
        if c.slide_break and blocks:
            blocks.append([c])
        elif not blocks:
            blocks.append([c])
        else:
            blocks[-1].append(c)

    n = len(blocks)
    if n == 0:
        return list(cells)
    if not (0 <= frm < n) or not (0 <= to < n) or frm == to:
        return list(cells)

    # Splice the moved block out, then re-insert at the target slide position.
    moved = blocks.pop(frm)
    # After the pop, indices at/after `frm` shifted down by one; `to` is the
    # DESIRED final slide position, so insert directly at `to` (clamped).
    to = max(0, min(to, len(blocks)))
    blocks.insert(to, moved)

    # Flatten and normalise slide-break flags so grouping stays exact.
    out: list = []
    for bi, block in enumerate(blocks):
        for ci, c in enumerate(block):
            if ci == 0:
                # First cell of each slide: MUST be a break unless it's the very
                # first slide (index 0), where a leading break is a no-op anyway.
                c.slide_break = bi != 0
            out.append(c)
    return out


# ── time / yaml helpers ───────────────────────────────────────────────────────


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dump_yaml(obj) -> str:
    """Dump plain dicts/lists to clean, human-readable YAML — NO python object
    tags, keys in insertion order, block style."""
    return yaml.safe_dump(obj, default_flow_style=False, sort_keys=False,
                          allow_unicode=True)


# ── report.md (de)serialization ───────────────────────────────────────────────


def serialize_report_md(doc: ReportDoc) -> str:
    """Serialize a :class:`ReportDoc` to the ``report.md`` text (YAML
    front-matter + markdown body). Inverse of :func:`parse_report_md`."""
    front = {
        "version": doc.version,
        "title": doc.title,
        "template": bool(doc.template),
        "type": _normalize_doc_type(getattr(doc, "doc_type", "report")),
        "created": doc.created,
        "modified": doc.modified,
    }
    parts = ["---\n", _dump_yaml(front), "---\n"]
    body_blocks: list[str] = []
    for c in doc.cells:
        # Present-mode markers ride as invisible comments in their OWN standalone
        # block (a blank line keeps them from being sucked into a markdown cell),
        # emitted BEFORE the cell they apply to. slide-break first, then the
        # per-slide kind/style (only meaningful on the slide's first cell, but
        # serialized wherever set), then any live-action, so parsing sees them in
        # a stable order.
        if c.slide_break:
            body_blocks.append("<!-- spyde:slide-break -->")
        kind = _normalize_slide_kind(c.slide_kind)
        if kind:
            body_blocks.append(f"<!-- spyde:slide-kind {kind} -->")
        style = _normalize_slide_style(c.slide_style)
        if style:
            body_blocks.append(f"<!-- spyde:slide-style {style} -->")
        notes_token = _encode_notes(c.notes)
        if notes_token:
            body_blocks.append(f"<!-- spyde:notes {notes_token} -->")
        if c.live_action:
            flow = yaml.safe_dump(dict(c.live_action), default_flow_style=True,
                                  sort_keys=True, allow_unicode=True).strip()
            body_blocks.append(f"<!-- spyde:live-action {flow} -->")
        if c.cell_type == "split":
            # A SPLIT block: an invisible marker carrying the layout, THIS cell's
            # id, and the TEXT side base64-encoded into the marker (opaque — a
            # lookalike in the text can never fragment the cell). The ONLY thing
            # that may follow is the split's OWN figure/photo ref (same id), which
            # fills the figure side: ``.png`` (a figure — has a sibling figures
            # yaml) OR ``.<ext>`` (a dropped photo — no yaml) OR omitted entirely
            # (empty drop zone until a figure/photo is dropped).
            layout = _normalize_split_layout(c.split_layout)
            # The text side is ALWAYS base64-encoded into the marker (a base64 of
            # "" is the empty string "" — still a 4th token position, just empty).
            # We emit "=" for empty so the token is always present and non-blank;
            # "=" is not valid standalone base64 content so it decodes to "".
            text_b64 = _b64_text(c.source or "")
            body_blocks.append(f"<!-- spyde:split {layout} {c.id} {text_b64} -->")
            if c.spec is not None:
                body_blocks.append(f"![{c.caption}](assets/{c.id}.png)")
            elif c.image_ext:
                ext = (c.image_ext or "png").lower()
                body_blocks.append(f"![{c.caption}](assets/{c.id}.{ext})")
        elif c.cell_type == "markdown":
            body_blocks.append(c.source.rstrip("\n"))
        elif c.cell_type == "image":
            # A photo: a standalone-paragraph image ref pointing at
            # ``assets/<id>.<ext>`` (NOT ``.png`` unless the photo IS a png), with
            # NO sibling figures yaml. alt text == caption.
            ext = (c.image_ext or "png").lower()
            body_blocks.append(f"![{c.caption}](assets/{c.id}.{ext})")
        elif c.cell_type == "figure":
            if c.placeholder:
                cap = (c.caption or "").strip()
                marker = f"<!-- spyde:placeholder {c.id}"
                marker += f" {cap} -->" if cap else " -->"
                body_blocks.append(marker)
            else:
                # Standalone-paragraph image ref; alt text == caption.
                body_blocks.append(f"![{c.caption}](assets/{c.id}.png)")
        elif c.cell_type == "movie":
            # An invisible movie marker (the ``movies/<id>.yaml`` sibling is what
            # makes it a movie on reload) THEN a standalone poster image ref (alt
            # text == caption). A placeholder movie (no source assigned yet) has no
            # poster — just the marker, so it re-parses as an empty movie cell.
            body_blocks.append(f"<!-- spyde:movie {c.id} -->")
            if not c.placeholder:
                body_blocks.append(f"![{c.caption}](assets/{c.id}.png)")
    # A blank line between blocks keeps each figure ref a standalone paragraph.
    body = "\n\n".join(body_blocks)
    return "".join(parts) + "\n" + body + ("\n" if body else "")


def parse_report_md(text: str) -> ReportDoc:
    """Parse ``report.md`` text back into a :class:`ReportDoc`.

    Splits YAML front-matter, then walks the body: standalone-paragraph
    ``assets/<id>.png`` image refs and ``spyde:placeholder`` comments become
    figure cells; everything else coalesces into markdown cells. Code fences are
    passed through verbatim (a fenced image-ref lookalike does NOT parse as a
    figure). Figure ``spec``s are attached later from ``figures/<id>.yaml``."""
    front, body = _split_front_matter(text)
    doc = ReportDoc(
        version=int(front.get("version", SCHEMA_VERSION)),
        title=str(front.get("title", "Untitled Report")),
        template=bool(front.get("template", False)),
        doc_type=_normalize_doc_type(front.get("type", "report")),
        created=str(front.get("created", "")),
        modified=str(front.get("modified", "")),
    )
    doc.cells = _parse_body_cells(body)
    return doc


def _split_front_matter(text: str) -> tuple[dict, str]:
    """Return (front_matter_dict, body). Front matter is a leading ``---`` /
    ``---`` fenced YAML block; absent → ({}, text)."""
    if text.startswith("---\n") or text.startswith("---\r\n"):
        # Find the closing fence line.
        lines = text.splitlines(keepends=True)
        end = None
        for i in range(1, len(lines)):
            if lines[i].rstrip("\r\n") == "---":
                end = i
                break
        if end is not None:
            fm_text = "".join(lines[1:end])
            body = "".join(lines[end + 1:])
            front = yaml.safe_load(fm_text) or {}
            if not isinstance(front, dict):
                front = {}
            return front, body.lstrip("\n")
    return {}, text


def _parse_body_cells(body: str) -> list:
    """Walk the markdown body line by line, tracking code-fence state so a
    fenced image-ref lookalike is never treated as a figure cell. Consecutive
    non-figure lines coalesce into one markdown cell."""
    cells: list = []
    md_buf: list[str] = []
    in_fence = False
    fence_char: str | None = None       # "`" or "~"
    fence_len = 0                        # opener run length (CommonMark)
    # Present-mode markers seen since the last cell was emitted — applied to the
    # NEXT cell created (markdown or figure). They flush the pending markdown
    # buffer first (so the marker starts a NEW cell rather than joining the one
    # already accumulating above it).
    pending: dict = {"slide_break": False, "live_action": None,
                     "slide_kind": "", "slide_style": "", "notes": ""}
    # SPLIT-cell parse state. A ``<!-- spyde:split <layout> <id> <text-b64> -->``
    # marker CREATES the split cell immediately (text decoded from the marker) and
    # sets ``open_split`` so ONLY an immediately-following figure/image ref WHOSE
    # ID MATCHES fills its figure side; a split with no matching ref is an
    # empty-figure-side split (a drop zone).
    open_split: "Cell | None" = None
    # MOVIE-cell parse state (mirrors open_split): a ``<!-- spyde:movie <id> -->``
    # marker CREATES the movie cell (placeholder=True) and sets ``open_movie`` so
    # ONLY an immediately-following ``.png`` ref whose id MATCHES fills its poster
    # (and clears placeholder); a movie with no matching poster stays a placeholder.
    open_movie: "Cell | None" = None

    def _apply_pending(cell: "Cell") -> "Cell":
        if pending["slide_break"]:
            cell.slide_break = True
        if pending["live_action"] is not None:
            cell.live_action = pending["live_action"]
        if pending["slide_kind"]:
            cell.slide_kind = pending["slide_kind"]
        if pending["slide_style"]:
            cell.slide_style = pending["slide_style"]
        if pending["notes"]:
            cell.notes = pending["notes"]
        pending["slide_break"] = False
        pending["live_action"] = None
        pending["slide_kind"] = ""
        pending["slide_style"] = ""
        pending["notes"] = ""
        return cell

    def flush_md() -> None:
        src = "\n".join(md_buf).strip("\n") if md_buf else ""
        md_buf.clear()
        # Drop a run that is nothing but blank lines (paragraph separators).
        if src.strip() != "":
            cells.append(_apply_pending(Cell(
                id=new_cell_id(), cell_type="markdown", source=src)))

    for raw in body.splitlines():
        line = raw.rstrip("\r")
        stripped = line.strip()
        # Fence tracking (``` or ~~~). Per CommonMark a fenced code block is
        # delimited by a run of ≥3 of the SAME char; the closing fence must use
        # the SAME char and be AT LEAST as long as the opener. Track the opener
        # char + length so a 4-backtick fence containing an inner ``` line does
        # NOT close early (which would corrupt round-trips + spawn phantom figure
        # cells from image-ref lookalikes inside the fence). Inside a fence
        # NOTHING is interpreted as a figure/placeholder.
        fence_open = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_open:
            run = fence_open.group(1)
            char = run[0]
            length = len(run)
            if not in_fence:
                in_fence = True
                fence_char = char
                fence_len = length
                md_buf.append(line)
                continue
            elif char == fence_char and length >= fence_len:
                # A closing fence: same char, length ≥ opener. An info string is
                # not allowed on a closing fence, so require the rest to be blank.
                if stripped[length:].strip() == "":
                    in_fence = False
                    fence_char = None
                    fence_len = 0
                    md_buf.append(line)
                    continue
            # A same-family fence line that does NOT close (too short, wrong char,
            # or carries an info string) is just fence content — fall through to
            # the in-fence passthrough below.
        if in_fence:
            md_buf.append(line)
            continue

        m_break = _SLIDE_BREAK_RE.match(stripped)
        m_kind = _SLIDE_KIND_RE.match(stripped)
        m_style = _SLIDE_STYLE_RE.match(stripped)
        m_notes = _SLIDE_NOTES_RE.match(stripped)
        m_live = _LIVE_ACTION_RE.match(stripped)
        m_split = _SPLIT_RE.match(stripped)
        m_movie = _MOVIE_RE.match(stripped)
        m_fig = _FIG_LINE_RE.match(stripped)
        m_image = _IMAGE_LINE_RE.match(stripped)
        m_ph = _PLACEHOLDER_RE.match(stripped)
        # A figure/image ref whose id MATCHES the open split's id is that split's
        # figure side; ANY other line — including a figure/image ref for a
        # DIFFERENT cell — closes the open split (an empty drop-zone split whose
        # figure side was never filled). Matching on the id (serialization always
        # writes the split's own id into its figure ref) is what prevents a split
        # from swallowing the NEXT cell's figure or a stray image-ref lookalike
        # inside its own text.
        _split_ref = None
        if open_split is not None:
            if m_fig is not None and m_fig.group("cid") == open_split.id:
                _split_ref = "fig"
            elif m_image is not None and m_image.group("cid") == open_split.id:
                _split_ref = "image"
        # A blank line between the split marker and its figure ref is just a
        # paragraph separator — it must NOT close the open split. Only a non-blank
        # line that isn't the split's own matching ref closes it (an empty
        # drop-zone split whose figure side was never written).
        if open_split is not None and _split_ref is None and stripped != "":
            open_split = None
        # An open movie's poster: a ``.png`` ref (matched by _FIG_LINE_RE) whose id
        # matches fills the poster + clears placeholder. A blank line is a paragraph
        # separator (keep it open); any other non-blank, non-matching line closes
        # the movie as a placeholder (no poster written yet).
        _movie_ref = (open_movie is not None and m_fig is not None
                      and m_fig.group("cid") == open_movie.id)
        if open_movie is not None and not _movie_ref and stripped != "":
            open_movie = None
        if m_split is not None:
            # A split marker CREATES the split cell now — layout + id + the text
            # side decoded from the marker (opaque). ``open_split`` then lets ONLY
            # the split's OWN matching-id figure/image ref (if any) fill the figure
            # side; a split with no matching ref is an empty drop zone.
            flush_md()
            cell = _apply_pending(Cell(
                id=(m_split.group("cid") or "").strip() or new_cell_id(),
                cell_type="split",
                source=_unb64_text(m_split.group("text")),
                split_layout=_normalize_split_layout(m_split.group("layout"))))
            cells.append(cell)
            open_split = cell
        elif m_movie is not None:
            # A movie marker CREATES the movie cell now (placeholder until a
            # matching poster ref fills it). ``open_movie`` lets ONLY the movie's
            # OWN matching-id ``.png`` poster ref (if any) clear the placeholder.
            flush_md()
            cell = _apply_pending(Cell(
                id=(m_movie.group("cid") or "").strip() or new_cell_id(),
                cell_type="movie", placeholder=True))
            cells.append(cell)
            open_movie = cell
        elif m_break is not None:
            # Ends the current markdown run; the marker applies to whatever cell
            # comes next.
            flush_md()
            pending["slide_break"] = True
        elif m_kind is not None:
            # A per-slide kind (title/content) applies to whatever cell comes
            # next (the slide's first cell); unknown → "" via _normalize.
            flush_md()
            pending["slide_kind"] = _normalize_slide_kind(m_kind.group("val"))
        elif m_style is not None:
            flush_md()
            pending["slide_style"] = _normalize_slide_style(m_style.group("val"))
        elif m_notes is not None:
            # Speaker notes (base64) apply to whatever cell comes next (the
            # slide's first cell). A malformed token decodes to "" (tolerant).
            flush_md()
            pending["notes"] = _decode_notes(m_notes.group("b64"))
        elif m_live is not None:
            flush_md()
            try:
                payload = yaml.safe_load(m_live.group("payload"))
            except yaml.YAMLError:
                # A malformed live-action marker (hand-edited flow mapping) is
                # ignored; a non-YAML error would be a parser bug worth surfacing.
                payload = None
            pending["live_action"] = (dict(payload)
                                      if isinstance(payload, dict) else None)
        elif m_image is not None:
            # A NON-png image ref is unambiguously an IMAGE cell (a figure is
            # always ``.png``). A ``.png`` image is matched by ``_FIG_LINE_RE``
            # below (default figure) and later promoted to an image cell by
            # ``read_report`` when it has no sibling figures yaml.
            flush_md()
            if _split_ref == "image":
                # Fill the open split cell's figure side with this PHOTO (its id
                # matches the split's — keep the id + text; adopt ext + caption).
                open_split.image_ext = (m_image.group("ext") or "").lower()
                open_split.caption = m_image.group("caption") or ""
                open_split = None
            else:
                cells.append(_apply_pending(Cell(
                    id=m_image.group("cid"), cell_type="image",
                    caption=m_image.group("caption") or "",
                    image_ext=(m_image.group("ext") or "").lower())))
        elif m_fig is not None:
            flush_md()
            if _movie_ref:
                # This ``.png`` ref is the open movie's POSTER (id matches). Adopt
                # its caption, clear the placeholder; the sibling ``movies/<id>.yaml``
                # (attached by read_report) supplies the MovieSpec. NOT a figure cell.
                open_movie.caption = m_fig.group("caption") or ""
                open_movie.placeholder = False
                open_movie = None
            elif _split_ref == "fig":
                # Fill the open split cell's figure side with this FIGURE (its id
                # matches the split's). The sibling ``figures/<id>.yaml`` (attached
                # by read_report) supplies the spec; keep the split id + text.
                open_split.caption = m_fig.group("caption") or ""
                open_split = None
            else:
                cells.append(_apply_pending(Cell(
                    id=m_fig.group("cid"), cell_type="figure",
                    caption=m_fig.group("caption") or "", placeholder=False)))
        elif m_ph is not None:
            flush_md()
            cells.append(_apply_pending(Cell(
                id=m_ph.group("cid"), cell_type="figure",
                caption=(m_ph.group("caption") or "").strip(),
                placeholder=True)))
        else:
            md_buf.append(line)
    flush_md()
    return cells


# ── fingerprint (nav_sidecar style) ───────────────────────────────────────────


def fingerprint_file(path: str | None) -> "dict | None":
    """Return ``{"size", "mtime"}`` for *path*, or None if it doesn't exist /
    isn't a real file. Matches the nav_sidecar size+mtime pattern."""
    if not path:
        return None
    try:
        st = os.stat(path)
        return {"size": int(st.st_size), "mtime": float(st.st_mtime)}
    except OSError:
        return None


# ── baked PNG fallback (headless save) ────────────────────────────────────────


def bake_line_fallback_png(y: np.ndarray, x_axis=None, *, color: str = "#4fc3f7",
                           linewidth: float = 1.5, label: "str | None" = None,
                           x_units: str = "", max_points: int = 4000) -> bytes:
    """Render a 1-D curve to PNG bytes with matplotlib Agg — the line-panel
    counterpart of :func:`bake_fallback_png`, used when a report cell's
    ``kind="line"`` panel has no renderer-harvested PNG (headless save).

    Draws an actual ``ax.plot(x, y)`` (not the 1-row-heatmap fallback
    ``bake_fallback_png`` uses for a stray 1-D array elsewhere) so the baked
    PNG reads as a real curve. Downsamples by striding when *y* is very long
    so a huge trace stays cheap; ``x_axis`` defaults to ``range(n)`` when
    omitted or length-mismatched. matplotlib is imported inside so it stays
    off the import path for callers that never save."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.shape[0]
    xa = np.asarray(x_axis, dtype=np.float64) if x_axis is not None else None
    if xa is None or xa.shape[0] != n:
        xa = np.arange(n, dtype=np.float64)

    if n > max_points:
        stride = int(np.ceil(n / max_points))
        xa = xa[::stride]
        y = y[::stride]

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=100)
    try:
        ax.plot(xa, y, color=color or "#4fc3f7",
               linewidth=float(linewidth) if linewidth else 1.5,
               label=(label or None))
        if x_units:
            ax.set_xlabel(str(x_units))
        if label:
            ax.legend(fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG")
        return buf.getvalue()
    finally:
        plt.close(fig)


def bake_fallback_png(array2d: np.ndarray, cmap: str = "viridis",
                      clim=None, max_edge: int = 1200) -> bytes:
    """Render *array2d* to PNG bytes with matplotlib Agg — the WYSIWYG fallback
    baked into the report when no renderer-harvested PNG exists (headless save).

    Downsamples by striding first so a huge frame stays cheap, caps the long
    edge at ``max_edge``. matplotlib is imported inside so it stays off the
    import path for callers that never save.

    NB: a bare 1-D array here (no panel context) is reshaped to a 1-row
    heatmap — this is the generic "any array" fallback. A report LINE PANEL
    (``kind="line"``) uses :func:`bake_line_fallback_png` instead, which
    draws a real ``ax.plot`` curve; see ``handlers.py``'s bake call sites."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.colors as mcolors
    from matplotlib import colormaps as mpl_colormaps
    from PIL import Image

    arr = np.asarray(array2d)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # RGB(A) passthrough (e.g. an IPF map) — no colormap.
    is_rgb = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    if not is_rgb and arr.ndim != 2:
        arr = np.atleast_2d(arr)

    # Stride-downsample the long edge to ~max_edge before any float work.
    long_edge = max(arr.shape[0], arr.shape[1])
    if long_edge > max_edge:
        stride = int(np.ceil(long_edge / max_edge))
        arr = arr[::stride, ::stride] if not is_rgb else arr[::stride, ::stride, :]

    if is_rgb:
        rgb = arr
        if np.issubdtype(rgb.dtype, np.floating):
            mx = float(np.nanmax(rgb)) if rgb.size else 1.0
            scale = 255.0 if mx <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0, 255)
        rgb8 = rgb.astype(np.uint8)
        img = Image.fromarray(rgb8, mode=("RGBA" if rgb8.shape[-1] == 4 else "RGB"))
    else:
        data = np.asarray(arr, dtype=np.float64)
        finite = data[np.isfinite(data)]
        if clim is not None and clim[0] is not None and clim[1] is not None:
            lo, hi = float(clim[0]), float(clim[1])
        elif finite.size:
            lo = float(np.nanmin(finite))
            hi = float(np.nanpercentile(finite, 99.5))
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        norm = mcolors.Normalize(vmin=lo, vmax=hi, clip=True)
        try:
            cmap_obj = mpl_colormaps[cmap]
        except (KeyError, ValueError, AttributeError):
            cmap_obj = mpl_colormaps["viridis"]
        rgba = cmap_obj(norm(np.nan_to_num(data, nan=lo)))
        rgb8 = (rgba[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb8, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── .spyde-report zip (de)serialization — atomic write ────────────────────────


REPORT_SUFFIX = ".spyde-report"


def write_report(doc: ReportDoc, path: str,
                 assets: "dict[str, bytes] | None" = None) -> None:
    """Write *doc* to a ``.spyde-report`` zip at *path*, ATOMICALLY (tmp file in
    the same dir + ``os.replace``) so a crash never leaves a torn container.

    ``assets`` maps ``cell_id -> bytes`` for the baked snapshot of each figure
    cell AND the raw image bytes of each image cell. Cells missing from ``assets``
    are written without an asset (the caller is expected to always provide one via
    harvest or bake for figures, and the held image bytes for image cells)."""
    assets = assets or {}
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.md", serialize_report_md(doc))
            for c in doc.cells:
                if c.cell_type == "image":
                    # A photo: the raw image bytes at assets/<id>.<ext>, NO yaml
                    # (the missing-yaml is what marks it an image cell on reload).
                    data = assets.get(c.id)
                    if data:
                        ext = (c.image_ext or "png").lower()
                        zf.writestr(f"assets/{c.id}.{ext}", data)
                    continue
                if c.cell_type == "split":
                    # A split cell's FIGURE SIDE writes exactly like a figure or
                    # image cell (same asset layout, keyed by the split cell id):
                    # a figure side (has a spec) → figures/<id>.yaml + assets/<id>.png;
                    # a photo side (image_ext, no spec) → assets/<id>.<ext>; an empty
                    # figure side → no asset. The text side lives in report.md.
                    if c.spec is not None:
                        zf.writestr(f"figures/{c.id}.yaml", c.spec.to_yaml())
                        png = assets.get(c.id)
                        if png:
                            zf.writestr(f"assets/{c.id}.png", png)
                    elif c.image_ext:
                        data = assets.get(c.id)
                        if data:
                            ext = (c.image_ext or "png").lower()
                            zf.writestr(f"assets/{c.id}.{ext}", data)
                    continue
                if c.cell_type == "movie":
                    # A movie cell: its MovieSpec at movies/<id>.yaml (the movies/
                    # sibling is what marks it a movie on reload) + a baked poster PNG
                    # at assets/<id>.png. A placeholder movie (no source) still writes
                    # its (mostly-empty) spec so the cell survives a round-trip.
                    if c.movie is not None:
                        zf.writestr(f"movies/{c.id}.yaml", c.movie.to_yaml())
                    poster = assets.get(c.id)
                    if poster:
                        zf.writestr(f"assets/{c.id}.png", poster)
                    continue
                if c.cell_type != "figure":
                    continue
                if c.spec is not None:
                    zf.writestr(f"figures/{c.id}.yaml", c.spec.to_yaml())
                png = assets.get(c.id)
                if png:
                    zf.writestr(f"assets/{c.id}.png", png)
        os.replace(tmp, path)
    except Exception:
        # Never leave the tmp file behind on failure (atomic contract).
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


# The top-level entries an exported markdown FOLDER is allowed to contain — so a
# re-export into a previous export overwrites cleanly, but we never clobber an
# arbitrary populated directory the user picked by mistake.
_MD_FOLDER_ALLOWED = frozenset({"report.md", "figures", "assets", "movies"})


def dir_is_safe_md_target(directory: str) -> bool:
    """True when *directory* is safe to write a markdown-folder export into:
    it doesn't exist yet, is empty, or contains ONLY the entries a prior export
    would have produced (``report.md`` / ``figures`` / ``assets``). A directory
    with other content is refused (conservative — don't clobber the user's data)."""
    if not os.path.isdir(directory):
        # A missing path (we'll create it) or a plain file (caller errors) — a
        # non-existent dir is safe to create; a file is handled by the caller.
        return not os.path.exists(directory)
    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    return all(name in _MD_FOLDER_ALLOWED for name in entries)


def write_report_dir(doc: ReportDoc, directory: str,
                     assets: "dict[str, bytes] | None" = None) -> None:
    """Write *doc* to *directory* as the UNZIPPED container — exactly the same
    files a ``.spyde-report`` zip holds, laid out on disk:

        <directory>/report.md
        <directory>/figures/<id>.yaml
        <directory>/assets/<id>.png

    This is the "Export as markdown folder" form (a plain, pandoc-ready markdown
    tree). The caller is expected to have vetted the directory with
    :func:`dir_is_safe_md_target` first."""
    assets = assets or {}
    os.makedirs(directory, exist_ok=True)
    figures_dir = os.path.join(directory, "figures")
    assets_dir = os.path.join(directory, "assets")
    with open(os.path.join(directory, "report.md"), "w", encoding="utf-8") as f:
        f.write(serialize_report_md(doc))
    for c in doc.cells:
        if c.cell_type == "image":
            data = assets.get(c.id)
            if data:
                os.makedirs(assets_dir, exist_ok=True)
                ext = (c.image_ext or "png").lower()
                with open(os.path.join(assets_dir, f"{c.id}.{ext}"), "wb") as f:
                    f.write(data)
            continue
        if c.cell_type == "split":
            # The split cell's figure side writes like a figure/photo cell
            # (§write_report): a figure side → figures/<id>.yaml + assets/<id>.png;
            # a photo side → assets/<id>.<ext>; an empty side → nothing.
            if c.spec is not None:
                os.makedirs(figures_dir, exist_ok=True)
                with open(os.path.join(figures_dir, f"{c.id}.yaml"), "w",
                          encoding="utf-8") as f:
                    f.write(c.spec.to_yaml())
                png = assets.get(c.id)
                if png:
                    os.makedirs(assets_dir, exist_ok=True)
                    with open(os.path.join(assets_dir, f"{c.id}.png"), "wb") as f:
                        f.write(png)
            elif c.image_ext:
                data = assets.get(c.id)
                if data:
                    os.makedirs(assets_dir, exist_ok=True)
                    ext = (c.image_ext or "png").lower()
                    with open(os.path.join(assets_dir, f"{c.id}.{ext}"), "wb") as f:
                        f.write(data)
            continue
        if c.cell_type == "movie":
            # A movie cell (§write_report): movies/<id>.yaml + assets/<id>.png poster.
            if c.movie is not None:
                movies_dir = os.path.join(directory, "movies")
                os.makedirs(movies_dir, exist_ok=True)
                with open(os.path.join(movies_dir, f"{c.id}.yaml"), "w",
                          encoding="utf-8") as f:
                    f.write(c.movie.to_yaml())
            poster = assets.get(c.id)
            if poster:
                os.makedirs(assets_dir, exist_ok=True)
                with open(os.path.join(assets_dir, f"{c.id}.png"), "wb") as f:
                    f.write(poster)
            continue
        if c.cell_type != "figure":
            continue
        if c.spec is not None:
            os.makedirs(figures_dir, exist_ok=True)
            with open(os.path.join(figures_dir, f"{c.id}.yaml"), "w",
                      encoding="utf-8") as f:
                f.write(c.spec.to_yaml())
        png = assets.get(c.id)
        if png:
            os.makedirs(assets_dir, exist_ok=True)
            with open(os.path.join(assets_dir, f"{c.id}.png"), "wb") as f:
                f.write(png)


def read_report(path: str) -> "tuple[ReportDoc, dict[str, bytes]]":
    """Read a ``.spyde-report`` zip → ``(ReportDoc, assets)`` where ``assets``
    maps ``cell_id -> bytes`` (a figure's baked PNG, or an image cell's raw image
    bytes). Figure ``spec``s are attached from ``figures/<id>.yaml``.

    **Figure vs image disambiguation** (the load-time rule): a ``.png`` ref
    parses as a FIGURE cell by default, but a figure MUST have a sibling
    ``figures/<id>.yaml`` — a ``.png`` cell with NO yaml is really a PHOTO (a
    pasted/dropped PNG), so it is PROMOTED to an ``"image"`` cell here. Non-png
    image refs already parsed as image cells. Existing figure-only reports (PNG +
    yaml) are unaffected, so this stays fully back-compatible."""
    assets: dict[str, bytes] = {}
    with zipfile.ZipFile(path, "r") as zf:
        md = zf.read("report.md").decode("utf-8")
        doc = parse_report_md(md)
        names = set(zf.namelist())
        for c in doc.cells:
            if c.cell_type == "image":
                ext = (c.image_ext or "png").lower()
                asset_name = f"assets/{c.id}.{ext}"
                if asset_name in names:
                    assets[c.id] = zf.read(asset_name)
                continue
            if c.cell_type == "split":
                # A split cell's figure side re-hydrates like a figure OR image
                # cell, keyed by the split cell id. A sibling figures/<id>.yaml →
                # a figure side (attach the spec, read the .png). No yaml but an
                # image_ext (parsed off the ref) → a photo side (read the raw
                # bytes). An empty figure side → nothing. The split cell is NEVER
                # promoted to an image cell (its cell_type stays "split").
                spec_name = f"figures/{c.id}.yaml"
                if spec_name in names:
                    try:
                        c.spec = FigureSpec.from_yaml(
                            zf.read(spec_name).decode("utf-8"))
                    except (yaml.YAMLError, ValueError, TypeError, KeyError) as e:
                        # A MALFORMED figure yaml (bad scalar, truncated write,
                        # hand-edit) drops the live spec to the baked PNG — but
                        # LOUDLY: record why so report_open can warn the user this
                        # figure lost its editability (and DON'T mask a code bug in
                        # from_dict, e.g. AttributeError — that still propagates).
                        c.spec = None
                        c.spec_error = str(e)
                        log.warning(
                            "report %s: figures/%s.yaml failed to parse (%s); "
                            "cell shown as baked image only", path, c.id, e)
                    png_name = f"assets/{c.id}.png"
                    if png_name in names:
                        assets[c.id] = zf.read(png_name)
                elif c.image_ext:
                    ext = (c.image_ext or "png").lower()
                    asset_name = f"assets/{c.id}.{ext}"
                    if asset_name in names:
                        assets[c.id] = zf.read(asset_name)
                else:
                    # No yaml and no non-png ext: a ``.png`` PHOTO side (a dropped
                    # PNG) parsed through the figure-ref branch (which never set
                    # ``image_ext``). Same figure-vs-image disambiguation as a
                    # standalone ``.png`` cell: no yaml → it's a photo. Read its
                    # bytes and stamp ``image_ext="png"`` so a re-save round-trips.
                    png_name = f"assets/{c.id}.png"
                    if png_name in names:
                        assets[c.id] = zf.read(png_name)
                        c.image_ext = "png"
                continue
            if c.cell_type == "movie":
                # A movie cell re-hydrates its MovieSpec from movies/<id>.yaml and
                # its poster PNG from assets/<id>.png. A malformed spec drops to an
                # empty MovieSpec but is recorded + logged (same tolerance as a
                # figure spec). A placeholder movie (no source assigned) may have no
                # yaml at all — it stays an empty movie cell.
                spec_name = f"movies/{c.id}.yaml"
                if spec_name in names:
                    try:
                        c.movie = MovieSpec.from_yaml(
                            zf.read(spec_name).decode("utf-8"))
                        c.placeholder = c.movie.source is None
                    except (yaml.YAMLError, ValueError, TypeError, KeyError) as e:
                        c.movie = MovieSpec()
                        c.spec_error = str(e)
                        log.warning(
                            "report %s: movies/%s.yaml failed to parse (%s); "
                            "movie cell shown as poster only", path, c.id, e)
                else:
                    c.movie = MovieSpec()
                poster_name = f"assets/{c.id}.png"
                if poster_name in names:
                    assets[c.id] = zf.read(poster_name)
                continue
            if c.cell_type != "figure":
                continue
            spec_name = f"figures/{c.id}.yaml"
            has_spec = spec_name in names
            if has_spec:
                try:
                    c.spec = FigureSpec.from_yaml(zf.read(spec_name).decode("utf-8"))
                except (yaml.YAMLError, ValueError, TypeError, KeyError) as e:
                    # See the split-cell branch above: a malformed spec drops to the
                    # baked PNG but is recorded + logged, and a code bug (AttributeError)
                    # still propagates rather than masquerading as "corrupt file".
                    c.spec = None
                    c.spec_error = str(e)
                    log.warning(
                        "report %s: figures/%s.yaml failed to parse (%s); "
                        "cell shown as baked image only", path, c.id, e)
            asset_name = f"assets/{c.id}.png"
            if asset_name in names:
                assets[c.id] = zf.read(asset_name)
            # A ``.png`` cell with NO figures yaml is really a PHOTO — promote it
            # to an image cell (its bytes are already harvested above). A
            # placeholder (template figure, never has a yaml) is left as-is.
            if not has_spec and not c.placeholder:
                c.cell_type = "image"
                c.image_ext = "png"
                c.placeholder = False
                c.spec = None
    return doc, assets
