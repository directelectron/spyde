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

import io
import os
import re
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml

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


# A standalone-paragraph image ref whose target is ``assets/<id>.png``. The
# basename (without extension) is the cell id; the alt text is the caption.
_FIG_LINE_RE = re.compile(
    r"^!\[(?P<caption>.*?)\]\(assets/(?P<cid>[^)/]+)\.png\)\s*$")
# A template placeholder comment: ``<!-- spyde:placeholder <id> [caption] -->``.
_PLACEHOLDER_RE = re.compile(
    r"^<!--\s*spyde:placeholder\s+(?P<cid>\S+)(?:\s+(?P<caption>.*?))?\s*-->\s*$")


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

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "fingerprint": (dict(self.fingerprint)
                            if self.fingerprint is not None else None),
            "tree_uid": self.tree_uid,
            "tree_node": self.tree_node,
            "view": self.view,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "SignalRef":
        d = d or {}
        fp = d.get("fingerprint")
        return cls(
            file_path=d.get("file_path"),
            fingerprint=(dict(fp) if isinstance(fp, dict) else None),
            tree_uid=d.get("tree_uid"),
            tree_node=d.get("tree_node"),
            view=d.get("view"),
            title=d.get("title"),
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
        try:
            sig = plot.plot_state.current_signal
            title = str(sig.metadata.get_item("General.title", default="") or "")
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
                   title=title or tree_node)

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
        # 1b) open trees by root/node title.
        if not candidates:
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
                    candidates.append(p)
                elif (self.file_path is not None
                      and getattr(tree, "source_path", None) == self.file_path):
                    candidates.append(p)
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
    the compose edit handlers."""
    source: SignalRef = field(default_factory=SignalRef)
    cmap: str = "viridis"
    clim: list | None = None                   # [lo, hi] or None (auto)
    alpha: float = 1.0
    visible: bool = True
    id: str = field(default_factory=new_layer_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source.to_dict(),
            "cmap": self.cmap,
            "clim": (list(self.clim) if self.clim is not None else None),
            "alpha": float(self.alpha),
            "visible": bool(self.visible),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayerSpec":
        d = d or {}
        clim = d.get("clim")
        return cls(
            source=SignalRef.from_dict(d.get("source")),
            cmap=d.get("cmap", "viridis"),
            clim=(list(clim) if clim is not None else None),
            alpha=float(d.get("alpha", 1.0)),
            visible=bool(d.get("visible", True)),
            id=d.get("id") or new_layer_id(),
        )


@dataclass
class PanelSpec:
    """One panel (axes cell) of a figure: ≥1 layer, calibrated axes, annotations,
    and decorations. Phase 1 uses a single panel with a single image layer."""
    id: str = "p1"
    grid_pos: list = field(default_factory=lambda: [0, 0])
    kind: str = "image"                        # "image" | "line"
    layers: list = field(default_factory=list)         # [LayerSpec]
    axes: dict | None = None                   # {units, scale:[sy,sx], offset:[oy,ox]}
    annotations: list = field(default_factory=list)    # [dict] (marker kwargs)
    scalebar: bool = False
    colorbar: bool = False
    title: str = ""
    insets: list = field(default_factory=list)         # [dict] (Phase 2)

    def to_dict(self) -> dict:
        return {
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

    @classmethod
    def from_dict(cls, d: dict) -> "PanelSpec":
        d = d or {}
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
        )


@dataclass
class FigureSpec:
    """The full recipe for a figure cell — ONE schema shared by report cells,
    the combined-figure editor, and MDI layering. Phase 1 emits
    ``layout={kind:single}`` with one panel / one layer."""
    layout: dict = field(default_factory=lambda: {"kind": "single"})
    panels: list = field(default_factory=list)          # [PanelSpec]
    nav_context: dict | None = None            # {"indices": [iy, ix]}

    def to_dict(self) -> dict:
        return {
            "layout": dict(self.layout),
            "panels": [p.to_dict() for p in self.panels],
            "nav_context": (dict(self.nav_context)
                            if self.nav_context is not None else None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FigureSpec":
        d = d or {}
        return cls(
            layout=dict(d.get("layout") or {"kind": "single"}),
            panels=[PanelSpec.from_dict(x) for x in (d.get("panels") or [])],
            nav_context=(dict(d["nav_context"])
                         if d.get("nav_context") is not None else None),
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
class Cell:
    """A document cell.

    ``cell_type`` is ``"markdown"`` or ``"figure"``. A figure cell carries a
    ``caption``, a ``fig_id`` (the FigureSpec / asset basename == this cell's
    id), a ``spec`` (FigureSpec, in memory), and — for a template — a
    ``placeholder`` flag when no figure has been dropped yet.

    ``html`` is a DERIVED, NON-PERSISTED field: the renderer's own
    marked+DOMPurify-sanitized rendering of ``source`` (delivered on every
    markdown commit). It is used ONLY by HTML export — never written into
    report.md / the zip, and absent after a reload until the next edit. HTML
    export falls back to escaping the raw markdown when it's empty."""
    id: str = field(default_factory=new_cell_id)
    cell_type: str = "markdown"
    source: str = ""                           # markdown text (markdown cells)
    caption: str = ""                          # figure caption / alt text
    placeholder: bool = False
    spec: FigureSpec | None = None             # figure recipe (figure cells)
    html: str = ""                             # derived, NON-persisted (export only)


@dataclass
class ReportDoc:
    """The in-memory report: metadata + an ordered list of :class:`Cell`."""
    title: str = "Untitled Report"
    template: bool = False
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

    def touch(self) -> None:
        self.modified = _utcnow()


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
        "created": doc.created,
        "modified": doc.modified,
    }
    parts = ["---\n", _dump_yaml(front), "---\n"]
    body_blocks: list[str] = []
    for c in doc.cells:
        if c.cell_type == "markdown":
            body_blocks.append(c.source.rstrip("\n"))
        elif c.cell_type == "figure":
            if c.placeholder:
                cap = (c.caption or "").strip()
                marker = f"<!-- spyde:placeholder {c.id}"
                marker += f" {cap} -->" if cap else " -->"
                body_blocks.append(marker)
            else:
                # Standalone-paragraph image ref; alt text == caption.
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
    fence_marker: str | None = None

    def flush_md() -> None:
        if md_buf:
            src = "\n".join(md_buf).strip("\n")
            # Drop a run that is nothing but blank lines (paragraph separators).
            if src.strip() != "":
                cells.append(Cell(id=new_cell_id(), cell_type="markdown",
                                  source=src))
            md_buf.clear()

    for raw in body.splitlines():
        line = raw.rstrip("\r")
        stripped = line.strip()
        # Fence tracking (``` or ~~~). A fence opens/closes only at the same
        # marker; inside a fence NOTHING is interpreted as a figure/placeholder.
        fence_open = re.match(r"^(```+|~~~+)", stripped)
        if fence_open:
            marker = fence_open.group(1)[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker is not None and stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = None
            md_buf.append(line)
            continue
        if in_fence:
            md_buf.append(line)
            continue

        m_fig = _FIG_LINE_RE.match(stripped)
        m_ph = _PLACEHOLDER_RE.match(stripped)
        if m_fig is not None:
            flush_md()
            cells.append(Cell(id=m_fig.group("cid"), cell_type="figure",
                              caption=m_fig.group("caption") or "",
                              placeholder=False))
        elif m_ph is not None:
            flush_md()
            cells.append(Cell(id=m_ph.group("cid"), cell_type="figure",
                              caption=(m_ph.group("caption") or "").strip(),
                              placeholder=True))
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


def bake_fallback_png(array2d: np.ndarray, cmap: str = "viridis",
                      clim=None, max_edge: int = 1200) -> bytes:
    """Render *array2d* to PNG bytes with matplotlib Agg — the WYSIWYG fallback
    baked into the report when no renderer-harvested PNG exists (headless save).

    Downsamples by striding first so a huge frame stays cheap, caps the long
    edge at ``max_edge``. matplotlib is imported inside so it stays off the
    import path for callers that never save."""
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

    ``assets`` maps ``cell_id -> PNG bytes`` for the baked snapshot of each
    figure cell. Figure cells missing from ``assets`` are written without an
    asset (the caller is expected to always provide one via harvest or bake)."""
    assets = assets or {}
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.md", serialize_report_md(doc))
            for c in doc.cells:
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
_MD_FOLDER_ALLOWED = frozenset({"report.md", "figures", "assets"})


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
    maps ``cell_id -> PNG bytes``. Figure ``spec``s are attached from
    ``figures/<id>.yaml``."""
    assets: dict[str, bytes] = {}
    with zipfile.ZipFile(path, "r") as zf:
        md = zf.read("report.md").decode("utf-8")
        doc = parse_report_md(md)
        names = set(zf.namelist())
        for c in doc.cells:
            if c.cell_type != "figure":
                continue
            spec_name = f"figures/{c.id}.yaml"
            if spec_name in names:
                try:
                    c.spec = FigureSpec.from_yaml(zf.read(spec_name).decode("utf-8"))
                except Exception:
                    c.spec = None
            asset_name = f"assets/{c.id}.png"
            if asset_name in names:
                assets[c.id] = zf.read(asset_name)
    return doc, assets
