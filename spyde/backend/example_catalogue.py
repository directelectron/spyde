"""
example_catalogue.py — the Examples menu's catalogue, from **em-database**.

SpyDE's example datasets come from the `em-database
<https://pypi.org/project/em-database/>`_ package rather than from each
analysis library's own hand-rolled fixtures. One curated, checksummed, cited
collection beats a handful of loaders scattered across pyxem/exspy/kikuchipy:
the datasets carry real acquisition metadata, they are versioned independently
of SpyDE, and adding one is a YAML file upstream rather than a code change
here.

What this module turns that into is the shape the menu needs:

* **grouped by technique** — ``metadata["technique"]`` is already there
  (4D-STEM, EELS, EBSD, Cryo-EM, In-situ TEM, STEM), so the menu gets one
  submenu per technique for free;
* **sized** — ``data_size`` is a ready-made human string ("1.4 GB");
* **shaped** — see :func:`_shape_of`, the one thing em-database does not
  (yet) carry for every dataset;
* **marked downloaded or not** — ``DownloadableDataset.filepath()`` returns
  the local path or ``None``, which is exactly that question.

Nothing here downloads anything. Building the catalogue must stay cheap enough
to run every time the menu opens.
"""
from __future__ import annotations

import inspect
import json
import logging
import os

log = logging.getLogger(__name__)

#: Techniques in the order the menu should show them — the modalities SpyDE is
#: built around first, then the rest alphabetically. Anything not listed still
#: appears, after these.
TECHNIQUE_ORDER = ("4D-STEM", "EELS", "EDS", "EBSD", "STEM", "In-situ TEM",
                   "Cryo-EM")

#: Shapes read out of downloaded files, cached beside the data so the menu
#: doesn't reopen every file each time it opens.
_SHAPE_CACHE_NAME = ".spyde_shapes.json"

_shape_cache: dict | None = None


def available() -> bool:
    """Is em-database importable? The menu degrades to Dummy Data if not."""
    try:
        import em_database  # noqa: F401
        return True
    except Exception as e:
        log.debug("em-database unavailable: %s", e)
        return False


def data_dir() -> str:
    """Where em-database keeps downloaded datasets (``EM_DATABASE_DATA_DIR``
    or ``~/em_database``). Shown by the Examples menu's "Show Example Data
    Directory"."""
    try:
        import em_database
        return str(em_database.get_data_dir())
    except Exception:
        return os.path.join(os.path.expanduser("~"), "em_database")


def datasets() -> list[tuple[str, object]]:
    """``(key, dataset)`` for every dataset em-database exposes.

    The key is the class name — stable, and what ``load_example {name}``
    carries. Filtered by ``issubclass`` rather than by name so the base class
    and incidental imports (``Path``) in the module namespace stay out.
    """
    try:
        import em_database
        import em_database.data as data
    except Exception as e:
        log.debug("listing em-database datasets failed: %s", e)
        return []

    base = em_database.DownloadableDataset
    out = []
    for name in dir(data):
        if name.startswith("_"):
            continue
        obj = getattr(data, name, None)
        if not inspect.isclass(obj) or obj is base or not issubclass(obj, base):
            continue
        try:
            out.append((name, obj()))
        except Exception as e:
            log.debug("dataset %s could not be instantiated: %s", name, e)
    return sorted(out, key=lambda kv: kv[0].lower())


def resolve(key: str):
    """The dataset object for a catalogue key, or None."""
    try:
        import em_database.data as data
        obj = getattr(data, str(key), None)
        return obj() if inspect.isclass(obj) else None
    except Exception as e:
        log.debug("resolving example %r failed: %s", key, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Shape
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> str:
    return os.path.join(data_dir(), _SHAPE_CACHE_NAME)


def _load_shape_cache() -> dict:
    global _shape_cache
    if _shape_cache is None:
        try:
            with open(_cache_path(), encoding="utf-8") as fh:
                _shape_cache = json.load(fh)
        except Exception:
            _shape_cache = {}
    return _shape_cache


def _save_shape_cache() -> None:
    try:
        os.makedirs(data_dir(), exist_ok=True)
        with open(_cache_path(), "w", encoding="utf-8") as fh:
            json.dump(_shape_cache or {}, fh)
    except Exception as e:
        log.debug("writing the shape cache failed: %s", e)


def _declared_shape(ds) -> str | None:
    """The shape em-database declares, if it does.

    Upstream (``cssfrancis/em_data``) describes each dataset with a YAML file;
    a ``shape`` there is authoritative and means the menu can show the shape
    BEFORE anything is downloaded. Not every dataset carries one yet, hence
    :func:`_read_shape`.
    """
    for source in (getattr(ds, "metadata", None) or {}, ds):
        for attr in ("shape", "data_shape"):
            val = (source.get(attr) if isinstance(source, dict)
                   else getattr(source, attr, None))
            if val:
                return _format_shape(val)
    return None


def _format_shape(val) -> str:
    """Normalise whatever shape upstream gives into ``nav | sig`` display form."""
    if isinstance(val, str):
        return val.strip()
    try:
        if (isinstance(val, dict)):
            nav, sig = val.get("navigation"), val.get("signal")
            if nav and sig:
                return f"{_tuple_str(nav)} | {_tuple_str(sig)}"
            return _tuple_str(nav or sig)
        return _tuple_str(val)
    except Exception:
        return str(val)


def _tuple_str(seq) -> str:
    return "×".join(str(int(v)) for v in seq)


def _cache_key(path: str) -> str | None:
    """Size+mtime keyed, so a re-download invalidates the cached shape."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime)}"


def shape_of_signal(sig) -> str | None:
    """``nav | sig`` for a loaded signal."""
    try:
        am = sig.axes_manager
        nav = tuple(int(a.size) for a in am.navigation_axes)
        sgl = tuple(int(a.size) for a in am.signal_axes)
        return (f"{_tuple_str(nav)} | {_tuple_str(sgl)}" if nav
                else _tuple_str(sgl))
    except Exception as e:
        log.debug("shape of signal failed: %s", e)
        return None


def record_shape(path: str | None, sig) -> None:
    """Remember a dataset's shape from a signal we already have in hand.

    Called when an example is loaded — free, exact, and it means the menu shows
    the shape from then on without ever opening the file just to look.
    """
    if not path:
        return
    key = _cache_key(path)
    shape = shape_of_signal(sig)
    if not key or not shape:
        return
    cache = _load_shape_cache()
    if cache.get(key) != shape:
        cache[key] = shape
        _save_shape_cache()


def _open_lazily(path: str):
    """Lazily open a downloaded example, metadata only.

    hyperspy's reader sniffing is ambiguous for some EBSD ``.h5`` files (it
    offers Arina/Delmic/Tofwerk/USID and refuses to choose), and those are
    exactly the files whose reader lives in kikuchipy. So: hyperspy first,
    kikuchipy as the fallback — which also keeps kikuchipy optional, since a
    dataset it cannot open simply has no shape shown.
    """
    import hyperspy.api as hs
    try:
        return hs.load(path, lazy=True)
    except Exception as e:
        log.debug("hyperspy could not open %s (%s); trying kikuchipy", path, e)
        import kikuchipy as kp
        return kp.load(path, lazy=True)


def read_shape(path: str) -> str | None:
    """Read a DOWNLOADED file's shape, without reading its data.

    A lazy load touches the store's metadata only, so this is cheap even for
    the multi-GB sets — but it still opens a file, which is why the catalogue
    never calls it inline (see :func:`warm_shapes`).
    """
    key = _cache_key(path)
    if key is None:
        return None
    cache = _load_shape_cache()
    if key in cache:
        return cache[key] or None

    shape = None
    try:
        sig = _open_lazily(path)
        if isinstance(sig, (list, tuple)):
            sig = sig[0]
        shape = shape_of_signal(sig)
        try:
            sig.data = None      # drop the dask graph promptly
        except Exception:
            pass
    except Exception as e:
        log.debug("reading the shape of %s failed: %s", path, e)

    cache[key] = shape or ""
    _save_shape_cache()
    return shape


def _cached_shape(path: str | None) -> str | None:
    """The shape we already know for a downloaded file — never opens it."""
    if not path:
        return None
    key = _cache_key(path)
    return (_load_shape_cache().get(key) or None) if key else None


def _shape_of(ds, path: str | None) -> str | None:
    """Declared shape if em-database has one, else one we have already read.

    Deliberately does NOT open files: the catalogue is rebuilt every time the
    Examples menu opens, and reading five multi-GB stores inline made that a
    7-second click. :func:`warm_shapes` fills the gap off the menu's path.
    """
    return _declared_shape(ds) or _cached_shape(path)


def warm_shapes() -> bool:
    """Read the shape of every downloaded dataset we don't know yet.

    Slow the first time (it opens each file), so run it on a worker and
    re-emit the catalogue if it returns True. Cheap and a no-op after that.
    """
    changed = False
    for _key, ds in datasets():
        if _declared_shape(ds):
            continue
        try:
            path = ds.filepath()
        except Exception:
            path = None
        if path and _cached_shape(path) is None:
            if read_shape(path):
                changed = True
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# The catalogue
# ─────────────────────────────────────────────────────────────────────────────

def _label(key: str) -> str:
    """``LayeredCuNb4DSTEM`` -> ``Layered Cu Nb 4D STEM`` is worse than the
    class name, so the class name IS the label. Kept as a hook for an upstream
    ``title`` field."""
    return key


def _technique(ds) -> str:
    md = getattr(ds, "metadata", None) or {}
    return str(md.get("technique") or "Other").strip() or "Other"


def entry(key: str, ds) -> dict:
    """One catalogue row — everything the menu draws for a dataset."""
    try:
        path = ds.filepath()
    except Exception as e:
        log.debug("filepath() for %s failed: %s", key, e)
        path = None
    md = getattr(ds, "metadata", None) or {}
    return {
        "key": key,
        "label": _label(key),
        "technique": _technique(ds),
        "size": str(getattr(ds, "data_size", "") or ""),
        "shape": _shape_of(ds, path),
        "downloaded": bool(path),
        "path": path or None,
        "file": str(getattr(ds, "file", "") or ""),
        "description": str(getattr(ds, "description", "") or ""),
        "tags": list(md.get("tags") or []),
        "microscope": str(md.get("microscope") or ""),
        "voltage": str(md.get("voltage") or ""),
        # The camera, as top-level dataset fields rather than metadata keys.
        # Worth surfacing: which detector a dataset came off is most of what
        # tells you what to expect from it (counted vs integrating, frame rate,
        # pixel size), and it is the field a user picking an example to test
        # their own data against actually looks for.
        "detector": str(getattr(ds, "detector", "") or ""),
        "detector_manufacturer": str(getattr(ds, "detector_manufacturer", "") or ""),
        "license": str(getattr(ds, "license", "") or ""),
        "source": str(getattr(ds, "source", "") or ""),
    }


def catalogue() -> dict:
    """The whole Examples menu, grouped by technique.

    ``{"available": bool, "data_dir": str, "groups": [{"technique", "items"}]}``
    — one group per technique in :data:`TECHNIQUE_ORDER`, then any others
    alphabetically.
    """
    items = [entry(key, ds) for key, ds in datasets()]
    by_tech: dict[str, list[dict]] = {}
    for it in items:
        by_tech.setdefault(it["technique"], []).append(it)

    def order(tech: str):
        try:
            return (0, TECHNIQUE_ORDER.index(tech))
        except ValueError:
            return (1, tech.lower())

    groups = [{"technique": t, "items": by_tech[t]}
              for t in sorted(by_tech, key=order)]
    return {
        "available": available(),
        "data_dir": data_dir(),
        "groups": groups,
        "n_downloaded": sum(1 for it in items if it["downloaded"]),
        "n_total": len(items),
    }
