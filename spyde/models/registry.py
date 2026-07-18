"""Upgradeable model registry for the SpotUNet disk detector.

The registry is what lets the model be revised indefinitely WITHOUT re-releasing
SpyDE. It maps a ``model_id`` to its architecture hyperparams and a weights source
(bundled-in-package, or a Hugging Face repo file downloaded on demand).

Three manifest layers are merged, later layers overriding earlier on ``id``
collision and contributing a newer ``default``:

    bundled (pinned, ships in the wheel)
      < remote  (~/.spyde/models/registry.json, fetched from Hugging Face)
      < user    (a user's own edits to ~/.spyde/models/registry.json)

(The remote and user manifests are the SAME file on disk — a remote refresh
overwrites it, but a user is free to hand-edit it; both are the "user-dir"
manifest and sit above the bundled one.)

Resolution + caching:
  - ``list_models()`` / ``available_models()`` → the merged manifest for the UI.
  - ``get_model(model_id)`` → a cached ``(model, device)``; resolves weights
    (bundled via importlib.resources, hf via huggingface_hub) and falls back to the
    bundled default on ANY failure so the wizard never crashes offline.
  - ``refresh_remote_registry()`` → pull the latest manifest from Hugging Face into
    the user dir (the "check for new models" path). Optional/lazy/offline-safe.

See ``RELEASING.md`` for the author-side workflow of shipping a revised model.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from importlib import resources
from typing import Optional

log = logging.getLogger(__name__)

# Hugging Face repo that hosts the .pt weights + a registry.json. The repo id is
# centralised here so a rename is one edit. (A model entry may override `repo`.)
HF_REPO = "cssfrancis/spyde-spotunet"
REMOTE_REGISTRY_FILE = "registry.json"

_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict = {}          # model_id -> (model, device)
_MANIFEST_CACHE: Optional[dict] = None


# ── user dir ──────────────────────────────────────────────────────────────────
def user_models_dir() -> str:
    """``~/.spyde/models`` — where remote-downloaded weights + the user manifest
    live (mirrors the ``~/.spyde`` settings dir used elsewhere). Created on demand."""
    d = os.path.join(os.path.expanduser("~"), ".spyde", "models")
    os.makedirs(d, exist_ok=True)
    return d


# ── manifest loading + merge ────────────────────────────────────────────────────
def _load_bundled_manifest() -> dict:
    try:
        text = (resources.files("spyde.models.weights") / "registry.json").read_text()
        return json.loads(text)
    except Exception as e:        # pragma: no cover — bundled file should always exist
        log.warning("[models] bundled registry.json unreadable: %s", e)
        return {"default": None, "models": []}


def _load_user_manifest() -> Optional[dict]:
    path = os.path.join(user_models_dir(), REMOTE_REGISTRY_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        log.warning("[models] user registry.json unreadable (%s); ignoring", e)
        return None


def _merge_manifests(*manifests: Optional[dict]) -> dict:
    """Merge in priority order (earliest = lowest). Later entries override earlier
    on ``id``; a later non-null ``default`` wins."""
    by_id: dict = {}
    default = None
    for man in manifests:
        if not man:
            continue
        for m in man.get("models", []):
            mid = m.get("id")
            if mid:
                by_id[mid] = {**by_id.get(mid, {}), **m}
        if man.get("default"):
            default = man["default"]
    # Keep a stable, bundled-first order for the UI.
    models = list(by_id.values())
    if default not in by_id and models:
        default = models[0]["id"]
    return {"default": default, "models": models}


def _manifest(force: bool = False) -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None and not force:
        return _MANIFEST_CACHE
    _MANIFEST_CACHE = _merge_manifests(_load_bundled_manifest(), _load_user_manifest())
    return _MANIFEST_CACHE


def _invalidate_manifest():
    global _MANIFEST_CACHE
    _MANIFEST_CACHE = None


# ── public manifest API (for the UI) ────────────────────────────────────────────
def list_models() -> list[dict]:
    """Full merged model entries (id/label/arch/source/version/notes)."""
    return list(_manifest().get("models", []))


def default_model_id() -> Optional[str]:
    return _manifest().get("default")


def available_models() -> dict:
    """Compact payload for the wizard Model dropdown: ``{default, models:[{id,label,
    version,notes}]}`` (arch/source omitted — the UI doesn't need them)."""
    return {
        "default": default_model_id(),
        "models": [
            {"id": m["id"], "label": m.get("label", m["id"]),
             "version": m.get("version"), "notes": m.get("notes")}
            for m in list_models()
        ],
    }


def _entry(model_id: Optional[str]) -> Optional[dict]:
    mid = model_id or default_model_id()
    for m in list_models():
        if m["id"] == mid:
            return m
    return None


# ── weight resolution ───────────────────────────────────────────────────────────
def _resolve_bundled(source: dict) -> str:
    """Return a real filesystem path to a bundled weight. ``importlib.resources``
    may hand back a path inside a zip in a frozen build, so copy to a temp file
    when ``as_file`` can't give a stable on-disk path."""
    fname = source["file"]
    ref = resources.files("spyde.models.weights") / fname
    try:
        # Fast path: a normal on-disk package (dev + most frozen builds).
        p = str(ref)
        if os.path.exists(p):
            return p
    except Exception:
        pass
    data = ref.read_bytes()
    tmp = os.path.join(tempfile.gettempdir(), f"spyde_{fname}")
    with open(tmp, "wb") as fh:
        fh.write(data)
    return tmp


def _resolve_hf(source: dict) -> str:
    """Download (or reuse the cached) HF-hosted weight into ~/.spyde/models. Raises
    on failure (the caller catches and falls back to bundled)."""
    from huggingface_hub import hf_hub_download

    repo = source.get("repo", HF_REPO)
    fname = source["file"]
    return hf_hub_download(repo_id=repo, filename=fname, local_dir=user_models_dir())


def _verify_sha256(path: str, expected: str, model_id) -> None:
    """Raise if the file at ``path`` doesn't hash to ``expected`` (hex sha256).
    Registry entries MAY carry a ``sha256`` — when present, a corrupted or
    tampered weight file is rejected (get_model then falls back to bundled)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    got = h.hexdigest()
    if got.lower() != str(expected).lower():
        raise ValueError(
            f"sha256 mismatch for model {model_id!r}: expected {expected}, got {got}")


def _resolve_weights(entry: dict) -> str:
    source = entry.get("source", {})
    stype = source.get("type")
    if stype == "bundled":
        path = _resolve_bundled(source)
    elif stype == "hf":
        path = _resolve_hf(source)
    else:
        raise ValueError(f"unknown model source type {stype!r} for {entry.get('id')}")
    expected = entry.get("sha256") or source.get("sha256")
    if expected:
        _verify_sha256(path, expected, entry.get("id"))
    return path


# ── public model API ────────────────────────────────────────────────────────────
def is_cached(model_id: Optional[str] = None) -> bool:
    """True when the model's weights are already present on this machine (bundled,
    or an HF file previously downloaded into ~/.spyde/models). Lets the action
    layer decide whether to surface a "downloading model…" status before
    ``ensure_local`` blocks on the network."""
    entry = _entry(model_id)
    if entry is None:
        return True                      # unknown id resolves to bundled default
    source = entry.get("source", {})
    if source.get("type") != "hf":
        return True
    return os.path.exists(os.path.join(user_models_dir(), source.get("file", "")))


def ensure_local(model_id: Optional[str] = None) -> Optional[str]:
    """Resolve the model's weights to a LOCAL file path, downloading once if the
    source is Hugging Face. Call this on the CLIENT before submitting a batch
    compute so dask workers never hit the network (they re-resolve to the same,
    now-present file). Returns the path, or None on failure (callers should then
    let ``get_model``'s bundled-default fallback handle it)."""
    entry = _entry(model_id)
    if entry is None:
        return None
    try:
        return _resolve_weights(entry)
    except Exception as e:
        log.warning("[models] ensure_local(%r) failed (%s); compute will fall back "
                    "to the bundled default", model_id, e)
        return None


def get_model(model_id: Optional[str] = None):
    """Return a cached ``(model, device)`` for ``model_id`` (default = registry
    ``default``). On ANY failure resolving a non-default model (no huggingface_hub,
    offline, 404, bad checkpoint) logs a warning and falls back to the bundled
    default — so the wizard always has a working detector."""
    from . import infer

    mid = model_id or default_model_id()
    with _CACHE_LOCK:
        if mid in _MODEL_CACHE:
            return _MODEL_CACHE[mid]

    entry = _entry(mid)
    if entry is None:
        log.warning("[models] unknown model_id %r; using default", mid)
        return get_model(None) if mid is not None else _raise_no_models()

    try:
        path = _resolve_weights(entry)
        result = infer.load_model(path, arch=entry.get("arch"))
    except Exception as e:
        default = default_model_id()
        if mid != default:
            log.warning("[models] could not load %r (%s); falling back to default %r",
                        mid, e, default)
            return get_model(default)
        raise

    with _CACHE_LOCK:
        _MODEL_CACHE[mid] = result
    return result


_CPU_MODEL_CACHE: dict = {}      # model_id -> (cpu_model, cpu_device)


def get_cpu_model(model_id: Optional[str] = None):
    """Return a cached ``(model, torch.device('cpu'))`` for ``model_id`` WITHOUT
    disturbing the primary (possibly MPS/CUDA) cache.

    Used by the Mac neural BATCH gate: when the batch is forced to CPU (the safe
    default) we need a CPU model, but the single-frame PREVIEW must keep the cached
    MPS model — so we can't move the shared cached model in-place. This keeps a
    separate CPU copy, loaded once per id. The CPU model is a full independent
    module (loaded from weights on CPU), so mutating it never affects the MPS one."""
    import torch as _torch
    mid = model_id or default_model_id()
    with _CACHE_LOCK:
        if mid in _CPU_MODEL_CACHE:
            return _CPU_MODEL_CACHE[mid]
    # Load a fresh CPU instance from the same weights (never reuses the primary
    # cache object, so the MPS model the preview uses is untouched).
    from . import infer
    entry = _entry(mid)
    if entry is None:
        model, _dev = get_model(None)
        model = model.to(_torch.device("cpu"))
    else:
        try:
            path = _resolve_weights(entry)
            model, _dev = infer.load_model(path, device=_torch.device("cpu"),
                                           arch=entry.get("arch"))
        except Exception as e:
            log.warning("[models] get_cpu_model(%r) load failed (%s); reusing "
                        "primary model on CPU", mid, e)
            model, _dev = get_model(mid)
            model = model.to(_torch.device("cpu"))
    model.eval()
    result = (model, _torch.device("cpu"))
    with _CACHE_LOCK:
        _CPU_MODEL_CACHE[mid] = result
    return result


def demote_cached_models_to_cpu() -> None:
    """Move every cached ``(model, device)`` to CPU and rewrite its device to CPU.

    Called when an MPS forward proves flaky (infer._forward_with_cpu_retry): the
    cache is shared across the process, so leaving a CPU-moved model paired with a
    stale ``mps`` device would make the NEXT ``get_model`` caller ``.to("mps")`` it
    and crash again. This pins the whole process's neural inference to CPU for the
    rest of the session — the safe response to a device that just failed. No-op if
    nothing is cached; idempotent (a model already on CPU is unchanged)."""
    import torch as _torch
    with _CACHE_LOCK:
        for mid, (model, device) in list(_MODEL_CACHE.items()):
            try:
                if getattr(device, "type", str(device)) != "cpu":
                    model = model.to(_torch.device("cpu"))
                    model.eval()
                    _MODEL_CACHE[mid] = (model, _torch.device("cpu"))
            except Exception as e:      # pragma: no cover — best-effort demotion
                log.debug("[models] demoting cached model %r to CPU failed: %s", mid, e)


def _raise_no_models():        # pragma: no cover
    raise RuntimeError("no models registered (bundled registry.json missing?)")


def refresh_remote_registry() -> dict:
    """Pull the latest ``registry.json`` from Hugging Face into ~/.spyde/models so
    new model entries appear without reinstalling SpyDE. Returns the freshly-merged
    ``available_models()``. Offline / missing-dep failures are logged, not raised
    (the bundled manifest keeps working)."""
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO, filename=REMOTE_REGISTRY_FILE,
            local_dir=user_models_dir())
        log.info("[models] refreshed remote registry from %s -> %s", HF_REPO, downloaded)
    except Exception as e:
        log.warning("[models] remote registry refresh failed (%s); keeping current", e)
    _invalidate_manifest()
    return available_models()
