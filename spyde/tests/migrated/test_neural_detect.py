"""Correctness tests for the vendored SpotUNet detector + the model registry.

Run in a SUBPROCESS that prints a JSON result and ``os._exit(0)`` after, matching
``test_vector_orientation_gpu.py``: torch teardown (esp. with CUDA) can segfault
at interpreter exit inside the pytest process on Windows. The model itself runs on
CPU here so these checks don't need a GPU.

Covers:
  - the bundled model loads via ``registry.get_model()`` with the registry arch,
  - ``detect`` localises planted Gaussian disks,
  - ``detect_batch`` matches per-frame ``detect`` (subpixel),
  - registry fallback: an unknown / unreachable model id falls back to the bundled
    default without raising,
  - registry upgrade: a user ``~/.spyde/models/registry.json`` adds a new entry and
    advances ``default`` over the bundled one (the no-reinstall upgrade path).
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

_DRIVER = textwrap.dedent(r"""
    import json, sys, os, tempfile
    import numpy as np
    from scipy.ndimage import gaussian_filter

    def planted_frame(centers, shape=(112, 112), amp=400.0, sigma=2.2):
        f = np.zeros(shape, np.float32)
        for (cy, cx) in centers:
            f[cy, cx] = amp
        return gaussian_filter(f, sigma)

    def nearest_dist(peaks, centers):
        # max over planted centers of the distance to the closest detection
        if len(peaks) == 0:
            return 1e9
        d = []
        for (cy, cx) in centers:
            dd = np.hypot(peaks[:, 0] - cy, peaks[:, 1] - cx).min()
            d.append(dd)
        return float(max(d))

    mode = sys.argv[1]
    out = {}

    if mode == "load":
        from spyde import models
        from spyde.models.unet import SpotUNet
        m, dev = models.get_model()
        out["is_spotunet"] = isinstance(m, SpotUNet)
        out["levels"] = int(m.levels)
        out["params"] = int(m.num_params())

    elif mode == "detect":
        from spyde import models
        m, dev = models.get_model()
        centers = [(40, 50), (70, 30), (20, 80)]
        f = planted_frame(centers)
        peaks = np.asarray(models.detect(m, f, dev, thresh=0.3), np.float32).reshape(-1, 3)
        # The net should localise the planted spots to within a few px (it fires
        # elsewhere too on this synthetic frame — we only require it FINDS these).
        out["n"] = int(len(peaks))
        out["worst_planted_dist"] = nearest_dist(peaks, centers)

    elif mode == "batch":
        from spyde import models
        m, dev = models.get_model()
        centers = [(40, 50), (70, 30)]
        frames = np.stack([planted_frame(centers), planted_frame(centers)])
        single = np.asarray(models.detect(m, frames[0], dev, thresh=0.3), np.float32).reshape(-1, 3)
        batch = models.detect_batch(m, frames, dev, thresh=0.3)
        b0 = np.asarray(batch[0], np.float32).reshape(-1, 3)
        out["n_frames"] = len(batch)
        out["single_n"] = int(len(single))
        out["batch0_n"] = int(len(b0))
        # Same count, and positions agree to subpixel (same model, same decode).
        out["counts_match"] = (len(single) == len(b0))
        if len(single) == len(b0) and len(single):
            so = single[np.lexsort((single[:, 1], single[:, 0]))]
            bo = b0[np.lexsort((b0[:, 1], b0[:, 0]))]
            out["max_pos_diff"] = float(np.abs(so[:, :2] - bo[:, :2]).max())
        else:
            out["max_pos_diff"] = 0.0

    elif mode == "fallback":
        # Unknown id with no working remote → bundled default, no exception.
        from spyde import models
        from spyde.models import registry, default_model_id
        m, dev = models.get_model("definitely-not-a-real-model-xyz")
        out["fell_back"] = m is not None
        out["default"] = default_model_id()

    elif mode == "upgrade":
        # A user registry.json adds a -v2 entry and advances `default`.
        from spyde.models import registry
        udir = registry.user_models_dir()
        man = {
            "default": "spotunet-base16-v2",
            "models": [{
                "id": "spotunet-base16-v2",
                "label": "SpotUNet base16 v2 (user)",
                "version": 2,
                "arch": {"base": 16, "in_ch": 1, "levels": 2},
                "source": {"type": "hf", "repo": "x/y", "file": "v2.pt"},
            }],
        }
        path = os.path.join(udir, "registry.json")
        # Don't clobber a real user file: back it up.
        backup = None
        if os.path.exists(path):
            backup = path + ".bak_test"
            os.replace(path, backup)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(man, fh)
            registry._invalidate_manifest()
            avail = registry.available_models()
            ids = [m["id"] for m in avail["models"]]
            out["default"] = avail["default"]
            out["has_v2"] = "spotunet-base16-v2" in ids
            out["still_has_bundled"] = "spotunet-base16-v1" in ids
        finally:
            os.remove(path)
            if backup:
                os.replace(backup, path)
            registry._invalidate_manifest()

    print("RESULT_JSON", json.dumps(out))
    sys.stdout.flush()
    os._exit(0)
""")


def _run(mode):
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, mode],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON"))
    return json.loads(line[len("RESULT_JSON "):])


class TestNeuralDetect:
    def test_bundled_model_loads(self):
        out = _run("load")
        assert out["is_spotunet"]
        # levels must match whatever the registry declares for the default model
        # (don't hardcode: the default has been upgraded before, e.g. production
        # levels=2 -> anyscale8 levels=3).
        from spyde.models import registry
        entry = next(m for m in registry.list_models()
                     if m["id"] == registry.default_model_id())
        assert out["levels"] == entry["arch"]["levels"]
        assert out["params"] > 0

    def test_detect_localises_planted_spots(self):
        out = _run("detect")
        assert out["n"] > 0
        # Each planted disk has a detection within a few px.
        assert out["worst_planted_dist"] <= 3.0, out

    def test_batch_matches_single(self):
        out = _run("batch")
        assert out["n_frames"] == 2
        assert out["counts_match"], out
        assert out["max_pos_diff"] <= 0.5, out

    def test_registry_fallback(self):
        out = _run("fallback")
        assert out["fell_back"]
        # fallback lands on whatever the registry's current default is
        from spyde.models import registry
        assert out["default"] == registry.default_model_id()

    def test_registry_upgrade_merge(self):
        out = _run("upgrade")
        assert out["has_v2"]
        assert out["still_has_bundled"]
        assert out["default"] == "spotunet-base16-v2"
