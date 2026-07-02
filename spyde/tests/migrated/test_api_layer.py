"""
test_api_layer.py — the spyde.api script-parity contract (NOTEBOOK_PARITY_PLAN §5).

Guards three things:

1. **The import graph**: ``spyde.api`` must never import ``spyde.backend`` or
   ``spyde.drawing`` — it is the layer that runs where no UI stack exists.
   Checked statically (source scan) AND dynamically (fresh subprocess import).
2. **Script parity smoke**: the api functions run headless on synthetic data
   and return the same self-contained result objects the app produces, with
   a provenance record stamped.
3. **Signature stability**: the ``client=`` seam the api relies on exists on
   the batch cores.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _synthetic_4d(nav=(4, 4), sig=(64, 64),
                  spots=((20, 20), (44, 44), (20, 44), (32, 32))):
    """Tiny 4D STEM signal with fixed bright disks — enough for nxcorr.

    Noise is kept tiny (1% of the disks) so peak detection is deterministic:
    every pattern yields exactly these spots, and the strain fit against the
    (identical) reference is ~zero. The (32, 32) spot sits at k=(0, 0) — the
    zero beam — so strain references keep the three off-zero spots."""
    import hyperspy.api as hs

    data = np.zeros(nav + sig, dtype=np.float32)
    yy, xx = np.mgrid[: sig[0], : sig[1]]
    for cy, cx in spots:
        disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 9
        data[..., disk] += 100.0
    rng = np.random.default_rng(42)
    data += rng.random(data.shape, dtype=np.float32)
    s = hs.signals.Signal2D(data)
    for ax in s.axes_manager.signal_axes:
        ax.scale = 0.01
        ax.offset = -0.32
    return s


class TestApiImportGraph:
    def test_api_source_never_imports_backend_or_drawing(self):
        # AST-level: no import statement anywhere in spyde/api.py (including
        # function bodies — the lazy imports) may name backend/drawing.
        import ast
        path = Path(__file__).parents[2] / "api.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = ("spyde.backend", "spyde.drawing")
        offenders = []
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            offenders += [n for n in names
                          if any(n.startswith(b) for b in banned)]
        assert not offenders, \
            f"spyde.api imports UI modules (script-parity rule): {offenders}"

    def test_api_import_pulls_no_ui_stack(self):
        # Fresh interpreter: importing spyde.api (and spyde itself, which
        # re-exports it) must not load backend/drawing modules.
        code = (
            "import sys, json\n"
            "import spyde.api\n"
            "bad = sorted(m for m in sys.modules\n"
            "             if m.startswith('spyde.backend') or m.startswith('spyde.drawing'))\n"
            "print(json.dumps(bad))\n"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr
        bad = json.loads(out.stdout.strip().splitlines()[-1])
        assert bad == [], f"spyde.api import loaded UI modules: {bad}"


class TestApiFindVectorsAndDownstream:
    @pytest.fixture(scope="class")
    def vectors(self):
        from spyde import api
        sig = _synthetic_4d()
        vecs = api.find_vectors(sig, method="nxcorr", kernel_radius=3,
                                threshold=0.7, min_distance=5)
        assert vecs is not None
        return sig, vecs

    def test_find_vectors_headless(self, vectors):
        _, vecs = vectors
        from spyde.signals import SpyDEDiffractionVectors
        assert isinstance(vecs, SpyDEDiffractionVectors)
        assert vecs.nav_shape == (4, 4)
        assert len(vecs.flat_buffer) > 0
        # every position sees the three synthetic disks
        assert int(np.min(vecs.count_map())) >= 3

    def test_provenance_stamped(self, vectors):
        _, vecs = vectors
        assert vecs.provenance is not None
        assert vecs.provenance["action"] == "find_vectors"
        assert vecs.provenance["params"]["method"] == "nxcorr"
        assert "spyde_version" in vecs.provenance

    def test_strain_map_default_reference(self, vectors):
        from spyde import api
        _, vecs = vectors
        sf = api.strain_map(vecs)
        assert sf.exx.shape == vecs.nav_shape
        # identical patterns everywhere → (near-)zero strain wherever fit
        assert float(np.nanmedian(np.abs(sf.exx))) < 0.02
        assert sf.provenance["action"] == "strain_map"
        assert sf.provenance["params"]["n_ref"] >= 2

    def test_strain_reference_is_zero_beam_filtered(self, vectors):
        # The wizard physics: a reference containing the zero beam must have
        # it stripped before the fit (same helper, same numbers).
        from spyde.actions.strain_mapping import zero_beam_filtered
        g = np.array([[0.001, 0.0], [0.2, 0.0], [0.0, 0.2]])
        out = zero_beam_filtered(g)
        assert len(out) == 2
        assert float(np.linalg.norm(out, axis=1).min()) > 0.05

    def test_vector_virtual_image(self, vectors):
        from spyde import api
        _, vecs = vectors
        # integrate around one synthetic spot (data coords)
        kx, ky = vecs.flat_buffer[0, 2], vecs.flat_buffer[0, 3]
        vvi = api.vector_virtual_image(vecs, cx=float(kx), cy=float(ky), r=0.05)
        assert vvi.shape == vecs.nav_shape
        assert float(vvi.max()) > 0

    def test_virtual_image_signal2d(self, vectors):
        from spyde import api
        sig, _ = vectors
        vi = api.virtual_image(sig, cx=0.0, cy=0.0, r=0.3, calculation="sum")
        assert vi.data.shape == (4, 4)
        assert float(vi.data.min()) > 0
        prov = vi.metadata.General.spyde_provenance
        assert prov["action"] == "virtual_image"

    def test_virtual_image_empty_detector_raises(self, vectors):
        from spyde import api
        sig, _ = vectors
        with pytest.raises(ValueError):
            api.virtual_image(sig, cx=99.0, cy=99.0, r=0.01)


class TestApiCenterZeroBeam:
    def test_center_zero_beam_smoke(self):
        from spyde import api
        sig = _synthetic_4d(nav=(2, 2), spots=((30, 34),))  # off-center beam
        out = api.center_zero_beam(sig, method="center_of_mass")
        assert out is not sig
        assert out.data.shape == sig.data.shape
        prov = out.metadata.General.spyde_provenance
        assert prov["action"] == "center_zero_beam"
        # the (single) bright disk should now sit at the pattern center
        mean_dp = out.data.mean(axis=(0, 1))
        cy, cx = np.unravel_index(int(np.argmax(mean_dp)), mean_dp.shape)
        assert abs(cy - 32) <= 2 and abs(cx - 32) <= 2


class TestClientSeam:
    """The api layer's client= plumbing exists on every batch core."""

    def test_batch_cores_accept_client(self):
        from spyde.actions.find_vectors import _do_compute_vectors
        from spyde.actions.orientation_compute import _do_compute_orientations
        from spyde.actions.vector_orientation import (
            compute_vector_orientation_chunked,
        )
        for fn in (_do_compute_vectors, _do_compute_orientations,
                   compute_vector_orientation_chunked):
            params = inspect.signature(fn).parameters
            assert "client" in params, f"{fn.__name__} lost the client= seam"
            assert params["client"].default is None

    def test_api_surface_complete(self):
        from spyde import api
        for name in api.__all__:
            assert callable(getattr(api, name))
        # orientation entry points exist with the documented signature pieces
        sig = inspect.signature(api.orientation_map)
        assert "client" in sig.parameters and "phases" in sig.parameters
        sig = inspect.signature(api.vector_orientation_map)
        assert "gpu" in sig.parameters
