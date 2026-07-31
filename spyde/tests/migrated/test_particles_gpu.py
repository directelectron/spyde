"""
The scribble engine on a REAL accelerator: does the GPU agree with the CPU?

Everything else about this engine is tested on the CPU (``select_device("cpu")``
throughout ``test_particles_scribble.py``) because torch-CUDA work segfaults
under the pytest process on Windows — a harness interaction, not a code defect
(CLAUDE.md § GPU Computing). That leaves exactly one thing unchecked, and it is
the thing that matters most for a path that ships GPU-first: **that the device
and the CPU produce the same particles.**

So this file runs the comparison in a **subprocess** that prints a JSON summary
and hard-exits, following ``test_vector_orientation_gpu.py``. Skipped entirely
when no GPU is present, which is CI and many user machines.

What is and is not asserted, and why
------------------------------------
Foreground agreement is held to a tight IoU rather than bit-equality. The two
devices run different kernels — cuDNN's convolutions accumulate in a different
order from the CPU's — so the probability maps differ in the last few bits, and
a threshold at 0.5 turns that into a handful of disagreeing pixels. Demanding
bit-equality here would be demanding that two float32 implementations coincide,
which is not a property either device offers.

The **count** is held exactly, because that is the number a user acts on. The
boundary route is the more fragile of the two here and deliberately so: it takes
connected components of ``fg & ~boundary``, so a single pixel flipping along a
seam can join or separate two bodies, where the watershed would have absorbed it.
If that ever becomes flaky the honest response is to record the sensitivity, not
to loosen the count assertion.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from spyde.particles.features import gpu_available

pytestmark = pytest.mark.skipif(
    not gpu_available(), reason="no torch GPU (CUDA / MPS) available")


_DRIVER = textwrap.dedent("""
    import json, sys, os
    import numpy as np
    from scipy import ndimage as ndi

    from spyde.data.synthetic import (particle_movie, ground_truth,
                                      particle_truth_at)
    from spyde.particles import SegmentParams, split_instances
    from spyde.particles.features import FeatureSpec, select_device
    from spyde.particles.scribble import (LabelStore, ScribbleClassifier,
                                          default_classes)

    T = 12
    s = particle_movie()
    gt = ground_truth(s)
    frame = np.asarray(s.data[T])
    pos, radii, present = particle_truth_at(gt, T)
    faint = np.asarray(gt["p_faint"], bool)
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w]

    # Scribbles: dabs on particles (incl. one faint probe), background sweeps,
    # and seam strokes along the joins between touching bodies.
    store = LabelStore(frame_shape=(h, w), classes=default_classes())
    idx = list(np.flatnonzero(present))
    discs = [((yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2) <= radii[i] ** 2
             for i in idx]
    for i in idx:
        store.paint_disc(T, pos[i, 0], pos[i, 1], max(1.5, radii[i] * 0.5), 0)
    far = np.ones((h, w), bool)
    for i in idx:
        far &= ((yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2) > (radii[i] + 3.) ** 2
    store.paint(T, far, 1)
    seam = np.zeros((h, w), bool)
    grown = [ndi.binary_dilation(d, iterations=2) for d in discs]
    for a in range(len(grown)):
        for b in range(a + 1, len(grown)):
            seam |= grown[a] & grown[b]
    if seam.any():
        store.paint(T, seam, 3)

    def run(dev):
        clf = ScribbleClassifier(FeatureSpec(), device=select_device(dev), seed=0)
        rep = clf.fit(store, {T: frame})
        fg, bnd = clf.predict_foreground_boundary(frame)
        p = SegmentParams(min_size=5)
        return dict(
            device=str(clf.device),
            has_boundary=bool(rep["has_boundary"]),
            fg=(fg > 0.5),
            bnd=(None if bnd is None else bnd > 0.5),
            by_boundary=split_instances(fg, p, boundary=bnd),
            by_watershed=split_instances(fg, p),
        )

    g = run(None)          # auto-selects CUDA / MPS
    c = run("cpu")
    assert g["device"] != "cpu", "no accelerator was selected"

    def iou(a, b):
        u = int((a | b).sum())
        return 1.0 if u == 0 else int((a & b).sum()) / u

    def areas(lab):
        cnt = np.bincount(lab.ravel())[1:]
        return np.sort(cnt[cnt > 0]).tolist()

    def found(lab, sel):
        return [bool(lab[int(round(pos[i, 0])), int(round(pos[i, 1]))])
                for i in np.flatnonzero(sel)]

    out = dict(
        device=g["device"],
        has_boundary=g["has_boundary"] and c["has_boundary"],
        fg_iou=iou(g["fg"], c["fg"]),
        bnd_iou=iou(g["bnd"], c["bnd"]),
        n_boundary_gpu=int(g["by_boundary"].max()),
        n_boundary_cpu=int(c["by_boundary"].max()),
        n_watershed_gpu=int(g["by_watershed"].max()),
        n_watershed_cpu=int(c["by_watershed"].max()),
        areas_boundary_gpu=areas(g["by_boundary"]),
        areas_boundary_cpu=areas(c["by_boundary"]),
        faint_gpu=found(g["by_boundary"], faint),
        faint_cpu=found(c["by_boundary"], faint),
        n_faint=int(faint.sum()),
    )
    print("RESULT_JSON", json.dumps(out))
    sys.stdout.flush()
    # torch + CUDA teardown segfaults at interpreter exit on Windows (harmless,
    # and after the result is out). Hard-exit so the parent sees rc == 0.
    os._exit(0)
""")


@pytest.fixture(scope="module")
def result():
    """One subprocess for the whole module — it trains twice and is ~30 s."""
    proc = subprocess.run([sys.executable, "-c", _DRIVER],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    line = next(l for l in proc.stdout.splitlines()
                if l.startswith("RESULT_JSON"))
    return json.loads(line[len("RESULT_JSON "):])


class TestGpuCpuAgreement:
    def test_an_accelerator_was_actually_used(self, result):
        """Guards the whole file from passing vacuously by running CPU twice."""
        assert result["device"] in ("cuda", "mps")

    def test_the_boundary_class_trained_on_both(self, result):
        assert result["has_boundary"]

    def test_the_foreground_maps_agree(self, result):
        assert result["fg_iou"] > 0.99, (
            f"GPU/CPU foreground IoU {result['fg_iou']:.4f}")

    def test_the_boundary_maps_agree(self, result):
        assert result["bnd_iou"] > 0.95, (
            f"GPU/CPU boundary IoU {result['bnd_iou']:.4f}")

    def test_the_particle_count_is_the_same_on_both(self, result):
        """The number the user acts on. Checked on both routes, because the
        boundary route is the more fragile one — see the module docstring."""
        assert result["n_boundary_gpu"] == result["n_boundary_cpu"], result
        assert result["n_watershed_gpu"] == result["n_watershed_cpu"], result

    def test_the_areas_agree(self, result):
        assert (result["areas_boundary_gpu"]
                == result["areas_boundary_cpu"]), result

    def test_the_faint_probes_are_found_on_both(self, result):
        """§0.9 on the device: the GPU path may not quietly lose a faint
        particle the CPU path finds."""
        assert all(result["faint_gpu"]), result
        assert result["faint_gpu"] == result["faint_cpu"], result
