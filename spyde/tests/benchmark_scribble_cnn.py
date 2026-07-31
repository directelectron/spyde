"""
benchmark_scribble_cnn.py — the CNN-vs-MLP scribble prototype, A/B'd.

Run directly (it is slow, and torch-CUDA segfaults under the pytest process on
Windows — CLAUDE.md)::

    python -m spyde.tests.benchmark_scribble_cnn
    python -m spyde.tests.benchmark_scribble_cnn --only quality --device cpu

It prints human-readable tables as it goes and one JSON blob at the end, then
``os._exit(0)`` to skip torch's CUDA teardown crash.

It answers exactly two questions and nothing else:

**1. TRAIN TIME.** ``ScribbleClassifier.fit`` is ~0.5 s from a few strokes on the
fixture and the caret's whole tuning loop depends on that. So the CNN's train
time is measured at two scales: the fixture (96×112, a few thousand labelled
pixels) and a REALISTIC one — a 2048² field with a real session's labelled-pixel
counts (~16 k particle / ~33 k support film / ~1.8 k vacuum) spread across it.
A steps sweep gives the accuracy-vs-time curve, so a knee is visible rather than
inferred.

**2. QUALITY, against exact ground truth.** ``particle_movie()`` knows where every
particle is (``particle_truth_at``), which two are the deliberately faint §0.9
probes, and which frame the touching pair merges on. Both engines are trained on
the SAME :class:`~spyde.particles.scribble.LabelStore` and scored on: particle
count, both faint probes found, the merge pair still split at the merge frame,
and foreground IoU.

Plus the forward-pass cost and peak VRAM at 4096², confirming the numbers the
prototype was designed around.

Why the two engines share one evaluator
---------------------------------------
:class:`~spyde.particles.scribble_cnn.ScribbleCNN` deliberately has the same
output contract as :class:`~spyde.particles.scribble.ScribbleClassifier` —
``predict_foreground_boundary`` and ``segment`` — so :func:`evaluate` takes
either one and there is no second scoring path that could flatter either.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from spyde.data.synthetic import (
    MERGE_PAIR,
    ground_truth,
    particle_movie,
    particle_truth_at,
)
from spyde.particles.classical import SegmentParams, split_instances
from spyde.particles.features import FeatureSpec, select_device
from spyde.particles.scribble import LabelStore, ScribbleClassifier, default_classes
from spyde.particles.scribble_cnn import CONFIGS, ScribbleCNN, build_net

#: The frame every quality number is measured on — all nine particles present.
FRAME_T = 12

#: Split parameters, matching ``test_particles_scribble.py``'s gates so the
#: numbers here are comparable to the ones already recorded there.
SPLIT = SegmentParams(min_size=5)


# ── the fixture's scribbles (the same eleven strokes the gates use) ──────────

def _clear_of_particles(shape, pos, radii, present, pad):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    keep = np.ones((h, w), bool)
    for i in np.flatnonzero(present):
        keep &= ((yy - pos[i, 0]) ** 2 +
                 (xx - pos[i, 1]) ** 2) > (radii[i] + pad) ** 2
    return keep


def paint_fixture_scribbles(geom, *, t=FRAME_T, seam=True) -> LabelStore:
    """Copied from ``test_particles_scribble.py``'s ``paint_scribbles`` (+ seam).

    Copied and not imported: the test module builds module-scoped fixtures at
    import time, and a benchmark that drags a pytest fixture graph in is a
    benchmark that measures the fixture graph. The strokes are what matter and
    they are reproduced exactly — four dabs on bright particles, one dab on the
    SMALLER faint probe (index 8; index 7 stays held out), four background
    sweeps, four background rings, and the seams between touching bodies.
    """
    from scipy import ndimage as ndi

    pos, radii, present, faint, shape = geom
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    store = LabelStore(frame_shape=shape, classes=default_classes())

    bright = [i for i in np.flatnonzero(present) if not faint[i]][:4]
    for i in bright:
        store.paint_disc(t, pos[i, 0], pos[i, 1], max(1.5, radii[i] * 0.5), 0)
    store.paint_disc(t, pos[8, 0], pos[8, 1], 1.5, 0)

    far = _clear_of_particles(shape, pos, radii, present, 3.0)
    for (y0, x0, y1, x1) in ((4, 4, 8, 100), (h - 8, 4, h - 4, 100),
                             (40, 4, 60, 8), (20, 40, 40, 46)):
        sweep = np.zeros((h, w), bool)
        sweep[y0:y1, x0:x1] = True
        store.paint(t, sweep & far, 1)

    touching = _clear_of_particles(shape, pos, radii, present, 0.0)
    for i in bright:
        d2 = (yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2
        ring = (d2 > (radii[i] + 2.5) ** 2) & (d2 <= (radii[i] + 4.0) ** 2)
        store.paint(t, ring & touching, 1)

    if seam:
        idx = list(np.flatnonzero(present))
        grown = [ndi.binary_dilation(
            ((yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2) <= radii[i] ** 2,
            iterations=2) for i in idx]
        edge = np.zeros((h, w), bool)
        for a in range(len(grown)):
            for b in range(a + 1, len(grown)):
                edge |= grown[a] & grown[b]
        if edge.any():
            store.paint(t, edge, 3)
    return store


def truth_mask(geom) -> np.ndarray:
    """Exact foreground at :data:`FRAME_T`.

    ``_soft_disc`` is ``0.5*(1 - tanh((r - radius)/0.9))``, which crosses 0.5
    exactly at ``r == radius`` — so the analytic disc IS the 0.5-probability
    contour and the IoU below is against ground truth, not against a rendering
    of it.
    """
    pos, radii, present, _faint, shape = geom
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.zeros((h, w), bool)
    for i in np.flatnonzero(present):
        m |= ((yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2) <= radii[i] ** 2
    return m


# ── the realistic-scale field (train-time question only) ─────────────────────

def big_field(edge: int = 2048, n_particles: int = 420, seed: int = 0):
    """A 2048² particle field with a vacuum hole — the realistic train-time case.

    The fixture is 96×112 with nine particles, which is the right size for a
    ground-truth quality gate and the wrong size for a train-time answer: it
    yields ONE training crop, so it measures the optimiser loop and nothing
    about how the cost grows with how much a user painted. This field is built
    from the same primitives as ``particle_movie`` (a speckled, ramped support
    film plus soft discs) at a size where scribbles genuinely spread out.

    Discs are drawn in their own bounding boxes, not over the full raster: 420
    full-frame adds at 2048² is ~7 s of pure setup and would dominate the thing
    being measured.

    Returns ``(frame, centres, radii, vacuum_rect)``.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    yy = np.mgrid[0:edge, 0:edge][0].astype(np.float32)
    film = 0.10 + 0.12 * (yy / edge)
    film += 0.09 * gaussian_filter(rng.standard_normal((edge, edge)).astype(
        np.float32), 1.6)

    # A hole in the carbon, off-centre and non-square so a transposed or
    # mirrored result is obvious. This is the "vacuum" class's home.
    vac = (int(edge * 0.06), int(edge * 0.30), int(edge * 0.62), int(edge * 0.94))
    film[vac[0]:vac[1], vac[2]:vac[3]] *= 0.05

    centres = rng.uniform(12, edge - 12, size=(n_particles, 2))
    radii = rng.uniform(5.0, 12.0, size=n_particles)
    amps = rng.uniform(0.45, 1.0, size=n_particles)
    for (cy, cx), r, a in zip(centres, radii, amps):
        # CLIP the window, and build the coordinate grid from the CLIPPED
        # bounds. Centres are drawn 12 px from the edge but radii reach 12 with
        # +4/+5 padding, so a disc can overhang: `film[y0:y1]` then clips
        # silently while `np.mgrid[y0:y1]` does not, and the add fails with a
        # (31,32)-vs-(32,32) broadcast error on whichever particle lands there.
        y0, y1 = max(0, int(cy - r - 4)), min(edge, int(cy + r + 5))
        x0, x1 = max(0, int(cx - r - 4)), min(edge, int(cx + r + 5))
        if y1 <= y0 or x1 <= x0:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
        film[y0:y1, x0:x1] += a * 0.5 * (1.0 - np.tanh((d - r) / 0.9))
    film += (0.015 * rng.standard_normal((edge, edge))).astype(np.float32)
    return film, centres, radii, vac


def paint_big_scribbles(shape, centres, radii, vac, *,
                        want=(16_259, 32_651, 1_816), seed: int = 0) -> LabelStore:
    """Scribbles matching a REAL session's per-class pixel counts.

    *want* is ``(particle, support film, vacuum)`` and defaults to the counts
    read off an actual SpyDE session. Matching the counts is the point: train
    time is a function of how much was painted and how far apart, and a
    fixture-sized scribble set answers a question nobody asked.

    Three classes and no boundary — the real session had none, and adding one
    here would inflate the crop count against a scenario that did not have it.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    store = LabelStore(frame_shape=shape, classes=default_classes())

    order = rng.permutation(len(centres))
    got = 0
    for i in order:
        if got >= want[0]:
            break
        got = store.paint_disc(0, centres[i, 0], centres[i, 1],
                               radii[i] * 0.85, 0)

    # Film sweeps: wide strokes at random places, skipping any pixel within a
    # particle radius, until the target count is reached.
    keep = np.ones((h, w), bool)
    for (cy, cx), r in zip(centres, radii):
        y0, y1 = max(0, int(cy - r - 4)), min(h, int(cy + r + 5))
        x0, x1 = max(0, int(cx - r - 4)), min(w, int(cx + r + 5))
        gy, gx = np.mgrid[y0:y1, x0:x1]
        keep[y0:y1, x0:x1] &= ((gy - cy) ** 2 + (gx - cx) ** 2) > (r + 3.0) ** 2
    keep[vac[0]:vac[1], vac[2]:vac[3]] = False        # that is vacuum, not film

    got = 0
    while got < want[1]:
        y0 = int(rng.integers(0, h - 40))
        x0 = int(rng.integers(0, w - 240))
        sweep = np.zeros((h, w), bool)
        sweep[y0:y0 + 6, x0:x0 + 240] = True
        got = store.paint(0, sweep & keep, 1)

    side = int(np.sqrt(want[2]))
    vy, vx = (vac[0] + vac[1]) // 2, (vac[2] + vac[3]) // 2
    box = np.zeros((h, w), bool)
    box[vy - side // 2:vy + side // 2, vx - side // 2:vx + side // 2] = True
    store.paint(0, box, 2)
    return store


# ── scoring ──────────────────────────────────────────────────────────────────

def evaluate(engine, movie_data, gt, geom) -> dict:
    """Score a trained engine against exact ground truth. Engine-agnostic.

    Both engines expose ``predict_foreground_boundary`` and ``segment``, which
    is the design claim being tested — so this function cannot tell them apart,
    and neither can :func:`spyde.particles.classical.split_instances`.
    """
    pos, radii, present, faint, shape = geom
    out: dict = {}

    t0 = time.perf_counter()
    fg, bnd = engine.predict_foreground_boundary(movie_data[FRAME_T])
    out["predict_s"] = time.perf_counter() - t0
    out["has_boundary"] = bnd is not None

    truth = truth_mask(geom)
    pred = fg > 0.5
    inter = int((pred & truth).sum())
    union = int((pred | truth).sum())
    out["iou"] = inter / max(1, union)
    out["truth_px"] = int(truth.sum())
    out["pred_px"] = int(pred.sum())

    labels = split_instances(fg, SPLIT, boundary=bnd)
    out["n_particles"] = int(labels.max())
    out["n_truth"] = int(present.sum())

    def _hit(lab, i):
        return int(lab[int(round(pos[i, 0])), int(round(pos[i, 1]))]) != 0

    out["faint_found"] = [int(i) for i in np.flatnonzero(faint) if _hit(labels, i)]
    out["faint_total"] = int(faint.sum())
    out["bright_missed"] = [int(i) for i in np.flatnonzero(present & ~faint)
                            if not _hit(labels, i)]

    # The merge frame: the pair deliberately overlap, and a boundary-trained
    # head is supposed to keep them apart where the watershed has to guess.
    mt = int(gt["merge_frame"])
    out["merge_frame"] = mt
    if mt >= 0:
        mpos, _r, mpresent = particle_truth_at(gt, mt)
        mfg, mbnd = engine.predict_foreground_boundary(movie_data[mt])
        mlab = split_instances(mfg, SPLIT, boundary=mbnd)
        a, b = MERGE_PAIR
        la = int(mlab[int(round(mpos[a, 0])), int(round(mpos[a, 1]))])
        lb = int(mlab[int(round(mpos[b, 0])), int(round(mpos[b, 1]))])
        out["merge_labels"] = [la, lb]
        out["merge_split"] = bool(la and lb and la != lb)
        out["merge_frame_particles"] = int(mlab.max())
        out["merge_frame_truth"] = int(mpresent.sum())
    return out


def _fmt(tag: str, rep: dict, sc: dict) -> str:
    return (f"  {tag:<26} train {rep['total_s']:7.2f}s | "
            f"IoU {sc['iou']:.3f} | n={sc['n_particles']:>3}/{sc['n_truth']} | "
            f"faint {len(sc['faint_found'])}/{sc['faint_total']} | "
            f"merge-split {str(sc.get('merge_split')):<5} | "
            f"predict {sc['predict_s']*1000:6.0f} ms")


# ── sections ─────────────────────────────────────────────────────────────────

def section_forward(device) -> dict:
    """4096² forward-pass cost and peak VRAM, both configs, tiled and not.

    Confirms the numbers the prototype was designed around and records the peak
    allocation, which is the thing that actually decides whether the whole-frame
    path is usable on a 12 GB card.
    """
    import torch

    from spyde.device_lock import accelerator_lock

    print("\n=== forward pass, 4096x4096, fp32 ===")
    out: dict = {}
    if device.type != "cuda":
        print("  (skipped — no CUDA device)")
        return out

    frame = np.random.default_rng(0).standard_normal(
        (4096, 4096)).astype(np.float32)
    for name, (base, levels) in CONFIGS.items():
        net = build_net(3, base=base, levels=levels).to(device).eval()
        params = sum(p.numel() for p in net.parameters())
        for mode in ("tiled-1024", "whole"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            clf = ScribbleCNN(base=base, levels=levels, device=device,
                              tile=1024 if mode == "tiled-1024" else 8192)
            clf._net, clf.classes = net, []
            try:
                with accelerator_lock(device), torch.no_grad():
                    for run in range(2):              # discard the cold run
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        h, w = frame.shape
                        for (y0, y1, x0, x1, *_r) in clf._tiles(h, w):
                            t = torch.as_tensor(frame[y0:y1, x0:x1],
                                                device=device)[None, None]
                            torch.softmax(net(t), dim=1)
                        torch.cuda.synchronize()
                        dt = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 2 ** 20
                out[f"{name}/{mode}"] = {"s": dt, "peak_MiB": peak,
                                         "params": params}
                print(f"  {name:<6} base={base} levels={levels} "
                      f"{params/1e3:7.1f}k params  {mode:<11} "
                      f"{dt:6.3f} s  peak {peak:7.0f} MiB")
            except torch.cuda.OutOfMemoryError as e:
                out[f"{name}/{mode}"] = {"error": "OOM"}
                print(f"  {name:<6} {mode:<11} OOM ({str(e)[:60]})")
            finally:
                torch.cuda.empty_cache()
        del net
    return out


def big_truth(shape, centres, radii) -> np.ndarray:
    """Exact foreground for :func:`big_field`, drawn in bounding boxes only.

    Same 0.5-contour identity as :func:`truth_mask` — ``_soft_disc`` crosses 0.5
    at ``r == radius`` — so this is ground truth, not a rendering of it.
    """
    m = np.zeros(shape, bool)
    for (cy, cx), r in zip(centres, radii):
        y0, y1 = max(0, int(cy - r - 2)), min(shape[0], int(cy + r + 3))
        x0, x1 = max(0, int(cx - r - 2)), min(shape[1], int(cx + r + 3))
        gy, gx = np.mgrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] |= ((gy - cy) ** 2 + (gx - cx) ** 2) <= r * r
    return m


def section_train(device, steps: int) -> dict:
    """Train time at fixture scale AND at realistic scale, MLP vs both CNNs.

    The realistic block also scores IoU and particle count against the field's
    own exact truth. That is not scope creep: the fixture gives the CNN 1.4 k
    labelled pixels in ONE crop, which is a regime where a conv net has almost
    nothing to learn from, so a bad fixture score does not by itself say the
    approach is bad. The 2048² field with a real session's ~50 k labels across
    ~50 crops is the regime the user actually paints in, and it costs nothing
    extra to score the models that were trained here anyway.
    """
    print("\n=== train time ===")
    out: dict = {}

    s = particle_movie()
    gt = ground_truth(s)
    pos, radii, present = particle_truth_at(gt, FRAME_T)
    geom = (pos, radii, present, np.asarray(gt["p_faint"], bool),
            tuple(gt["frame_shape"]))
    store = paint_fixture_scribbles(geom)
    frames = {FRAME_T: s.data[FRAME_T]}
    print(f"  fixture  96x112, {len(store)} labelled px, {store.counts()}")
    out["fixture"] = _train_row(store, frames, device, steps)

    t0 = time.perf_counter()
    big, centres, big_radii, vac = big_field()
    bstore = paint_big_scribbles(big.shape, centres, big_radii, vac)
    truth = big_truth(big.shape, centres, big_radii)
    print(f"\n  realistic 2048x2048 ({time.perf_counter() - t0:.1f} s to build),"
          f" {len(bstore)} labelled px, {bstore.counts()}")
    out["realistic"] = _train_row(bstore, {0: big}, device, steps,
                                  truth=truth, frame=big)
    return out


def _train_row(store, frames, device, steps, *, truth=None, frame=None) -> dict:
    from scipy import ndimage as ndi

    row: dict = {}
    n_truth = int(ndi.label(truth)[1]) if truth is not None else 0

    def score_into(cell: dict, engine) -> str:
        if truth is None:
            return ""
        t0 = time.perf_counter()
        fg, bnd = engine.predict_foreground_boundary(frame)
        dt = time.perf_counter() - t0
        pred = fg > 0.5
        iou = int((pred & truth).sum()) / max(1, int((pred | truth).sum()))
        n = int(split_instances(fg, SPLIT, boundary=bnd).max())
        cell.update(iou=iou, n_particles=n, n_truth=n_truth, predict_s=dt)
        return f"  | IoU {iou:.3f} n={n}/{n_truth} predict {dt:5.2f} s"

    mlp = ScribbleClassifier(FeatureSpec(), device=device, seed=0)
    t0 = time.perf_counter()
    rep = mlp.fit(store, frames)
    cell = row["mlp"] = {"total_s": time.perf_counter() - t0,
                         "featurise_s": rep["featurise_s"],
                         "fit_s": rep["fit_s"], "n_pixels": rep["n_pixels"]}
    tail = score_into(cell, mlp)
    print(f"    MLP (36ch + head)   {cell['total_s']:7.2f} s "
          f"(featurise {rep['featurise_s']:.2f} + fit {rep['fit_s']:.2f})"
          f"{tail}")

    for name, (base, levels) in CONFIGS.items():
        cnn = ScribbleCNN(base=base, levels=levels, steps=steps, device=device,
                          seed=0)
        t0 = time.perf_counter()
        rep = cnn.fit(store, frames)
        cell = row[name] = {"total_s": time.perf_counter() - t0,
                            "crops_s": rep["crops_s"], "fit_s": rep["fit_s"],
                            "n_crops": rep["n_crops"], "steps": steps,
                            "params": rep["params"],
                            "train_accuracy": rep["train_accuracy"]}
        tail = score_into(cell, cnn)
        print(f"    CNN {name:<6} b{base}/L{levels}  {cell['total_s']:7.2f}"
              f" s (crops {rep['crops_s']:.2f} + fit {rep['fit_s']:.2f}), "
              f"{rep['n_crops']} crops, {steps} steps, "
              f"acc {rep['train_accuracy']:.3f}{tail}")
    return row


def section_quality(device, steps: int) -> dict:
    """Both engines, same scribbles, scored against exact truth.

    Run TWICE, with and without the seam strokes, because the two exercise
    different downstream routes and only one of them is what the shipped §0.9
    gate measures. Without a boundary both engines go through the watershed;
    with one they both take ``split_instances``' connected-components route. An
    A/B that reported only the boundary route would be comparing the CNN's
    boundary head against the MLP's, and an A/B that reported only the watershed
    route would never test the merge split at all.
    """
    s = particle_movie()
    gt = ground_truth(s)
    pos, radii, present = particle_truth_at(gt, FRAME_T)
    geom = (pos, radii, present, np.asarray(gt["p_faint"], bool),
            tuple(gt["frame_shape"]))
    frames = {FRAME_T: s.data[FRAME_T]}
    out: dict = {}

    for seam in (False, True):
        route = "boundary route" if seam else "watershed route"
        print(f"\n=== quality vs ground truth (frame 12, {route}) ===")
        store = paint_fixture_scribbles(geom, seam=seam)
        print(f"  labels: {store.counts()}")
        block: dict = {}

        mlp = ScribbleClassifier(FeatureSpec(), device=device, seed=0)
        t0 = time.perf_counter()
        rep = mlp.fit(store, frames)
        rep["total_s"] = time.perf_counter() - t0
        sc = evaluate(mlp, s.data, gt, geom)
        block["mlp"] = {"train": rep, "score": sc}
        print(_fmt("MLP (36ch + head)", rep, sc))

        for name, (base, levels) in CONFIGS.items():
            cnn = ScribbleCNN(base=base, levels=levels, steps=steps,
                              device=device, seed=0)
            t0 = time.perf_counter()
            rep = cnn.fit(store, frames)
            rep["total_s"] = time.perf_counter() - t0
            sc = evaluate(cnn, s.data, gt, geom)
            block[name] = {"train": rep, "score": sc}
            print(_fmt(f"CNN {name} b{base}/L{levels}", rep, sc))
        out["seam" if seam else "no_seam"] = block
    return out


def section_curve(device, sweep) -> dict:
    """Accuracy-vs-time: is there a knee worth stopping at?"""
    print("\n=== accuracy vs train time (CNN, fixture) ===")
    s = particle_movie()
    gt = ground_truth(s)
    pos, radii, present = particle_truth_at(gt, FRAME_T)
    geom = (pos, radii, present, np.asarray(gt["p_faint"], bool),
            tuple(gt["frame_shape"]))
    store = paint_fixture_scribbles(geom)
    frames = {FRAME_T: s.data[FRAME_T]}

    out: dict = {}
    for name, (base, levels) in CONFIGS.items():
        rows = []
        for steps in sweep:
            cnn = ScribbleCNN(base=base, levels=levels, steps=steps,
                              device=device, seed=0)
            t0 = time.perf_counter()
            cnn.fit(store, frames)
            total = time.perf_counter() - t0
            sc = evaluate(cnn, s.data, gt, geom)
            rows.append({"steps": steps, "total_s": total, "iou": sc["iou"],
                         "n_particles": sc["n_particles"],
                         "faint": len(sc["faint_found"]),
                         "merge_split": sc.get("merge_split")})
            print(f"  {name:<6} steps {steps:>4}  {total:6.2f} s  "
                  f"IoU {sc['iou']:.3f}  n={sc['n_particles']:>3}  "
                  f"faint {len(sc['faint_found'])}/{sc['faint_total']}  "
                  f"merge-split {sc.get('merge_split')}")
        out[name] = rows
    return out


# ── entry point ──────────────────────────────────────────────────────────────

def _warm(device) -> None:
    """Pay the one-time CUDA costs BEFORE anything is timed.

    Cold CUDA context creation, cuDNN algorithm selection and the feature
    stack's first kernel launches are a ~1.5 s one-off that lands entirely on
    whichever engine happens to run first — which measured as the MLP's
    "featurise 1.47 s" on a 96×112 frame, three orders of magnitude above its
    warm cost. Warming both stacks here is the difference between an A/B and a
    coin toss about ordering.
    """
    import torch
    import torch.nn.functional as F

    from spyde.particles.features import sample_features

    if device.type == "cuda":
        torch.cuda.init()
    net = build_net(3, base=16, levels=2).to(device)
    x = torch.zeros((2, 1, 64, 64), device=device)
    y = torch.zeros((2, 64, 64), dtype=torch.long, device=device)
    F.cross_entropy(net(x), y).backward()
    sample_features(np.zeros((64, 64), np.float32), np.arange(64),
                    FeatureSpec(), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="all",
                    choices=("all", "forward", "train", "quality", "curve"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sweep", default="50,100,200,300,600")
    ap.add_argument("--json", default=None, help="write the JSON blob here too")
    args = ap.parse_args(argv)

    device = select_device(args.device)
    print(f"device: {device}")
    _warm(device)
    results: dict = {"device": str(device), "steps": args.steps}

    if args.only in ("all", "forward"):
        results["forward"] = section_forward(device)
    if args.only in ("all", "train"):
        results["train"] = section_train(device, args.steps)
    if args.only in ("all", "quality"):
        results["quality"] = section_quality(device, args.steps)
    if args.only in ("all", "curve"):
        results["curve"] = section_curve(
            device, [int(v) for v in args.sweep.split(",")])

    blob = json.dumps(results, indent=2, default=float)
    print("\n=== JSON ===")
    print(blob)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(blob)
    sys.stdout.flush()
    # torch's CUDA teardown crashes on exit here (CLAUDE.md); the numbers are
    # already printed, so leave before it runs.
    os._exit(0)


if __name__ == "__main__":
    main()
