"""compute_pdcusi_vectors.py — find the diffraction vectors for the PdCuSi
in-situ crystallization example, and save them beside the dataset.

    python -m scripts.compute_pdcusi_vectors [out.npz]

The dataset (em-database ``PdCuSiCrystallization``) is a 5-D in-situ series:
400 time steps x 47x39 real space x 128x128 diffraction — 733,200 patterns,
5.7 GB on disk. It is far too big to recompute inside a docs build or a test,
so this runs ONCE and writes a compact ``SpyDEDiffractionVectors`` npz that the
report generator (``scripts/gen_report_pdcusi.py``) reads back with
``build_vectors_result_tree``.

Parameters match what the Find Vectors caret sends for the settings this was
run at — spot size 8 px, threshold 0.35, neural detector. The caret's single
"Spot size (px)" slider drives BOTH ``kernel_radius`` and (for the neural
method) ``spot_radius``, and derives ``min_distance`` as ``round(radius / 2)``;
see FindVectorsWizard.tsx `params()`. Passing only one of them would quietly
detect at a different scale than the app does.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# The caret's payload for spot size 8 / threshold 0.30 on the neural detector,
# with the stage-2 NEIGHBOUR REFINE on.
#
# The threshold is LOWER than a plain run would want, deliberately: `persistence`
# re-scores every peak against the peaks found at its scan NEIGHBOURS and drops
# the ones no neighbour confirms (models/refine.py). A real reflection persists
# across adjacent probe positions; a detector artefact does not. So the pair
# trades a permissive first pass for a spatial second opinion, which is both
# stricter about noise and cheaper to embed than 0.35 alone was.
SPOT_RADIUS_PX = 8
THRESHOLD = 0.30
NEIGHBOUR_REFINE = True

PARAMS = dict(
    method="neural",
    model_id="",
    sigma=0.0,                                   # neural never applies nav blur
    kernel_radius=SPOT_RADIUS_PX,
    spot_radius=float(SPOT_RADIUS_PX),
    min_distance=max(1, round(SPOT_RADIUS_PX / 2)),
    threshold=THRESHOLD,
    subpixel=True,
    bg_sigma=12.0,
    dog_sigma1=0.8, dog_sigma2=2.0,
    beamstop_auto=False, beamstop_dilate=5,
    persistence=NEIGHBOUR_REFINE, show_transform=False,
)

def default_out() -> str:
    """Beside the dataset in the em-database data dir — NOT in the repo.

    The buffer is ~80 MB for this run: too big to commit, and it must not land
    under ``docs-site/public/`` where Vite would publish it. It is a build input
    (the report generator reads it), and it belongs next to the 5.7 GB file it
    was computed from."""
    from spyde.backend import example_catalogue as catalogue
    return os.path.join(catalogue.data_dir(), "PdCuSiCrystallization-vectors.npz")


def _load():
    """The example, lazy and storage-aligned (Live-Display §1: chunks must span
    whole signal frames, which this file already does)."""
    import hyperspy.api as hs
    from spyde.backend import example_catalogue as catalogue

    ds = catalogue.resolve("PdCuSiCrystallization")
    if ds is None:
        raise SystemExit("em-database has no PdCuSiCrystallization dataset")
    path = ds.filepath()
    if not path or not os.path.exists(path):
        raise SystemExit(
            "PdCuSiCrystallization is not downloaded — open it once from the "
            "Examples menu, or call dataset.fetch(), then re-run.")
    print(f"[pdcusi] loading {path}", flush=True)
    sig = hs.load(path, lazy=True)
    sig.metadata.General.title = "PdCuSi crystallization (in-situ)"
    return sig


def main(out_path: str) -> int:
    from spyde.backend.heavy_imports import ensure_heavy_imports
    ensure_heavy_imports()
    from spyde.actions.find_vectors import _do_compute_vectors
    from spyde import models

    sig = _load()
    am = sig.axes_manager
    n_pat = int(np.prod([a.size for a in am.navigation_axes]))
    print(f"[pdcusi] shape={sig.data.shape} chunks={sig.data.chunksize} "
          f"patterns={n_pat:,}", flush=True)
    print(f"[pdcusi] params={PARAMS}", flush=True)

    # Resolve the model to a local file BEFORE the workers need it, exactly as
    # the action does (_ensure_model_local) — otherwise every worker races to
    # download it.
    models.ensure_local(None)

    t0 = time.monotonic()
    done = {"n": 0}

    def _on_chunk(nav_slices, block):
        done["n"] += 1
        if done["n"] % 20 == 0:
            el = time.monotonic() - t0
            print(f"[pdcusi] {done['n']} chunks in {el / 60:.1f} min",
                  flush=True)

    vecs = _do_compute_vectors(sig, PARAMS, on_chunk_block=_on_chunk)
    if vecs is None:
        print("[pdcusi] compute returned nothing", flush=True)
        return 1
    el = time.monotonic() - t0
    n = int(len(vecs.flat_buffer))
    print(f"[pdcusi] {n:,} vectors in {el / 60:.1f} min "
          f"({n / max(1, n_pat):.2f} per pattern)", flush=True)
    print(f"[pdcusi] full_nav_shape={vecs.full_nav_shape} n_time={vecs.n_time}",
          flush=True)
    series = np.asarray(vecs.count_map_series())
    totals = series.reshape(series.shape[0], -1).sum(axis=1)
    print(f"[pdcusi] per-slice totals: min={totals.min()} max={totals.max()} "
          f"first={totals[:5].tolist()} last={totals[-5:].tolist()}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    vecs.save(out_path)
    print(f"[pdcusi] wrote {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else default_out()
    rc = main(os.path.abspath(out))
    sys.stdout.flush()
    os._exit(rc)
