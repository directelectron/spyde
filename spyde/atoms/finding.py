"""finding.py — atom positions, refined on the GPU (#75, #77).

The division of labour, and the reason this wave is small:

* **atomap owns the structure.** Initial peak finding, sublattices, nearest
  neighbours, zone axes and dumbbell pairing are its job, and reimplementing
  them would mean diverging from published atomap results.
* **SpyDE owns the refinement.** Refining atom positions IS a batched 2-D
  gaussian fit, which :mod:`spyde.fitting` already does for the whole field at
  once. atomap fits one atom at a time with scipy; the engine fits every atom
  in one batched Levenberg-Marquardt.

The refinement here is deliberately independent of atomap so it can be tested
(and used) without the extra — only :func:`find_atoms` needs it, and that is
``requires_package``-gated at the toolbar.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class MissingExtra(RuntimeError):
    """Raised when the ``atoms`` extra is needed but not installed."""


def _require_atomap():
    try:
        import atomap.api  # noqa: F401
    except ImportError as e:
        raise MissingExtra(
            'atom finding needs atomap — install with: pip install "spyde[atoms]"'
        ) from e


def find_atoms(signal, *, separation: float = 10.0, threshold_rel: float = 0.2,
               pca: bool = False, subtract_background: bool = False):
    """Initial atom positions via atomap -> ``(N, 2)`` array of ``(x, y)``.

    *separation* is the minimum distance between atoms in pixels and is the one
    parameter that matters: too small splits one atom into several, too large
    merges neighbours. It is swept interactively in the UI (#76).
    """
    _require_atomap()
    import atomap.api as am

    data = np.asarray(signal.data if hasattr(signal, "data") else signal)
    positions = am.get_atom_positions(
        _as_signal(data), separation=float(separation),
        threshold_rel=float(threshold_rel), pca=pca,
        subtract_background=subtract_background)
    return np.asarray(positions, float)


def _as_signal(data):
    import hyperspy.api as hs
    return data if hasattr(data, "axes_manager") else hs.signals.Signal2D(data)


def refine_center_of_mass(image, positions, *, box: int = 7,
                          iterations: int = 2) -> np.ndarray:
    """Centre-of-mass refinement in a box around each atom.

    Cheap, robust, and a good starting point for the gaussian fit — but biased
    toward the box centre when neighbours intrude, which is exactly why it is
    a *pre*-step rather than the answer.

    Vectorised over atoms: one gather of all boxes, then a weighted mean. No
    Python loop over atoms.
    """
    img = np.asarray(image, float)
    pos = np.asarray(positions, float).reshape(-1, 2).copy()
    h, w = img.shape
    r = int(box) // 2
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]

    for _ in range(int(iterations)):
        cx = np.clip(np.rint(pos[:, 0]).astype(int), r, w - r - 1)
        cy = np.clip(np.rint(pos[:, 1]).astype(int), r, h - r - 1)
        ys = cy[:, None, None] + oy[None]
        xs = cx[:, None, None] + ox[None]
        patch = img[ys, xs]
        patch = np.clip(patch - patch.min(axis=(1, 2), keepdims=True), 0, None)
        total = patch.sum(axis=(1, 2))
        good = total > 0
        # An empty box would divide by zero; leave those atoms where they are.
        pos[good, 0] = (cx[good] +
                        (patch * ox[None]).sum(axis=(1, 2))[good] / total[good])
        pos[good, 1] = (cy[good] +
                        (patch * oy[None]).sum(axis=(1, 2))[good] / total[good])
    return pos


def refine_gaussian(image, positions, *, box: int = 11, sigma: float = 2.5,
                    device=None, max_iter: int = 60, refine_mask=None):
    """Batched 2-D gaussian refinement of every atom at once.

    Each atom becomes one row of a ``(N, box*box)`` stack and one fit in the
    batched engine — the same code path a spectrum image uses, with
    :func:`~spyde.fitting.components.image_coordinates` as the axis.

    Parameters
    ----------
    refine_mask : array of bool, optional
        False for atoms whose position is pinned (the red/green toggle in
        #76). A pinned atom keeps its input position exactly and is excluded
        from the fit — not fitted and then discarded, which would waste the
        work and let a diverging neighbour perturb the shared solve.

    Returns
    -------
    (positions, params)
        Refined ``(N, 2)`` ``(x, y)``, and the full per-atom parameter table
        (``A``, ``centre_x``, ``centre_y``, ``sigma_x``, ``sigma_y``) in image
        coordinates, which is what the property maps (#79) are built from.
    """
    from spyde.fitting import ModelSpec
    from spyde.fitting.components import image_coordinates
    from spyde.fitting.engine import fit_batched
    from spyde.fitting.spec import ComponentSpec, ParameterSpec

    img = np.asarray(image, float)
    pos = np.asarray(positions, float).reshape(-1, 2)
    n = len(pos)
    mask = (np.ones(n, bool) if refine_mask is None
            else np.asarray(refine_mask, bool).ravel())
    if mask.size != n:
        raise ValueError(f"refine_mask has {mask.size} entries for {n} atoms")

    h, w = img.shape
    r = int(box) // 2
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    # Clamp box centres so every patch lies inside the image; the offset is
    # tracked so results come back in IMAGE coordinates, not patch ones.
    cx = np.clip(np.rint(pos[:, 0]).astype(int), r, w - r - 1)
    cy = np.clip(np.rint(pos[:, 1]).astype(int), r, h - r - 1)

    out_pos = pos.copy()
    out_par = np.full((n, 5), np.nan)
    if not mask.any():
        return out_pos, out_par

    sel = np.flatnonzero(mask)
    patches = img[cy[sel][:, None, None] + oy[None],
                  cx[sel][:, None, None] + ox[None]].reshape(len(sel), -1)

    # Start each atom at the CENTRE of its own patch, with the amplitude scaled
    # to the patch — a gaussian's A is its VOLUME, so seeding it with the peak
    # height would start every fit an order of magnitude low.
    peak = patches.max(1)
    spec = ModelSpec(components=[ComponentSpec(
        kind="Gaussian2D", parameters=[
            ParameterSpec("A", 1.0, linear=True),
            ParameterSpec("centre_x", float(r)),
            ParameterSpec("centre_y", float(r)),
            ParameterSpec("sigma_x", float(sigma), bmin=0.3, bmax=float(box)),
            ParameterSpec("sigma_y", float(sigma), bmin=0.3, bmax=float(box)),
        ])])
    names = spec.parameter_names()
    start = np.broadcast_to(spec.flat_values(), (len(sel), 5)).copy()
    start[:, names.index("Gaussian2D.A")] = peak * 2 * np.pi * sigma * sigma
    start[:, names.index("Gaussian2D.centre_x")] = pos[sel, 0] - cx[sel] + r
    start[:, names.index("Gaussian2D.centre_y")] = pos[sel, 1] - cy[sel] + r

    xy = image_coordinates((box, box)).numpy().astype(float)
    res = fit_batched(spec, patches, xy, device=device, max_iter=max_iter,
                      initial=start)

    px = res.values[:, names.index("Gaussian2D.centre_x")] + cx[sel] - r
    py = res.values[:, names.index("Gaussian2D.centre_y")] + cy[sel] - r

    # A fit that ran away from its own box is worse than the input. Keep the
    # starting position for those rather than returning a wild coordinate.
    ok = (np.abs(px - pos[sel, 0]) <= r) & (np.abs(py - pos[sel, 1]) <= r) \
        & np.isfinite(px) & np.isfinite(py)
    out_pos[sel[ok], 0] = px[ok]
    out_pos[sel[ok], 1] = py[ok]
    out_par[sel] = res.values
    out_par[sel, names.index("Gaussian2D.centre_x")] += cx[sel] - r
    out_par[sel, names.index("Gaussian2D.centre_y")] += cy[sel] - r
    if not ok.all():
        log.info("%d/%d atom fits left their box and kept their input "
                 "position", int((~ok).sum()), len(sel))
    return out_pos, out_par


def refine_atoms(image, positions, *, com_box: int = 7, box: int = 11,
                 sigma: float = 2.5, device=None, refine_mask=None):
    """Centre-of-mass then batched gaussian — the standard two-step.

    The COM pass is cheap and pulls a rough peak-find onto the atom; the
    gaussian pass is what gives sub-pixel accuracy. Running the gaussian alone
    from a poor start is markedly less reliable, which is why atomap does the
    same two steps.
    """
    coarse = refine_center_of_mass(image, positions, box=com_box)
    if refine_mask is not None:
        keep = ~np.asarray(refine_mask, bool).ravel()
        coarse[keep] = np.asarray(positions, float).reshape(-1, 2)[keep]
    return refine_gaussian(image, coarse, box=box, sigma=sigma, device=device,
                           refine_mask=refine_mask)
