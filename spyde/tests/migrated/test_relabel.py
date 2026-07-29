"""Component identity must be the same at every position of a scan.

Two gaussians are exchangeable — nothing in the model says which is the broad
one — so fitted position by position they land in whichever slot each fit
picks. Measured on hyperspy's `two_gaussians`, the first component was the
broad peak at only 43% of positions. That makes the caret's numbers jump as
you scrub (one component looks suppressed to zero) and makes a committed
component map a checkerboard of two different peaks.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.actions.fit_action import new_component_spec
from spyde.fitting import ModelSpec
from spyde.fitting.relabel import (
    MIN_SEPARATION, consistency, exchangeable_groups, relabel_scan,
)


def _two_gaussian_spec(n=2):
    spec = ModelSpec()
    for i in range(n):
        c = new_component_spec("Gaussian")
        c.name = f"g{i}"
        spec.append(c)
    return spec


def _scan(p=64, swap_every=3, seed=0):
    """A scan whose two components are cleanly separable but land in random
    slots — the state a batched fit leaves behind."""
    rng = np.random.default_rng(seed)
    wide = np.stack([rng.normal(57000, 2000, p),      # A
                     rng.normal(50, 0.4, p),          # centre
                     rng.normal(25.5, 0.3, p)], 1)    # sigma
    narrow = np.stack([rng.normal(900, 80, p),
                       rng.normal(50, 0.5, p),
                       rng.normal(2.5, 0.15, p)], 1)
    values = np.empty((p, 6))
    swapped = np.zeros(p, bool)
    for i in range(p):
        if i % swap_every == 0:
            values[i] = np.concatenate([narrow[i], wide[i]])
            swapped[i] = True
        else:
            values[i] = np.concatenate([wide[i], narrow[i]])
    return values, swapped


class TestGrouping:
    def test_two_of_a_kind_are_exchangeable(self):
        assert exchangeable_groups(_two_gaussian_spec()) == [[0, 1]]

    def test_a_lone_component_is_not(self):
        assert exchangeable_groups(_two_gaussian_spec(1)) == []

    def test_different_kinds_are_not_exchangeable(self):
        spec = ModelSpec()
        spec.append(new_component_spec("Gaussian"))
        spec.append(new_component_spec("Lorentzian"))
        assert exchangeable_groups(spec) == []

    def test_a_held_parameter_makes_them_distinguishable(self):
        """If one component's centre is FIXED and the other's is free they are
        no longer interchangeable — swapping would move a fitted value into a
        slot that means something else."""
        spec = _two_gaussian_spec()
        spec.components[0]["centre"].free = False
        assert exchangeable_groups(spec) == []


class TestRelabelling:
    def test_a_scrambled_scan_becomes_consistent(self):
        spec = _two_gaussian_spec()
        values, swapped = _scan()
        assert swapped.any()
        assert consistency(spec, values) < 0.9, "the fixture is not scrambled"
        out = relabel_scan(spec, values, (8, 8))
        assert consistency(spec, out) == 1.0

    def test_it_permutes_and_invents_nothing(self):
        """A relabelling moves values between slots. Every position must end
        up holding exactly the numbers it started with."""
        spec = _two_gaussian_spec()
        values, _ = _scan()
        out = relabel_scan(spec, values, (8, 8))
        for before, after in zip(values, out):
            assert sorted(np.round(before, 9).tolist()) == \
                   sorted(np.round(after, 9).tolist())

    def test_whole_components_move_together(self):
        """Not per-parameter sorting: a component's A, centre and sigma must
        travel as one block, or the result is two chimeras."""
        spec = _two_gaussian_spec()
        values, _ = _scan()
        out = relabel_scan(spec, values, (8, 8))
        for row in out:
            first, second = row[:3], row[3:]
            # The narrow one leads (smaller sigma), and it must carry ITS own
            # amplitude — around 900, not the broad component's 57000.
            assert first[2] < second[2]
            assert first[0] < 5000, "a slot got another component's amplitude"
            assert second[0] > 20000

    def test_an_already_consistent_scan_is_left_alone(self):
        """Already in the canonical (ascending) order — nothing to do."""
        spec = _two_gaussian_spec()
        values, _ = _scan(swap_every=1)               # narrow first, every row
        assert consistency(spec, values) == 1.0
        assert np.array_equal(relabel_scan(spec, values, (8, 8)), values)

    def test_inseparable_components_are_not_reordered(self):
        """If nothing tells the components apart, any ordering is an invention
        — and a wrong one is worse than an inconsistent one, because it makes
        a component map look meaningful when it is not."""
        rng = np.random.default_rng(2)
        spec = _two_gaussian_spec()
        # Both components drawn from the SAME distribution: no discriminant.
        v = np.concatenate([rng.normal([1000, 50, 5], [50, 1, 0.3], (40, 3)),
                            rng.normal([1000, 50, 5], [50, 1, 0.3], (40, 3))],
                           axis=1)
        out = relabel_scan(spec, v, (8, 5))
        assert np.array_equal(out, v)

    def test_the_separation_threshold_is_what_decides(self):
        spec = _two_gaussian_spec()
        rng = np.random.default_rng(3)
        p = 60
        a = np.stack([rng.normal(1000, 10, p), rng.normal(50, 0.1, p),
                      rng.normal(5.0, 0.1, p)], 1)
        # Second component's sigma sits a long way off — clearly separable.
        b = a.copy()
        b[:, 2] = rng.normal(5.0 + 20 * MIN_SEPARATION * 0.1, 0.1, p)
        v = np.concatenate([b, a], axis=1)            # deliberately wrong order
        out = relabel_scan(spec, v, (6, 10))
        assert np.all(out[:, 2] < out[:, 5])

    def test_a_single_position_is_untouched(self):
        spec = _two_gaussian_spec()
        v = np.array([[900.0, 50.0, 2.5, 57000.0, 50.0, 25.5]])
        assert np.array_equal(relabel_scan(spec, v, (1, 1)), v)


class TestItDoesNotChangeTheFit:
    """The residual is a property of the model's VALUES, not of which slot
    they sit in. Swapping two interchangeable components leaves the summed
    curve identical — assert it, rather than trusting the argument.
    """

    def test_the_model_curve_is_unchanged(self):
        from spyde.fitting import components as tcomp
        spec = _two_gaussian_spec()
        values, _ = _scan(p=16)
        x = np.linspace(0.0, 100.0, 256)
        before = tcomp.evaluate(spec, torch.as_tensor(x),
                                torch.as_tensor(values)).numpy()
        out = relabel_scan(spec, values, (4, 4))
        after = tcomp.evaluate(spec, torch.as_tensor(x),
                               torch.as_tensor(out)).numpy()
        assert np.allclose(before, after, rtol=1e-12, atol=1e-9)


class TestOnTheRealScan:
    """The measurement that started this, on the dataset it was found on."""

    def test_two_gaussians_becomes_consistent(self):
        hs = pytest.importorskip("hyperspy.api")
        from spyde.actions.fit_action import (
            _seed_for_preview, clamp_to_axis, scale_to_data, spread_repeats,
        )
        from spyde.fitting.engine import fit_batched

        s = hs.data.two_gaussians()
        x = s.axes_manager.signal_axes[0].axis
        data = np.asarray(s.data).astype(float)
        flat = data.reshape(-1, data.shape[-1])
        lo, hi = float(x.min()), float(x.max())

        spec = ModelSpec()
        for _ in range(2):
            c = new_component_spec("Gaussian")
            _seed_for_preview(c, lo, hi)
            scale_to_data(c, x, flat[0])
            spread_repeats(c, spec, lo, hi)
            clamp_to_axis(c, lo, hi)
            while any(e.name == c.name for e in spec.components):
                c.name += "'"
            spec.append(c)

        r = fit_batched(spec, flat, x, device="cpu", max_iter=60)
        before = consistency(spec, r.values)
        assert before < 0.9, f"the scan was already consistent ({before})"
        out = relabel_scan(spec, r.values, data.shape[:2],
                           converged=r.converged)
        assert consistency(spec, out) == 1.0
