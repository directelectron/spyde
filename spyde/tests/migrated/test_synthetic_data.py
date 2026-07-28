"""Contract tests for the bundled synthetic datasets (spyde/data/synthetic.py).

These pin the two properties Waves 1-4 actually rely on:

1. **The physics is where it claims to be** — an edge steps up at its onset, a
   line peaks at its energy, a Kikuchi pattern changes when the orientation
   does. Data that merely *has the right shape* would let a broken fit pass.
2. **Nothing is symmetric.** Every spatial map differs from its transpose and
   from both mirrors, so a flipped axis or a transposed nav grid fails a test
   instead of silently looking fine.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.data import ebsd_patterns, eds_si, eels_si, ground_truth
from spyde.data.synthetic import EDS_LINES, EELS_EDGES


def _asymmetric(arr: np.ndarray) -> bool:
    """True when `arr` is distinguishable from its transpose and both mirrors."""
    a = np.asarray(arr, float)
    return (not np.allclose(a, a.T)
            and not np.allclose(a, a[:, ::-1])
            and not np.allclose(a, a[::-1]))


class TestEELS:
    def test_shape_and_calibration(self):
        s = eels_si(nav=(6, 5), n_channels=256)
        assert s.data.shape == (6, 5, 256)
        ax = s.axes_manager.signal_axes[0]
        assert ax.units == "eV"
        assert ax.offset == pytest.approx(200.0)
        # The axis must actually span the edges, or the fit has nothing to fit.
        e = ax.axis
        for onset in EELS_EDGES.values():
            assert e[0] < onset < e[-1]

    def test_edges_step_up_at_their_onsets(self):
        """A core-loss edge is a STEP: intensity just above the onset exceeds
        intensity just below it.

        Measured at the pixel where that element is most abundant. Testing a
        fixed pixel would be measuring the wrong thing — at a pixel where the
        element is nearly absent the power-law background's own decay across
        the window is larger than the edge, which is exactly why background
        modelling (#64) is a real task and not a formality.
        """
        s = eels_si(nav=(8, 8), n_channels=2048, noise=False)
        e = s.axes_manager.signal_axes[0].axis
        conc = ground_truth(s)["concentration"]
        for name, onset in EELS_EDGES.items():
            iy, ix = np.unravel_index(int(np.argmax(np.asarray(conc[name]))),
                                      s.data.shape[:2])
            spec = s.data[iy, ix].astype(float)
            i = int(np.argmin(np.abs(e - onset)))
            pre, post = spec[i - 40:i - 8], spec[i + 8:i + 40]
            assert post.mean() > pre.mean(), (
                f"no step at {onset} eV ({name}) in the {name}-rich pixel")

    def test_ground_truth_is_plain_python(self):
        s = eels_si(nav=(4, 4), n_channels=128)
        gt = ground_truth(s)
        assert gt["kind"] == "eels"
        assert set(gt["concentration"]) == set(EELS_EDGES)
        # DictionaryTreeBrowser has no .values(); ground_truth() must undo that.
        assert isinstance(gt["concentration"], dict)
        assert list(gt["concentration"].values())

    def test_concentration_maps_are_asymmetric_and_distinct(self):
        s = eels_si(nav=(12, 12), n_channels=128)
        maps = ground_truth(s)["concentration"]
        for name, m in maps.items():
            assert _asymmetric(np.asarray(m)), f"{name} map is symmetric"
        vals = [np.asarray(m) for m in maps.values()]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                assert not np.allclose(vals[i], vals[j])

    def test_edge_intensity_tracks_its_concentration_map(self):
        """The whole point of the ground truth: more of element X really does
        mean a bigger X edge. A quantification result can be scored on this."""
        s = eels_si(nav=(10, 10), n_channels=1024, noise=False)
        e = s.axes_manager.signal_axes[0].axis
        gt = ground_truth(s)
        onset = EELS_EDGES["O_K"]
        i = int(np.argmin(np.abs(e - onset)))
        step = s.data[..., i + 3:i + 30].mean(-1) - s.data[..., i - 30:i - 3].mean(-1)
        conc = np.asarray(gt["concentration"]["O_K"])
        # Rank correlation is enough — background varies independently.
        assert np.corrcoef(step.ravel(), conc.ravel())[0, 1] > 0.8


class TestEDS:
    def test_shape_and_calibration(self):
        s = eds_si(nav=(5, 6), n_channels=512)
        assert s.data.shape == (5, 6, 512)
        assert s.axes_manager.signal_axes[0].units == "keV"

    def test_lines_peak_at_their_real_energies(self):
        s = eds_si(nav=(4, 4), n_channels=4096, noise=False)
        e = s.axes_manager.signal_axes[0].axis
        spec = s.data.sum((0, 1)).astype(float)
        for el, lines in EDS_LINES.items():
            energy = lines[0][1]                     # the Ka line
            i = int(np.argmin(np.abs(e - energy)))
            win = spec[i - 30:i + 30]
            # The local maximum must sit at the line, not merely nearby.
            assert abs(int(np.argmax(win)) - 30) <= 3, f"{el} Ka misplaced"

    def test_ka_kb_ratio_recoverable_for_an_isolated_family(self):
        """Ka:Kb ratio is what makes family-aware fitting testable — checked on
        Fe, whose Kb (7.058) is the one line with no near neighbour, and at the
        Fe-rich pixel so the other families contribute almost nothing."""
        from scipy.ndimage import percentile_filter

        s = eds_si(nav=(8, 8), n_channels=4096, noise=False)
        e = s.axes_manager.signal_axes[0].axis
        conc = np.asarray(ground_truth(s)["concentration"]["Fe"])
        iy, ix = np.unravel_index(int(np.argmax(conc)), conc.shape)
        spec = s.data[iy, ix].astype(float)

        # Rolling low percentile over a window several FWHM wide follows the
        # bremsstrahlung and ignores the narrow lines. Hand-picked flat windows
        # do NOT work here: at 4096 channels every "clean" window near Fe-Kb is
        # within a few hundred eV of another line.
        width = int(round(1.0 / float(e[1] - e[0])))            # ~1 keV
        baseline = percentile_filter(spec, 10, size=width)
        net = spec - baseline

        def height(energy):
            i = int(np.argmin(np.abs(e - energy)))
            return float(net[i - 20:i + 20].max())

        (_, e_a, r_a), (_, e_b, r_b) = EDS_LINES["Fe"][0], EDS_LINES["Fe"][1]
        assert height(e_b) / height(e_a) == pytest.approx(r_b / r_a, rel=0.25)

    def test_families_overlap_as_intended(self):
        """The overlaps are a FEATURE — they are why a peak-per-element finder
        is not good enough and family-aware fitting (#62) is needed. Pin them
        so a future tweak to the line table cannot quietly remove the
        difficulty this dataset exists to pose."""
        from spyde.data.synthetic import _detector_sigma
        flat = [(el, en) for el, ls in EDS_LINES.items() for _n, en, _r in ls]
        close = []
        for i, (el_a, e_a) in enumerate(flat):
            for el_b, e_b in flat[i + 1:]:
                if el_a == el_b:
                    continue
                fwhm = 2.3548 * float(_detector_sigma((e_a + e_b) / 2))
                if abs(e_a - e_b) < 2.5 * fwhm:
                    close.append((el_a, el_b, abs(e_a - e_b)))
        assert close, "no inter-element overlaps left in the EDS line table"

    def test_ground_truth_lines_readable(self):
        gt = ground_truth(eds_si(nav=(3, 3), n_channels=128))
        assert gt["kind"] == "eds"
        assert set(gt["lines"]) == set(EDS_LINES)
        assert all(_asymmetric(np.asarray(m))
                   for m in gt["concentration"].values())


class TestEBSD:
    def test_shape_and_dtype(self):
        s = ebsd_patterns(nav=(5, 4), detector=(32, 32))
        assert s.data.shape == (5, 4, 32, 32)
        assert s.data.dtype == np.uint8

    def test_patterns_are_not_mirror_symmetric(self):
        """A Kikuchi pattern that survives a flip cannot catch a flipped
        detector axis — the bug this data exists to make visible."""
        s = ebsd_patterns(nav=(3, 3), detector=(48, 48), noise=0.0)
        p = s.data[0, 0].astype(float)
        assert not np.allclose(p, p[:, ::-1])
        assert not np.allclose(p, p[::-1])

    def test_orientation_field_has_two_distinct_grains(self):
        s = ebsd_patterns(nav=(16, 16), detector=(40, 40), noise=0.0)
        gt = ground_truth(s)
        eul, mask = np.asarray(gt["euler"]), np.asarray(gt["grain2_mask"], bool)
        assert mask.any() and not mask.all(), "wedge grain missing"
        # Grain 2 is a single orientation; grain 1 drifts along x.
        assert np.allclose(eul[mask], eul[mask][0])
        assert not np.allclose(eul[~mask], eul[~mask][0])

    def test_same_orientation_gives_the_same_pattern(self):
        """Ground truth is only usable if the mapping orientation->pattern is
        deterministic: two pixels of grain 2 must be pixel-identical."""
        s = ebsd_patterns(nav=(16, 16), detector=(40, 40), noise=0.0)
        mask = np.asarray(ground_truth(s)["grain2_mask"], bool)
        ys, xs = np.nonzero(mask)
        a = s.data[ys[0], xs[0]]
        b = s.data[ys[-1], xs[-1]]
        assert np.array_equal(a, b)

    def test_different_orientation_gives_a_different_pattern(self):
        s = ebsd_patterns(nav=(16, 16), detector=(40, 40), noise=0.0)
        mask = np.asarray(ground_truth(s)["grain2_mask"], bool)
        ys, xs = np.nonzero(mask)
        ys2, xs2 = np.nonzero(~mask)
        g2 = s.data[ys[0], xs[0]].astype(float)
        g1 = s.data[ys2[0], xs2[0]].astype(float)
        # Normalised cross-correlation well below 1 — i.e. actually indexable.
        ncc = float(np.corrcoef(g1.ravel(), g2.ravel())[0, 1])
        assert ncc < 0.9, f"grains too similar to index (ncc={ncc:.3f})"

    def test_band_count_matches_cubic_families(self):
        """4x{111} + 3x{200} + 6x{220} = 13 Friedel-unique bands."""
        assert ground_truth(ebsd_patterns(nav=(2, 2),
                                         detector=(16, 16)))["n_bands"] == 13


class TestNoOptionalExtrasRequired:
    def test_generators_work_without_exspy_or_kikuchipy(self):
        """The extras are optional, so generation must never depend on them —
        the signal just stays a generic Signal1D/Signal2D."""
        import hyperspy.api as hs
        assert isinstance(eels_si(nav=(2, 2), n_channels=64), hs.signals.BaseSignal)
        assert isinstance(eds_si(nav=(2, 2), n_channels=64), hs.signals.BaseSignal)
        assert isinstance(ebsd_patterns(nav=(2, 2), detector=(8, 8)),
                          hs.signals.BaseSignal)


class TestSessionLoaders:
    """The three harness loaders reach the UI path (a tree + plots appear)."""

    @pytest.mark.parametrize("action", ["load_test_data_eels",
                                        "load_test_data_eds",
                                        "load_test_data_ebsd"])
    def test_loader_creates_a_tree(self, window, action):
        session = window["window"]
        session.dispatch_action({"action": action,
                                 "payload": {"nav": (4, 4)}})
        assert len(window["signal_trees"]) == 1
        assert window["plots"]
