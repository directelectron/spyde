"""
test_dpc.py — the DPC physics, pinned against ground truth and against pyxem.

The suite is built around one fact: a DPC map is *only* useful if its
directions are right, and a direction error is invisible — a rotated field map
looks exactly as plausible as a correct one. So the tests here are almost all
about x/y, sign, and phase, and almost none about "did it return an array".

What is pinned, and why each one can silently break:

* **The colour wheel reads TRUE.** pyxem's phase is ``arctan2(x, y)`` with a
  −30° hue offset, so the legend has to be rotated +90° relative to the map. Get
  it wrong and every published figure's arrows point somewhere else. Two
  independent checks: the wheel agrees with the map's own colours, and it agrees
  with what ``plot_beam_shift_color`` draws.
* **The corner boxes are the pixels that get fitted.** The overlay and the mask
  come from one function; on a non-square scan pyxem's own helper transposes
  them.
* **The rotation solver recovers a known angle and handedness**, on both the
  curl-free (electric) and divergence-free (magnetic) constraint.
* **The sign convention.** Magnetic is parallel to the beam shift, electric is
  ANTI-parallel (``calibrate_electric_shifts`` divides by ``-e``). Neither is
  arbitrary and neither is guessable from the code.
* **The electric field equals pyxem's own chain**, exactly.
"""
from __future__ import annotations

import colorsys
import math
import re
from pathlib import Path

import numpy as np
import pytest

from spyde.actions import dpc


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hue(rgb) -> float:
    """Hue in degrees of an RGB triple (uint8 or float 0-1)."""
    a = np.asarray(rgb, dtype=float)
    if a.max() > 1.0:
        a = a / 255.0
    return colorsys.rgb_to_hsv(*a[:3])[0] * 360.0


def _hue_gap(a: float, b: float) -> float:
    return abs(((a - b) + 180.0) % 360.0 - 180.0)


def _curl_free_field(n: int = 64) -> np.ndarray:
    """``-∇V`` of a Gaussian bump: curl-free by construction."""
    yy, xx = np.mgrid[0:n, 0:n] / (n / 2.0) - 1.0
    V = np.exp(-(xx ** 2 + yy ** 2) * 3.0)
    fy, fx = np.gradient(-V)
    return np.stack([fx, fy], axis=-1)


def _div_free_field(n: int = 64) -> np.ndarray:
    """The 2-D curl of a scalar: divergence-free by construction."""
    yy, xx = np.mgrid[0:n, 0:n] / (n / 2.0) - 1.0
    A = np.exp(-(xx ** 2 + yy ** 2) * 3.0)
    ay, ax = np.gradient(A)
    return np.stack([ay, -ax], axis=-1)


def _uniform(sx: float, sy: float, n: int = 4) -> np.ndarray:
    out = np.empty((n, n, 2), dtype=np.float64)
    out[..., 0], out[..., 1] = sx, sy
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The phase convention — the reason this module exists
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseConvention:
    """The colour wheel must describe the map it sits on.

    Screen frame throughout: the map is drawn ``origin="upper"``, so ``+x`` is
    RIGHT and ``+y`` is DOWN. A screen angle of 0° points right and 90° points
    UP.
    """

    @staticmethod
    def _map_colour(screen_deg: float, rotation):
        """The map's colour for a uniform shift pointing at *screen_deg*."""
        sx = math.cos(math.radians(screen_deg))
        sy = -math.sin(math.radians(screen_deg))     # screen up = -y
        rgb = dpc.magnitude_phase_rgb(_uniform(sx, sy), rotation=rotation,
                                      autolim=False, magnitude_limits=(0.0, 1.0))
        return rgb[2, 2]

    @staticmethod
    def _wheel_colour(wheel: np.ndarray, screen_deg: float, frac: float = 0.6):
        """The wheel's colour at *screen_deg*, *frac* of the way to the rim."""
        n = wheel.shape[0]
        c = (n - 1) / 2.0
        i = int(round(c - frac * c * math.sin(math.radians(screen_deg))))
        j = int(round(c + frac * c * math.cos(math.radians(screen_deg))))
        return wheel[i, j, :3]

    @pytest.mark.parametrize("rotation", [None, 0.0, 45.0, -110.0, 210.0])
    def test_wheel_matches_the_map_it_describes(self, rotation):
        """Look a colour up on the wheel and it points the way the field does.

        This is the whole contract. It holds because `color_wheel_rgba` runs the
        map's own colour function over a field of known screen directions —
        which is why it survives a rotation the map applies too.
        """
        wheel = dpc.color_wheel_rgba(201, rotation=rotation)
        for deg in range(0, 360, 15):
            gap = _hue_gap(_hue(self._map_colour(deg, rotation)),
                           _hue(self._wheel_colour(wheel, deg)))
            assert gap < 5.0, (
                f"at rotation={rotation}, screen direction {deg}°: map hue and "
                f"wheel hue differ by {gap:.1f}° — the legend is lying")

    def test_wheel_matches_pyxem_published_indicator(self):
        """Agree with ``plot_beam_shift_color``'s indicator, not the marker
        ``get_magnitude_phase_signal`` embeds.

        pyxem is internally inconsistent here: the standalone plotting function
        draws its wheel at ``phase_rotation + 60`` (= map_rotation + 90, correct),
        while the marker attached to the signal reuses ``map_rotation`` and is
        therefore 90° out. Following the marker would put SpyDE's legend 90° off
        from every figure in the pyxem docs.
        """
        import pyxem.utils._beam_shift_tools as bst

        def pyxem_indicator_hue(screen_deg, phase_rotation=0.0):
            n = 501
            x, y = np.mgrid[-2.0:2.0:complex(n), -2.0:2.0:complex(n)]
            t = np.arctan2(x, y) + math.radians(phase_rotation + 60.0)
            t = (t + np.pi) % (2 * np.pi) - np.pi
            rgb = bst._get_rgb_phase_magnitude_array(t, np.ones_like(t),
                                                     only_phase=True)
            # drawn origin="lower": axis 0 (x) is UP, axis 1 (y) is RIGHT
            c = (n - 1) / 2.0
            i = int(round(c + 0.6 * c * math.sin(math.radians(screen_deg))))
            j = int(round(c + 0.6 * c * math.cos(math.radians(screen_deg))))
            return _hue(rgb[i, j])

        ours = dpc.color_wheel_rgba(501, rotation=0.0, only_phase=True)
        for deg in range(0, 360, 15):
            gap = _hue_gap(_hue(self._wheel_colour(ours, deg)),
                           pyxem_indicator_hue(deg))
            assert gap < 5.0, (
                f"screen direction {deg}°: {gap:.1f}° from pyxem's own "
                f"plot_beam_shift_color indicator")

    def test_the_wheel_labels_sit_where_they_read(self):
        """The compass points must name the direction actually AT that spot.

        The wheel is drawn in the map's screen frame (+x right, +y DOWN), so the
        label at the bottom is +y — not "down", which would silently describe
        −y to anyone who assumes the usual maths convention.
        """
        from spyde.actions import dpc_display
        wheel = dpc.color_wheel_rgba(201)
        n = wheel.shape[0]
        c = (n - 1) / 2.0
        for spec in dpc_display.WHEEL_LABELS:
            # The key's label coordinates are image fractions: x right, y down.
            fx, fy, text = spec["x"], spec["y"], spec["text"]
            ux, uy = fx - 0.5, fy - 0.5              # screen offset from centre
            sx = np.sign(ux) if abs(ux) > 0.2 else 0.0
            sy = np.sign(uy) if abs(uy) > 0.2 else 0.0
            expect = {(1.0, 0.0): "+x", (-1.0, 0.0): "−x",
                      (0.0, 1.0): "+y", (0.0, -1.0): "−y"}[(sx, sy)]
            assert text == expect, \
                f"the label at ({fx}, {fy}) says {text!r} but points {expect}"
            # …and the colour there really is the colour of a shift that way.
            i = int(round(c + 0.6 * c * sy))
            j = int(round(c + 0.6 * c * sx))
            m = dpc.magnitude_phase_rgb(_uniform(float(sx), float(sy)),
                                        autolim=False, magnitude_limits=(0., 1.))
            assert _hue_gap(_hue(wheel[i, j, :3]), _hue(m[2, 2])) < 5.0

    def test_wheel_is_transparent_outside_the_disc(self):
        w = dpc.color_wheel_rgba(64)
        assert w.shape == (64, 64, 4)
        assert w[0, 0, 3] == 0 and w[-1, -1, 3] == 0, "corners must be clear"
        assert w[32, 32, 3] == 255, "the centre must be opaque"

    def test_wheel_centre_is_dark_and_rim_is_saturated(self):
        """The legend keys MAGNITUDE as well as direction: black at zero field."""
        w = dpc.color_wheel_rgba(129)
        assert w[64, 64, :3].max() < 30, "the centre should read as zero field"
        rim = w[64, int(64 + 0.9 * 64), :3]
        assert rim.max() > 200, "the rim should read as the display maximum"

    def test_the_legend_does_not_turn_with_the_scan_rotation(self):
        """The scan↔detector rotation is applied to the VECTORS, so the colour
        wheel must be identical at every rotation.

        This is the bug that shipped and was caught by looking at the app: the
        display used to rebuild the wheel at ``result.rotation``, which applies
        the angle a SECOND time — the map is already in the scan frame by then.
        The result is a legend pointing ``rotation`` degrees away from the map
        it describes: a wrong-direction figure that looks completely plausible,
        which is the whole failure mode this module exists to prevent.
        """
        import hyperspy.api as hs
        field = _curl_free_field(24)
        gy, gx = np.mgrid[0:16, 0:16]
        s = hs.signals.Signal2D(
            np.zeros((24, 24, 16, 16), np.float32))
        s.set_signal_type("electron_diffraction")

        wheels, maps = [], []
        for rot in (0.0, 25.0, 137.0):
            r = dpc.compute_dpc(s, shifts=field, reference=None,
                                center_mode="none", mode="magnetic",
                                rotation=rot)
            wheels.append(r.wheel)
            maps.append(r.rgb)
        for w in wheels[1:]:
            assert np.array_equal(w, wheels[0]), \
                "the colour wheel turned with the scan rotation — it must not"
        assert not np.array_equal(maps[0], maps[1]), \
            "the MAP must still turn (otherwise rotation does nothing at all)"

    def test_display_rotation_is_shared_by_map_and_legend(self):
        """One constant feeds both, so they cannot be given different values."""
        assert dpc.DISPLAY_ROTATION is None
        field = _uniform(1.0, 0.0, n=8)
        direct = dpc.magnitude_phase_rgb(field, rotation=dpc.DISPLAY_ROTATION,
                                         autolim=False,
                                         magnitude_limits=(0.0, 1.0))
        wheel = dpc.color_wheel_rgba(129, rotation=dpc.DISPLAY_ROTATION)
        # +x is screen-RIGHT: the wheel's right edge must carry the map's hue.
        assert _hue_gap(_hue(direct[4, 4]),
                        _hue(wheel[64, int(64 + 0.6 * 64), :3])) < 5.0

    def test_phase_map_matches_the_rgb_hue(self):
        """The committed ``phase`` map is the same angle the RGB map paints."""
        for deg in (0, 40, 155, 300):
            sx = math.cos(math.radians(deg))
            sy = -math.sin(math.radians(deg))
            shifts = _uniform(sx, sy)
            expect = np.degrees(dpc.phase(shifts)[2, 2])
            got = _hue(self._map_colour(deg, None))
            assert _hue_gap(expect, got) < 5.0


# ─────────────────────────────────────────────────────────────────────────────
# The corner reference
# ─────────────────────────────────────────────────────────────────────────────

class TestCornerGeometry:
    @pytest.mark.parametrize("nav", [(64, 64), (40, 120), (120, 40), (7, 5)])
    def test_drawn_boxes_are_the_fitted_pixels(self, nav):
        """The overlay and the fit mask come from one source.

        A user who moves the corner-size slider is told, by the boxes, which
        pixels are being used. If the mask disagreed, the picture would be a
        confident lie — and on a NON-SQUARE scan pyxem's own ``_get_corner_slices``
        derives the two block sizes from one axis pair and applies them to the
        other, which is exactly that bug.
        """
        mask = dpc.corner_mask(nav, 0.1)
        drawn = np.ones(nav, dtype=bool)
        for (x, y, w, h) in dpc.corner_boxes(nav, 0.1):
            drawn[int(y):int(y + h), int(x):int(x + w)] = False
        assert np.array_equal(drawn, mask)

    def test_boxes_scale_with_their_own_axis(self):
        (x0, y0, w, h), *_ = dpc.corner_boxes((40, 120), 0.1)
        assert (w, h) == (12.0, 4.0), "each box spans 10% of ITS OWN axis"

    def test_four_boxes_at_the_four_corners(self):
        boxes = dpc.corner_boxes((50, 50), 0.1)
        assert len(boxes) == 4
        corners = {(x == 0, y == 0) for (x, y, _w, _h) in boxes}
        assert corners == {(True, True), (False, True), (True, False), (False, False)}

    @pytest.mark.parametrize("frac", [0.001, 0.5, 1.0, -1.0])
    def test_extreme_fractions_still_select_something(self, frac):
        mask = dpc.corner_mask((16, 16), frac)
        assert (~mask).sum() >= 4, "every corner must contribute at least a pixel"

    def test_a_plane_ramp_is_removed_exactly(self):
        shifts = np.zeros((30, 40, 2))
        rows, cols = np.mgrid[0:30, 0:40]
        shifts[..., 0] = 0.3 * cols - 0.1 * rows + 2.0
        shifts[..., 1] = -0.2 * cols + 0.4 * rows - 1.0
        residual = dpc.apply_reference(shifts, dpc.corner_reference(shifts, 0.1))
        assert np.abs(residual).max() < 1e-9

    def test_the_corners_drive_the_fit_not_the_middle(self):
        """A feature in the MIDDLE must not pull the reference plane.

        This is the assumption the mode is built on, stated as a test: whatever
        is happening in the centre of the scan is signal, and the plane must
        ignore it.
        """
        shifts = np.zeros((40, 40, 2))
        shifts[15:25, 15:25, 0] = 50.0          # a huge central "field"
        ref = dpc.corner_reference(shifts, 0.1)
        assert np.abs(ref).max() < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Centering report + the other two references
# ─────────────────────────────────────────────────────────────────────────────

class TestBeamRegion:
    """The shape the user drags onto the beam — one object, two jobs.

    It picks WHICH PIXELS the centroid integrates and WHERE the undeflected
    beam is. Both are load-bearing and both fail silently: a region on empty
    detector still returns a centroid, and a centroid dragged by a diffracted
    disc still returns a plausible field.
    """

    @staticmethod
    def _scan(hollow_core=False):
        """A beam off-centre, plus a bright blob that would drag an unmasked
        centroid — the situation the region exists for."""
        import pyxem as pxm
        k = 64
        gy, gx = np.mgrid[0:k, 0:k]
        beam = (((gy - 36.0) ** 2 + (gx - 40.0) ** 2) < 36).astype(np.float32) * 100
        if hollow_core:                      # a beam stop / saturated centre
            beam[((gy - 36.0) ** 2 + (gx - 40.0) ** 2) < 9] = 0.0
        junk = (((gy - 8.0) ** 2 + (gx - 8.0) ** 2) < 25).astype(np.float32) * 60
        return pxm.signals.ElectronDiffraction2D(np.tile(beam + junk, (2, 3, 1, 1))), k

    def test_circle_is_identical_to_pyxems_own_mask(self):
        """One code path serves circle AND ring, because pyxem's ``mask=(x,y,r)``
        covers only the disc. That is only safe if the disc case is provably the
        same computation — so compare it, rather than assume it."""
        s, _k = self._scan()
        region = dpc.BeamRegion("circle", 40.0, 36.0, 12.0)
        ours = dpc.measure_beam_shifts(s, region=region)
        theirs = np.asarray(s.get_direct_beam_position(
            method="center_of_mass", lazy_output=False,
            mask=(40.0, 36.0, 12.0)).data)
        assert np.array_equal(ours, theirs)
        assert np.allclose(ours[0, 0], [-8.0, -4.0], atol=0.01)

    def test_without_a_region_the_centroid_is_dragged_off(self):
        """The motivation, as a test: this is what the circle is protecting."""
        s, _k = self._scan()
        unmasked = dpc.measure_beam_shifts(s)
        assert not np.allclose(unmasked[0, 0], [-8.0, -4.0], atol=1.0)

    def test_ring_finds_a_beam_with_a_blocked_core(self):
        """A saturated or beam-stopped direct beam carries no usable centre, so
        the centroid is taken over the disc EDGE — which is symmetric about the
        true position, so it still lands on it."""
        s, _k = self._scan(hollow_core=True)
        got = dpc.measure_beam_shifts(
            s, region=dpc.BeamRegion("ring", 40.0, 36.0, 12.0, 3.0))
        assert np.allclose(got[0, 0], [-8.0, -4.0], atol=0.05)

    def test_masks_are_the_shapes_they_claim(self):
        circle = dpc.BeamRegion("circle", 32, 32, 10).mask((64, 64))
        ring = dpc.BeamRegion("ring", 32, 32, 10, 5).mask((64, 64))
        assert circle[32, 32] and not ring[32, 32]
        assert ring[32, 39] and circle.sum() > ring.sum()
        assert dpc.BeamRegion().mask((64, 64)) is None

    def test_contains_agrees_with_the_mask(self):
        """`contains` is what `beam_inside_region` judges by, so it must be the
        same shape the centroid was actually taken over."""
        region = dpc.BeamRegion("ring", 32, 32, 10, 5)
        mask = region.mask((64, 64))
        for x in range(18, 47):
            for y in range(18, 47):
                assert region.contains(x, y) == bool(mask[y, x])

    def test_an_empty_region_gives_NaN_not_a_centred_reading(self):
        s, _k = self._scan()
        empty = dpc.measure_beam_shifts(s, region=dpc.BeamRegion("circle", 2, 2, 1.0))
        assert np.isnan(empty).all(), "a silent 0 would read as a centred beam"

    def test_a_slipped_region_still_returns_plausible_numbers(self):
        """THE failure mode, stated as a test: a region on empty detector
        returns the centroid of whatever is inside it — finite, plausible, and
        wrong. Nothing in the result says so, which is why the caret has to."""
        s, _k = self._scan()
        slipped = dpc.measure_beam_shifts(
            s, region=dpc.BeamRegion("circle", 12.0, 12.0, 8.0))
        assert np.all(np.isfinite(slipped))
        assert not np.allclose(slipped[0, 0], [-8.0, -4.0], atol=1.0)

    def test_containment_cannot_catch_a_slipped_CIRCLE(self):
        """A convex region always contains its own centroid, so this check is
        vacuous for a disc — worth pinning so nobody later relies on it.

        It is the reason the caret warns on BRIGHTNESS instead.
        """
        s, k = self._scan()
        off = dpc.BeamRegion("circle", 12.0, 12.0, 8.0)
        slipped = dpc.measure_beam_shifts(s, region=off)
        assert dpc.beam_inside_region(slipped, off, (k, k)), \
            "a disc trivially contains its own centroid — not a useful test"

    def test_containment_DOES_catch_an_off_centre_ring(self):
        """A ring is not convex, so its centroid can land in the hole — which is
        exactly what an off-centre ring does, and is worth reporting."""
        s, k = self._scan()
        ring = dpc.BeamRegion("ring", 40.0, 36.0, 26.0, 14.0)
        shifts = dpc.measure_beam_shifts(s, region=ring)
        assert not dpc.beam_inside_region(shifts, ring, (k, k))

    def test_brightness_is_scale_free(self):
        """Density over the frame average, NOT captured fraction: the fraction
        depends on how much else the pattern contains, so no threshold works
        across datasets."""
        s, _k = self._scan()
        on = dpc.region_brightness(s, dpc.BeamRegion("circle", 40, 36, 12))
        off = dpc.region_brightness(s, dpc.BeamRegion("circle", 5, 58, 6))
        assert on > 2.0 > off
        assert dpc.region_brightness(s, dpc.BeamRegion()) == float("inf")

    def test_the_region_centre_serves_as_the_manual_reference(self):
        """A shape already dragged onto the beam has answered "where is the
        beam" — Manual must not demand a second marker saying the same thing."""
        s, _k = self._scan()
        r = dpc.compute_dpc(s, mode="magnetic", center_mode="manual",
                            region=dpc.BeamRegion("circle", 40.0, 36.0, 12.0),
                            rotation=0.0)
        residual = r.raw_shifts - r.reference
        assert abs(residual[..., 0].mean()) < 0.02
        assert abs(residual[..., 1].mean()) < 0.02

    def test_an_explicit_center_still_wins_over_the_region(self):
        s, _k = self._scan()
        r = dpc.compute_dpc(s, mode="magnetic", center_mode="manual",
                            center_xy=(30.0, 30.0),
                            region=dpc.BeamRegion("circle", 40.0, 36.0, 12.0),
                            rotation=0.0)
        assert r.reference[..., 0] == pytest.approx(32.0 - 30.0)

    def test_region_round_trips_through_a_dict(self):
        r = dpc.BeamRegion("ring", 1.5, 2.5, 9.0, 3.0)
        assert dpc.BeamRegion.from_dict(r.as_dict()) == r
        assert dpc.BeamRegion.from_dict(None).shape == "circle"
        assert dpc.BeamRegion.from_dict({"shape": "nonsense"}).shape == "circle"
        assert dpc.BeamRegion.from_dict({"shape": "off"}).shape == "circle", \
            "a saved report from before the region was always-on must still load"

    def test_default_region_is_the_largest_that_fits(self):
        """It must not CLIP the beam. The region is always on the pattern, so
        this default is what an unattended scan is measured with, and a smaller
        circle pulls the centroid towards its own centre — see
        ``default_beam_region`` for the measured under-read."""
        d = dpc.default_beam_region((256, 512))
        assert (d.cx, d.cy) == (256.0, 128.0)      # centred on the DETECTOR
        assert d.r == 128.0                         # HALF the short axis
        assert 0 < d.r_inner < d.r
        assert d.active

    def test_the_default_region_does_not_bias_the_shift(self):
        """The claim the radius rests on, checked rather than asserted: on a
        centred disc displaced by a known amount, the default region returns the
        SAME shift the whole frame does."""
        from scipy import ndimage as ndi
        k = 32
        gy, gx = np.mgrid[0:k, 0:k].astype(np.float32)
        region = dpc.default_beam_region((k, k))
        keep = region.mask((k, k))
        for true_shift in (1.5, 3.0, 4.5):
            r = np.hypot(gx - (k / 2 - true_shift), gy - k / 2)
            frame = np.clip((k * 0.22 - r) / (k * 0.05) + 0.5, 0, 1)
            masked = dpc._com_shift_frame(frame, keep, k / 2, k / 2)
            whole_cy, whole_cx = ndi.center_of_mass(frame)
            assert masked[0] == pytest.approx(k / 2 - whole_cx, abs=0.02)
            assert masked[0] == pytest.approx(true_shift, abs=0.05)

    def test_the_region_lands_in_provenance(self):
        s, _k = self._scan()
        region = dpc.BeamRegion("ring", 40.0, 36.0, 12.0, 3.0)
        r = dpc.compute_dpc(s, mode="magnetic", center_mode="none",
                            region=region, rotation=0.0)
        assert r.params["beam_region"] == region.as_dict()


class TestCentering:
    def test_a_zero_field_is_centered(self):
        assert dpc.centering_report(np.zeros((20, 30, 2))).centered

    def test_an_offset_is_not_centered(self):
        rep = dpc.centering_report(_uniform(3.0, 0.0, n=20))
        assert not rep.centered
        assert rep.offset[0] == pytest.approx(3.0)

    def test_a_ramp_with_zero_mean_is_not_centered(self):
        """Offset and ramp are both needed; either alone misses a real case.

        A ramp centred on zero has a mean of zero — judged on the mean alone,
        the worst descan in the suite would be reported as already correct.
        """
        shifts = np.zeros((20, 20, 2))
        shifts[..., 0] = np.mgrid[0:20, 0:20][1] * 0.5 - 4.75
        rep = dpc.centering_report(shifts)
        assert abs(rep.offset[0]) < 1e-9
        assert rep.ramp[0] == pytest.approx(9.5, abs=0.01)
        assert not rep.centered

    def test_manual_reference_is_the_constant_pyxem_would_report(self):
        """``centre − picked``, matching ``get_direct_beam_position``'s sign."""
        ref = dpc.constant_reference((5, 6), (30.0, 12.0), (48, 64))
        assert ref[..., 0] == pytest.approx(64 / 2.0 - 30.0)
        assert ref[..., 1] == pytest.approx(48 / 2.0 - 12.0)
        assert ref.shape == (5, 6, 2)

    def test_vacuum_reference_removes_itself(self):
        shifts = np.zeros((20, 20, 2))
        rows, cols = np.mgrid[0:20, 0:20]
        shifts[..., 0] = 0.3 * cols + 1.0
        shifts[..., 1] = -0.2 * rows - 2.0
        residual = dpc.apply_reference(shifts, dpc.vacuum_reference(shifts, shifts))
        assert np.abs(residual).max() < 1e-9

    def test_vacuum_reference_smooths_away_shot_noise(self):
        """The default fits a plane, so the vacuum scan's noise stays out of the
        sample's field — subtracting it verbatim would inject it."""
        rows, cols = np.mgrid[0:24, 0:24]
        clean = np.stack([0.2 * cols, 0.1 * rows], axis=-1).astype(float)
        noisy = clean + np.random.RandomState(0).normal(0, 0.5, clean.shape)
        smoothed = dpc.vacuum_reference(clean, noisy, smooth=True)
        verbatim = dpc.vacuum_reference(clean, noisy, smooth=False)
        assert np.abs(smoothed - clean).max() < np.abs(verbatim - clean).max()

    def test_vacuum_reference_resamples_a_coarser_scan(self):
        """Documented assumption: same field of view, different sampling."""
        rows, cols = np.mgrid[0:41, 0:41]
        fine = np.stack([0.2 * cols, 0.1 * rows], axis=-1).astype(float)
        coarse_r, coarse_c = np.mgrid[0:21, 0:21] * 2.0
        coarse = np.stack([0.2 * coarse_c, 0.1 * coarse_r], axis=-1)
        ref = dpc.vacuum_reference(fine, coarse)
        assert ref.shape == fine.shape
        assert np.abs(ref - fine).max() < 1e-9

    def test_a_malformed_vacuum_field_is_rejected(self):
        with pytest.raises(ValueError, match="beam-shift field"):
            dpc.vacuum_reference(np.zeros((4, 4, 2)), np.zeros((4, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# Rotation
# ─────────────────────────────────────────────────────────────────────────────

class TestRotation:
    def test_rotate_matches_pyxem(self):
        """SpyDE's angle and pyxem's ``rotate_beam_shifts`` angle mean the same."""
        shifts = _curl_free_field(16)
        ours = dpc.rotate_shifts(shifts, 37.0)
        theirs = dpc.as_beam_shift(shifts).rotate_beam_shifts(37.0).data
        assert np.allclose(ours, theirs)

    def test_flip_is_applied_before_the_rotation(self):
        """Order matters — a mirror does not commute with a rotation.

        Flipping afterwards silently negates the fitted angle, so a solver
        result and a manual slider would disagree by 2θ.
        """
        shifts = _curl_free_field(16)
        got = dpc.rotate_shifts(shifts, 30.0, flip=True)
        expect = dpc.rotate_shifts(np.stack([shifts[..., 1], shifts[..., 0]], -1),
                                   30.0)
        assert np.allclose(got, expect)

    def test_reverse_is_exactly_180(self):
        shifts = _curl_free_field(8)
        assert np.allclose(dpc.rotate_shifts(shifts, 20.0, reverse=True),
                           dpc.rotate_shifts(shifts, 200.0))

    @pytest.mark.parametrize("truth", [0.0, 37.0, 90.0, 137.0, 220.0, 310.0])
    def test_electric_rotation_recovered(self, truth):
        """A curl-free field pins the rotation, modulo the 180° it cannot see."""
        observed = dpc.rotate_shifts(_curl_free_field(), -truth)
        est = dpc.estimate_rotation(observed, mode="electric")
        err = min(abs((est.angle - truth) % 180.0),
                  180.0 - abs((est.angle - truth) % 180.0))
        assert err < 1.5, f"estimated {est.angle}°, truth {truth}°"
        assert not est.flip

    @pytest.mark.parametrize("truth", [0.0, 55.0, 143.0, 265.0])
    def test_magnetic_rotation_recovered(self, truth):
        """A divergence-free field pins it through the other symmetry."""
        observed = dpc.rotate_shifts(_div_free_field(), -truth)
        est = dpc.estimate_rotation(observed, mode="magnetic")
        err = min(abs((est.angle - truth) % 180.0),
                  180.0 - abs((est.angle - truth) % 180.0))
        assert err < 1.5, f"estimated {est.angle}°, truth {truth}°"

    def test_handedness_is_detected(self):
        """A detector reading out with the opposite chirality is not a rotation.

        No angle can undo a mirror, so without the flip search the solver would
        return a confident, wrong angle rather than failing visibly.
        """
        field = _curl_free_field()
        observed = dpc.rotate_shifts(field, -73.0)
        mirrored = np.stack([observed[..., 1], observed[..., 0]], axis=-1)
        est = dpc.estimate_rotation(mirrored, mode="electric")
        assert est.flip is True
        err = min(abs((est.angle - 73.0) % 180.0),
                  180.0 - abs((est.angle - 73.0) % 180.0))
        assert err < 1.5

    def test_the_fit_reports_how_well_it_did(self):
        """`improvement` is the caret's honesty signal — a field with no
        preferred angle must NOT come back looking confident."""
        strong = dpc.estimate_rotation(dpc.rotate_shifts(_curl_free_field(), -40.0),
                                       mode="electric")
        assert strong.improvement > 10
        rng = np.random.RandomState(3)
        noise = rng.normal(0, 1, (64, 64, 2))
        weak = dpc.estimate_rotation(noise, mode="electric")
        assert weak.improvement < 3

    def test_180_is_genuinely_ambiguous(self):
        """Documented, not a bug: both residuals are even under θ → θ+180."""
        observed = dpc.rotate_shifts(_curl_free_field(), -40.0)
        est = dpc.estimate_rotation(observed, mode="electric")
        _, curl_a = dpc.field_symmetry(dpc.rotate_shifts(observed, est.angle))
        _, curl_b = dpc.field_symmetry(
            dpc.rotate_shifts(observed, est.angle, reverse=True))
        assert curl_a == pytest.approx(curl_b, rel=1e-9)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="mode must be"):
            dpc.estimate_rotation(_curl_free_field(8), mode="gravitational")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration + the sign convention
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibration:
    def test_mrad_per_pixel_from_reciprocal_axes(self):
        """θ = k·λ — the same relation pyxem's convert_signal_units uses."""
        s = _pn_junction()
        expect = s.axes_manager.signal_axes[0].scale
        s2 = _pn_junction()
        s2.calibration.convert_signal_units("mrad")
        assert dpc.mrad_per_pixel(s2) == pytest.approx(
            s2.axes_manager.signal_axes[0].scale)
        assert dpc.mrad_per_pixel(s) == pytest.approx(
            expect * dpc._wavelength_nm(200) * 1e3, rel=1e-9)

    def test_uncalibrated_axes_return_none(self):
        import hyperspy.api as hs
        s = hs.signals.Signal2D(np.zeros((2, 2, 8, 8), np.float32))
        assert dpc.mrad_per_pixel(s) is None

    def test_electric_field_matches_pyxem_exactly(self):
        """The published chain, bit for bit.

        ``pixels_to_calibrated_units().calibrate_electric_shifts(t)`` is what the
        pyxem electric-field example runs. SpyDE reaches the same numbers by a
        different route (explicit mrad/px, so an uncalibrated file still works),
        so this is the check that the route is equivalent and not merely close.
        """
        s = _pn_junction()
        s.calibration.convert_signal_units("mrad")
        com = s.get_direct_beam_position(method="center_of_mass", lazy_output=False)
        reference = np.asarray(com.pixels_to_calibrated_units()
                               .calibrate_electric_shifts(thickness=60).data)
        ours = dpc.to_electric_field(
            dpc.measure_beam_shifts(s), mrad_per_px=dpc.mrad_per_pixel(s),
            thickness_nm=60, beam_energy_kev=200, like=s)
        assert np.array_equal(ours, reference)

    @pytest.mark.parametrize("kwargs", [
        {"mrad_per_px": 0.0, "thickness_nm": 60, "beam_energy_kev": 200},
        {"mrad_per_px": 0.1, "thickness_nm": 0, "beam_energy_kev": 200},
        {"mrad_per_px": 0.1, "thickness_nm": 60, "beam_energy_kev": -1},
    ])
    def test_nonsense_calibration_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            dpc.to_electric_field(np.zeros((2, 2, 2)), **kwargs)


class TestSignConvention:
    """Neither sign is arbitrary, and neither is readable off the code."""

    def test_get_direct_beam_position_is_centre_minus_beam(self):
        """The whole pipeline's sign rests on this. If pyxem ever flips it,
        every DPC map in SpyDE flips with it — so assert it directly rather
        than inheriting it."""
        import pyxem as pxm
        k = 32
        gy, gx = np.mgrid[0:k, 0:k]
        beam = ((gy - (k / 2.0 + 3)) ** 2 + (gx - (k / 2.0 + 5)) ** 2) < 16
        data = np.tile(beam.astype(np.float32), (3, 4, 1, 1))
        shifts = dpc.measure_beam_shifts(pxm.signals.ElectronDiffraction2D(data))
        assert shifts[..., 0].mean() == pytest.approx(-5.0, abs=0.1)
        assert shifts[..., 1].mean() == pytest.approx(-3.0, abs=0.1)

    def test_magnetic_is_parallel_to_the_beam_shift(self):
        s = _pn_junction()
        r = dpc.compute_dpc(s, mode="magnetic", center_mode="none", rotation=0.0)
        assert np.allclose(np.sign(r.field), np.sign(r.raw_shifts))

    def test_electric_is_antiparallel_to_the_beam_shift(self):
        """``calibrate_electric_shifts`` divides by ``-e``. This is pyxem's
        published convention and the user chose to match it; a future change
        here would silently invert every electric-field figure."""
        s = _pn_junction()
        s.calibration.convert_signal_units("mrad")
        r = dpc.compute_dpc(s, mode="electric", center_mode="none", rotation=0.0,
                            thickness_nm=60, beam_energy_kev=200)
        assert r.units == "MV/cm"
        strong = np.abs(r.raw_shifts) > 0.5
        assert np.all(np.sign(r.field[strong]) == -np.sign(r.raw_shifts[strong]))


# ─────────────────────────────────────────────────────────────────────────────
# The pipeline end to end, against the synthetic fixture's ground truth
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dpc_truth():
    """The bundled synthetic DPC scan, built exactly as the app's test-data
    loader builds it, with its ground truth."""
    from spyde.backend._session_testharness import TestHarnessMixin  # noqa: F401
    return _build_fixture()


def _build_fixture(n=32, k=48, rotation=25.0, offset=(1.5, -1.0),
                   ramp=(2.0, -1.5), amp=3.0):
    import hyperspy.api as hs
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
    V = (np.exp(-(((xx - 0.35) ** 2 + (yy - 0.40) ** 2) / 0.018))
         - np.exp(-(((xx - 0.68) ** 2 + (yy - 0.62) ** 2) / 0.012)))
    fy, fx = np.gradient(-V)
    true_field = np.stack([fx, fy], -1) * (amp / float(np.hypot(fx, fy).max()))
    observed = dpc.rotate_shifts(true_field, -rotation)
    observed[..., 0] += offset[0] + ramp[0] * (xx - 0.5)
    observed[..., 1] += offset[1] + ramp[1] * (yy - 0.5)

    gy, gx = np.mgrid[0:k, 0:k].astype(np.float32)
    radius, edge = k * 0.22, k * 0.05
    data = np.empty((n, n, k, k), np.float32)
    for iy in range(n):
        for ix in range(n):
            r = np.hypot(gx - (k / 2.0 - observed[iy, ix, 0]),
                         gy - (k / 2.0 - observed[iy, ix, 1]))
            data[iy, ix] = 1000.0 / (1.0 + np.exp((r - radius) / edge))
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax, name in zip(s.axes_manager.signal_axes, ("kx", "ky")):
        ax.scale, ax.units, ax.name = 0.002, "nm^-1", name
        ax.offset = -(ax.size / 2.0) * float(ax.scale)
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200)
    return {"signal": s, "shift_field": true_field, "observed": observed,
            "rotation": rotation, "offset": offset, "ramp": ramp, "amp": amp}


class TestGroundTruth:
    def test_the_reader_returns_the_shifts_that_were_baked_in(self, dpc_truth):
        measured = dpc.measure_beam_shifts(dpc_truth["signal"])
        assert np.abs(measured - dpc_truth["observed"]).max() < 0.1

    def test_the_corner_plane_recovers_the_field(self, dpc_truth):
        measured = dpc.measure_beam_shifts(dpc_truth["signal"])
        centered = dpc.apply_reference(measured,
                                       dpc.corner_reference(measured, 0.08))
        expect = dpc.rotate_shifts(dpc_truth["shift_field"], -dpc_truth["rotation"])
        assert np.abs(centered - expect).max() < 0.25

    def test_the_rotation_is_recovered_from_the_data(self, dpc_truth):
        measured = dpc.measure_beam_shifts(dpc_truth["signal"])
        centered = dpc.apply_reference(measured,
                                       dpc.corner_reference(measured, 0.08))
        est = dpc.estimate_rotation(centered, mode="electric")
        err = min(abs((est.angle - dpc_truth["rotation"]) % 180.0),
                  180.0 - abs((est.angle - dpc_truth["rotation"]) % 180.0))
        assert err < 2.0 and not est.flip
        assert est.improvement > 20

    def test_directions_match_the_truth_where_the_field_is_strong(self, dpc_truth):
        """The end-to-end direction check — the one an eyeballed map cannot do."""
        r = dpc.compute_dpc(dpc_truth["signal"], mode="magnetic",
                            center_mode="corners", corner_fraction=0.08,
                            rotation=dpc_truth["rotation"])
        truth = dpc_truth["shift_field"]
        mag = np.hypot(truth[..., 0], truth[..., 1])
        strong = mag > 0.35 * mag.max()
        delta = np.angle(np.exp(1j * (np.arctan2(r.fy[strong], r.fx[strong])
                                      - np.arctan2(truth[..., 1][strong],
                                                   truth[..., 0][strong]))))
        assert np.median(np.abs(np.degrees(delta))) < 5.0

    def test_the_estimator_must_match_the_field_being_measured(self, dpc_truth):
        """The fixture's field is CURL-free, so only ``mode="electric"`` recovers
        its rotation; the magnetic constraint lands ~90° away.

        Not a defect — it is what choosing a mode MEANS. The two modes assert
        different symmetries, so picking the wrong one gives a confident answer
        that is a right angle out, and every direction on the map with it. Pinned
        here because it is the most plausible way to misuse this feature.
        """
        r = dpc.compute_dpc(dpc_truth["signal"], center_mode="corners",
                            corner_fraction=0.08, mode="electric",
                            rotation=None, auto_rotation=True)
        err = min(abs((r.rotation - dpc_truth["rotation"]) % 180.0),
                  180.0 - abs((r.rotation - dpc_truth["rotation"]) % 180.0))
        assert err < 2.0

        wrong = dpc.compute_dpc(dpc_truth["signal"], center_mode="corners",
                                corner_fraction=0.08, mode="magnetic",
                                rotation=None, auto_rotation=True)
        off = min(abs((wrong.rotation - dpc_truth["rotation"]) % 180.0),
                  180.0 - abs((wrong.rotation - dpc_truth["rotation"]) % 180.0))
        assert off > 80.0

    def test_the_vacuum_path_is_exact(self, dpc_truth):
        """An exact descan reference reproduces the field exactly — in the
        detector's own angular units, since this dataset IS calibrated."""
        n = dpc_truth["shift_field"].shape[0]
        yy, xx = np.mgrid[0:n, 0:n] / (n - 1.0)
        vac = np.zeros((n, n, 2))
        vac[..., 0] = dpc_truth["offset"][0] + dpc_truth["ramp"][0] * (xx - 0.5)
        vac[..., 1] = dpc_truth["offset"][1] + dpc_truth["ramp"][1] * (yy - 0.5)
        r = dpc.compute_dpc(dpc_truth["signal"], mode="magnetic",
                            center_mode="vacuum", vacuum_shifts=vac,
                            rotation=dpc_truth["rotation"])
        assert r.units == "mrad"
        scale = dpc.mrad_per_pixel(dpc_truth["signal"])
        assert np.abs(r.field - dpc_truth["shift_field"] * scale).max() < 0.15 * scale

    def test_magnetic_units_follow_the_calibration(self, dpc_truth):
        """mrad when the detector scale is known, raw pixels when it is not —
        and ``units`` says which, so a map is never silently mislabelled."""
        calibrated = dpc.compute_dpc(dpc_truth["signal"], mode="magnetic",
                                     center_mode="none", rotation=0.0)
        assert calibrated.units == "mrad"

        import copy
        bare = copy.deepcopy(dpc_truth["signal"])
        for ax in bare.axes_manager.signal_axes:
            ax.units, ax.scale = "px", 1.0
        bare.metadata.Acquisition_instrument.TEM.beam_energy = None
        raw = dpc.compute_dpc(bare, mode="magnetic", center_mode="none",
                              rotation=0.0)
        assert raw.units == "px"

    def test_the_manual_path_removes_the_constant(self, dpc_truth):
        k = dpc_truth["signal"].axes_manager.signal_axes[0].size
        r = dpc.compute_dpc(
            dpc_truth["signal"], mode="magnetic", center_mode="manual",
            center_xy=(k / 2.0 - dpc_truth["offset"][0],
                       k / 2.0 - dpc_truth["offset"][1]),
            rotation=dpc_truth["rotation"])
        residual = r.raw_shifts - r.reference
        assert abs(residual[..., 0].mean()) < 0.15
        assert abs(residual[..., 1].mean()) < 0.15

    def test_result_exposes_every_component(self, dpc_truth):
        r = dpc.compute_dpc(dpc_truth["signal"], mode="magnetic",
                            center_mode="corners", rotation=0.0)
        for name in dpc.COMPONENTS:
            assert r.component(name).shape == r.field.shape[:2]
        assert r.rgb.shape == r.field.shape[:2] + (3,)
        assert r.wheel.ndim == 3 and r.wheel.shape[2] == 4

    def test_bad_modes_are_refused(self, dpc_truth):
        with pytest.raises(ValueError, match="mode must be"):
            dpc.compute_dpc(dpc_truth["signal"], mode="thermal")
        with pytest.raises(ValueError, match="center_mode must be"):
            dpc.compute_dpc(dpc_truth["signal"], center_mode="vibes")

    def test_manual_without_a_centre_is_refused(self, dpc_truth):
        with pytest.raises(ValueError, match="center_xy"):
            dpc.compute_dpc(dpc_truth["signal"], center_mode="manual")


# ─────────────────────────────────────────────────────────────────────────────
# Memory safety
# ─────────────────────────────────────────────────────────────────────────────

class TestLazy:
    """The pass over a lazy dataset must stream, and must stream ALIGNED.

    Live-Display §1: a 4D-STEM signal is chunked so each chunk holds whole
    frames. If the beam-shift graph rechunked, the streaming granularity would
    stop matching what the reader reads and every "chunk" would touch several
    storage chunks — the shuffle that section exists to forbid.
    """

    @staticmethod
    def _lazy(nav=16, sig=32, nav_chunk=8):
        import dask.array as da
        import hyperspy.api as hs
        gy, gx = np.mgrid[0:sig, 0:sig].astype(np.float32)
        frame = 1000.0 / (1 + np.exp((np.hypot(gx - (sig / 2 + 2),
                                               gy - (sig / 2 - 2)) - sig * 0.2)
                                     / 1.5))
        arr = da.from_array(np.tile(frame, (nav, nav, 1, 1)),
                            chunks=(nav_chunk, nav_chunk, sig, sig))
        s = hs.signals.Signal2D(arr).as_lazy()
        s.set_signal_type("electron_diffraction")
        return s, arr

    def test_the_graph_is_lazy_and_keeps_the_nav_chunking(self):
        import dask.array as da
        s, arr = self._lazy()
        for region in (None, dpc.BeamRegion("circle", 18.0, 14.0, 8.0)):
            graph = dpc.beam_shift_graph(s, region=region)
            assert isinstance(graph, da.Array)
            assert graph.chunks[:2] == arr.chunks[:2], \
                "the nav chunking changed — streaming would stop matching storage"
            assert not [k for k in graph.dask.layers
                        if "rechunk" in str(k).lower()], "a rechunk crept in"

    def test_eager_data_has_no_graph(self):
        """`None` is the signal to the wizard that there is nothing to stream."""
        import hyperspy.api as hs
        s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), np.float32))
        s.set_signal_type("electron_diffraction")
        assert dpc.beam_shift_graph(s) is None

    def test_the_kernel_sees_one_chunk_and_allocates_only_the_result(self):
        """Peak memory is a chunk — never the dataset — and the kernel is
        VECTORISED over that chunk rather than called per frame.

        It used to be a Python call per frame with a ``scipy.ndimage`` call
        inside it, which is why a pass over a real scan took most of a minute:
        at tens of thousands of frames the per-call overhead IS the runtime.
        One ``einsum`` per chunk measured 37x faster on a 64x64x64x64 scan and
        returns bit-identical numbers (see
        ``test_the_vectorised_kernel_matches_the_per_frame_reference``).

        What still has to hold is the memory rule: the kernel must not
        materialise a float64 copy of the block it is handed. It contracts the
        block against detector-sized weights instead, so the only thing it
        allocates is the nav-sized result.
        """
        s, arr = self._lazy()
        seen = {"blocks": [], "n": 0}
        real = dpc._com_shift_blocks

        def spy(block, *a, **kw):
            seen["n"] += 1
            seen["blocks"].append(np.asarray(block).nbytes)
            out = real(block, *a, **kw)
            assert out.nbytes < np.asarray(block).nbytes, \
                "the kernel returned something the size of its input"
            return out

        dpc._com_shift_blocks = spy
        try:
            shifts = dpc.measure_beam_shifts(
                s, region=dpc.BeamRegion("circle", 18.0, 14.0, 8.0))
        finally:
            dpc._com_shift_blocks = real

        chunk_bytes = max(
            np.prod([c[0] for c in arr.chunks]) * arr.dtype.itemsize, 1)
        assert seen["n"] == len(arr.chunks[0]) * len(arr.chunks[1]), \
            "the kernel ran per frame, not per chunk"
        assert max(seen["blocks"]) <= chunk_bytes * 1.01, \
            "the kernel was handed more than one chunk"
        assert shifts.shape == (arr.shape[0], arr.shape[1], 2)

    def test_the_vectorised_kernel_matches_the_per_frame_reference(self):
        """The fast path and the reference must agree EXACTLY, not closely.

        ``_com_shift_frame`` is the documented equivalence to pyxem's
        ``center_of_mass_from_image`` (``TestBeamRegion`` pins that), so the
        vectorised kernel inherits that guarantee only for as long as the two
        return the same bits. ``allclose`` would let a real drift through.
        """
        rng = np.random.default_rng(0)
        k = 16
        block = (rng.random((3, 4, k, k)) * 1000).astype(np.uint16)
        region = dpc.BeamRegion("circle", k / 2, k / 2, k * 0.4)
        keep = region.mask((k, k))

        fast = dpc._com_shift_blocks(
            block, dpc._region_weights(keep, (k, k)), k / 2, k / 2)
        for iy in range(block.shape[0]):
            for ix in range(block.shape[1]):
                one = dpc._com_shift_frame(block[iy, ix], keep, k / 2, k / 2)
                assert np.array_equal(fast[iy, ix], one), (iy, ix)

    def test_an_empty_region_reads_as_no_measurement(self):
        """A region with nothing under it yields NaN, not a centred beam.

        Zero is the dangerous answer: it is exactly what an undeflected beam
        looks like, so it would show as a real measurement of no field.
        """
        k = 16
        block = np.zeros((2, 2, k, k), dtype=np.uint16)
        block[1, 1, k // 2, k // 2] = 500          # one frame has signal
        region = dpc.BeamRegion("circle", k / 2, k / 2, k * 0.4)
        out = dpc._com_shift_blocks(
            block, dpc._region_weights(region.mask((k, k)), (k, k)),
            k / 2, k / 2)
        assert np.isnan(out[0, 0]).all(), "an empty region reported a position"
        assert np.isfinite(out[1, 1]).all(), "a real frame was dropped"


    def test_a_partial_field_survives_every_downstream_stage(self):
        """The whole point of streaming: NaN where nothing has landed yet must
        flow through centering, rotation and display without poisoning them."""
        s, _arr = self._lazy()
        partial = np.full((16, 16, 2), np.nan)
        partial[:8, :8] = dpc.measure_beam_shifts(s)[:8, :8]   # one chunk in

        report = dpc.centering_report(partial)
        assert np.isfinite(report.worst)
        ref = dpc.corner_reference(partial, 0.2)
        assert np.isfinite(ref).all(), "the plane fit produced non-finite values"
        est = dpc.estimate_rotation(dpc.apply_reference(partial, ref),
                                    mode="electric")
        assert np.isfinite(est.angle)
        rgb = dpc.magnitude_phase_rgb(partial)
        assert rgb.shape == (16, 16, 3) and rgb.dtype == np.uint8

    def test_unmeasured_positions_paint_black_not_broken(self):
        """A single NaN used to take out the whole map: `autolim` derives its
        limits from a mean and a standard deviation."""
        field = _curl_free_field(16) * 5.0
        holed = field.copy()
        holed[4:8, 4:8] = np.nan
        rgb = dpc.magnitude_phase_rgb(holed)
        assert rgb[6, 6].max() == 0, "unmeasured should read as zero field"
        assert rgb[12, 12].max() > 0, "the measured part must still be coloured"
        assert np.array_equal(dpc.magnitude_phase_rgb(field)[12, 12],
                              rgb[12, 12]), \
            "a hole must not change the colours elsewhere"


class TestMemorySafety:
    def test_measuring_shifts_never_materialises_a_lazy_dataset(self):
        """``get_direct_beam_position`` is a REDUCTION — only the ``(ny, nx, 2)``
        result may be computed. The CLAUDE.md rule, enforced the same way
        ``test_find_vectors_memory`` enforces it for find-vectors."""
        import dask.array as da
        import hyperspy.api as hs
        from unittest.mock import patch

        n, k = 6, 16
        gy, gx = np.mgrid[0:k, 0:k]
        frame = (((gy - k / 2) ** 2 + (gx - k / 2) ** 2) < 9).astype(np.float32)
        arr = da.from_array(np.tile(frame, (n, n, 1, 1)), chunks=(2, 2, k, k))
        s = hs.signals.Signal2D(arr).as_lazy()
        s.set_signal_type("electron_diffraction")

        full = (n, n, k, k)
        seen: list[tuple] = []
        real_compute = da.Array.compute

        def guarded(self, *a, **kw):
            seen.append(tuple(self.shape))
            assert tuple(self.shape) != full, \
                "the FULL dataset was computed — see the CLAUDE.md memory rule"
            return real_compute(self, *a, **kw)

        with patch.object(da.Array, "compute", guarded):
            shifts = dpc.measure_beam_shifts(s)
        assert shifts.shape == (n, n, 2)


class TestPrivateView:
    """Two DPC passes overlap routinely — StrictMode's open/close/open starts a
    second measure while the first is still on a worker, because the generation
    guard drops the superseded RESULT and not the superseded WORK. Both then run
    hyperspy ``map``, and hyperspy mutates the signal a method is called on."""

    def _signal(self):
        import hyperspy.api as hs
        s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.float32))
        s.axes_manager.signal_axes[0].scale = 0.25
        s.axes_manager.signal_axes[0].units = "mrad"
        s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 200.0)
        return s

    def test_the_view_is_a_different_object_over_the_same_buffer(self):
        s = self._signal()
        view = dpc.private_view(s)
        assert view is not s
        assert view.data is s.data, "the data buffer must be shared, not copied"

    def test_the_view_keeps_what_the_pipeline_reads(self):
        s = self._signal()
        view = dpc.private_view(s)
        assert view.axes_manager.signal_axes[0].scale == 0.25
        assert view.axes_manager.signal_axes[0].units == "mrad"
        assert dpc.beam_energy_kv(view) == 200.0
        assert view.axes_manager.navigation_shape == \
            s.axes_manager.navigation_shape

    def test_a_lazy_signal_stays_lazy(self):
        import dask.array as da
        import hyperspy.api as hs
        s = hs.signals.Signal2D(
            da.zeros((4, 4, 8, 8), chunks=(2, 2, 8, 8))).as_lazy()
        view = dpc.private_view(s)
        assert view._lazy and hasattr(view.data, "chunks")

    def test_setting_the_signal_type_does_not_touch_the_original(self):
        """``measure_beam_shifts`` calls ``set_signal_type`` on what it is
        given. On the shared object that is a worker thread mutating the tree's
        signal out from under the main thread."""
        s = self._signal()
        before = s.__class__
        dpc.measure_beam_shifts(dpc.private_view(s))
        assert s.__class__ is before

    def test_a_deepcopy_transiently_publishes_a_placeholder(self):
        """WHY the view exists, pinned so the reasoning cannot rot.

        ``_deepcopy_with_new_data`` sets ``self.data = None`` on the LIVE object
        while it copies the wrapper; hyperspy's setter stores that as a length-1
        OBJECT array. A concurrent ``map`` reading it there dies with "Chunks do
        not add up to shape ... shape=(1,)" — the CI failure this fixes."""
        from unittest.mock import patch
        s = self._signal()
        seen = []
        real = type(s).deepcopy

        def spy(self, *a, **kw):
            seen.append(getattr(self.data, "shape", None))
            return real(self, *a, **kw)

        with patch.object(type(s), "deepcopy", spy):
            s._deepcopy_with_new_data(s.data)
        assert seen == [(1,)], \
            f"expected the (1,) placeholder mid-copy, saw {seen}"
        assert s.data.shape == (4, 4, 8, 8), "the original must be restored"


# ─────────────────────────────────────────────────────────────────────────────
# Host parity: the schema, the caret defaults, the api wrapper
# ─────────────────────────────────────────────────────────────────────────────

_TSX = (Path(__file__).resolve().parents[3] / "electron" / "src" / "renderer"
        / "src" / "components" / "DpcWizard.tsx")


class TestHostParity:
    def test_schema_resolves_through_the_registry(self):
        from spyde.actions import registry
        schema = registry.wizard_parameters("dpc")
        assert schema and "rotation" in schema and "center_mode" in schema

    def test_schema_defaults_match_the_handler_defaults(self):
        from spyde.actions import registry
        from spyde.actions.dpc_action import DEFAULTS
        schema = registry.wizard_parameters("dpc")
        for key, spec in schema.items():
            assert spec["default"] == DEFAULTS[key], \
                f"dpc schema/{key} drifted from dpc_action.DEFAULTS"

    def test_every_stage_is_registered(self):
        from spyde.actions import registry
        for stage in ("open", "close", "set_center", "pick_center",
                      "load_vacuum", "auto_rotation", "tune", "set_view",
                      "run", "commit"):
            assert registry.resolve_staged(f"dpc_{stage}") is not None

    @pytest.mark.skipif(not _TSX.exists(), reason="renderer sources not present")
    def test_caret_defaults_match_the_backend(self):
        """Parse ``DpcWizard.tsx``'s DEFAULTS and compare, key for key.

        A caret default that drifts from the Python one WINS SILENTLY: the caret
        sends its own value in every payload, so the backend's DEFAULTS is never
        consulted and nothing fails. That has cost this project a session before
        (see the caret-defaults note in CLAUDE.md), which is why this reads the
        TSX rather than trusting a comment that says the two agree.
        """
        from spyde.actions.dpc_action import DEFAULTS
        text = _TSX.read_text(encoding="utf-8")
        block = re.search(r"const DEFAULTS:\s*DpcSaved\s*=\s*\{(.*?)\n\}",
                          text, re.S)
        assert block, "could not find the DEFAULTS literal in DpcWizard.tsx"
        found = dict(re.findall(r"(\w+):\s*('[^']*'|[-\w.]+)\s*,", block.group(1)))
        assert found, "DEFAULTS literal parsed to nothing — the regex has rotted"

        def camel(snake: str) -> str:
            head, *rest = snake.split("_")
            return head + "".join(w.title() for w in rest)

        for key, py_value in DEFAULTS.items():
            raw = found.get(camel(key))
            assert raw is not None, f"DpcWizard.tsx has no default for {key}"
            if isinstance(py_value, bool):
                ts_value = raw == "true"
            elif isinstance(py_value, (int, float)):
                ts_value = float(raw)
                py_value = float(py_value)
            else:
                ts_value = raw.strip("'")
            assert ts_value == py_value, (
                f"DpcWizard.tsx {camel(key)}={raw} but dpc_action.DEFAULTS "
                f"[{key}]={py_value!r} — the caret's value would win silently")

    @pytest.mark.skipif(not _TSX.exists(), reason="renderer sources not present")
    def test_the_guard_would_catch_a_drifted_value(self):
        """A guard nobody has seen fail is not a guard. Mutate the parsed text
        and confirm the comparison rejects it."""
        text = _TSX.read_text(encoding="utf-8")
        block = re.search(r"const DEFAULTS:\s*DpcSaved\s*=\s*\{(.*?)\n\}",
                          text, re.S).group(1)
        drifted = block.replace("cornerFraction: 0.05", "cornerFraction: 0.25")
        found = dict(re.findall(r"(\w+):\s*('[^']*'|[-\w.]+)\s*,", drifted))
        from spyde.actions.dpc_action import DEFAULTS
        assert float(found["cornerFraction"]) != float(DEFAULTS["corner_fraction"])

    def test_api_wrapper_matches_the_action(self, dpc_truth):
        """``spyde.api.dpc_map`` and the wizard must be the same computation."""
        from spyde import api
        scripted = api.dpc_map(dpc_truth["signal"], mode="magnetic",
                               center="corners", corner_fraction=0.08,
                               rotation=dpc_truth["rotation"])
        direct = dpc.compute_dpc(dpc_truth["signal"], mode="magnetic",
                                 center_mode="corners", corner_fraction=0.08,
                                 rotation=dpc_truth["rotation"])
        assert np.array_equal(scripted.field, direct.field)
        assert scripted.provenance["action"] == "dpc_map"

    def test_api_does_not_import_the_ui_stack(self):
        """``spyde.api`` must stay importable with no UI — the module rule."""
        import subprocess, sys
        code = ("import sys, spyde.api as a; a.dpc_map;"
                "assert not [m for m in sys.modules if m.startswith"
                "(('spyde.backend', 'spyde.drawing'))], "
                "[m for m in sys.modules if m.startswith(('spyde.backend','spyde.drawing'))]")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True)
        assert out.returncode == 0, out.stderr


def _pn_junction():
    """pyxem's simulated p-n junction — the electric-field example's dataset."""
    import pyxem as pxm
    return pxm.data.simulated_pn_junction()
