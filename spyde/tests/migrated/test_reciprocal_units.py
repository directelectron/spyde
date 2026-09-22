"""
Reciprocal units (spyde/reciprocal_units.py) and the paths that depend on them.

Two layers, and the second is the point of the file:

  - the converter itself — parsing the spellings a file might carry, the
    factors, and a toggle that round-trips through every unit without drift.
  - **parity**: one physical dataset, written down four different ways, has to
    give the SAME crystallographic answer. Å⁻¹ is what diffsims, orix and
    pyxem's matcher all work in and none of them checks, so a path that reads
    ``ax.scale`` directly instead of converting returns an answer that is a
    factor of ten wrong and raises nothing. A parity assertion catches that;
    reading the code for it does not.

The ZrNb precipitate scan — calibrated in nm⁻¹, which is what made this
necessary — is the case the parity tests are modelled on.
"""
import numpy as np
import pytest

from spyde import reciprocal_units as ru


# The four ways to write down ONE detector: 0.01 Å⁻¹ per pixel, 64 px square,
# origin at the centre. At 200 kV that is also 0.1 nm⁻¹/px and, through
# θ = k·λ, 0.2508 mrad/px.
VOLTAGE_KV = 200.0
WAVELENGTH = ru.electron_wavelength_angstrom(VOLTAGE_KV)
INVERSE_ANGSTROM_PER_PIXEL = 0.01
DETECTOR_SIZE = 64


def _signal(scale, units, *, size=DETECTOR_SIZE, beam_energy=VOLTAGE_KV):
    """A square diffraction signal calibrated at *scale* per pixel in *units*."""
    import hyperspy.api as hs

    signal = hs.signals.Signal2D(np.zeros((size, size), np.float32))
    for axis in signal.axes_manager.signal_axes:
        axis.scale = scale
        axis.offset = -scale * size / 2.0
        axis.units = units
    signal.set_signal_type("electron_diffraction")
    # set_signal_type stamps units="px" on the axes, so re-assert the label —
    # that overwrite is itself one of the behaviours this file pins.
    for axis in signal.axes_manager.signal_axes:
        axis.units = units
    if beam_energy is not None:
        signal.metadata.set_item(
            "Acquisition_instrument.TEM.beam_energy", beam_energy)
    return signal


def _equivalent_signals():
    """The same detector, labelled every way a real file might label it."""
    per_pixel = INVERSE_ANGSTROM_PER_PIXEL
    return {
        "A^-1": _signal(per_pixel, "A^-1"),
        "nm^-1": _signal(per_pixel * 10.0, "nm^-1"),
        "mrad": _signal(per_pixel * 1e3 * WAVELENGTH, "mrad"),
        # No label at all: read as Å⁻¹, which is what the app did before it
        # tracked units, so old datasets keep indexing the way they did.
        "unlabelled": _signal(per_pixel, ""),
        # pyxem stamps "px" by default when typing a signal as electron
        # diffraction, so on a calibrated detector that label is left over.
        "pyxem-default-px": _signal(per_pixel, "px"),
    }


class TestParseUnit:
    @pytest.mark.parametrize("text", [
        "nm^-1", "nm-1", "1/nm", "nm⁻¹", "NM^-1", " nm^-1 ", "nm$^{-1}$",
    ])
    def test_per_nanometre_spellings(self, text):
        assert ru.parse_unit(text) == ru.PER_NANOMETRE

    @pytest.mark.parametrize("text", [
        "A^-1", "1/A", "Å⁻¹", "å^-1", r"$\AA^{-1}$", "angstrom^-1", "A-1",
    ])
    def test_per_angstrom_spellings(self, text):
        assert ru.parse_unit(text) == ru.PER_ANGSTROM

    def test_pixels_and_angle(self):
        assert ru.parse_unit("px") == ru.PIXELS
        assert ru.parse_unit("pixels") == ru.PIXELS
        assert ru.parse_unit("mrad") == ru.MILLIRADIAN

    def test_unspecified_is_not_pixels(self):
        # A blank label means nobody said, which is not the same claim as "this
        # axis counts pixels" — the two are read differently downstream.
        for text in ("", "   ", "<undefined>", None, "furlongs"):
            assert ru.parse_unit(text) is None

    def test_is_reciprocal(self):
        assert ru.is_reciprocal("nm^-1") and ru.is_reciprocal("Å⁻¹")
        assert not ru.is_reciprocal("mrad")
        assert not ru.is_reciprocal("px")


class TestFactors:
    def test_per_nanometre_is_a_factor_of_ten(self):
        assert ru.inverse_angstrom_factor("nm^-1") == pytest.approx(0.1)
        assert ru.inverse_angstrom_factor("A^-1") == 1.0

    def test_reciprocal_conversion_needs_no_beam_energy(self):
        # The thing that actually broke the ZrNb scan is a factor of ten. It
        # does not involve the electron wavelength, so it must not require one.
        assert ru.inverse_angstrom_factor("nm^-1", wavelength=None) == pytest.approx(0.1)

    def test_milliradian_needs_a_wavelength(self):
        assert ru.inverse_angstrom_factor("mrad") is None
        assert ru.inverse_angstrom_factor("mrad", wavelength=WAVELENGTH) \
            == pytest.approx(1.0 / (1e3 * WAVELENGTH))

    def test_pixels_have_no_factor(self):
        assert ru.inverse_angstrom_factor("px") is None

    def test_wavelength_matches_the_standard_relativistic_form(self):
        # 200 kV is 2.5079 pm — the textbook value, and diffsims'.
        assert ru.electron_wavelength_angstrom(200.0) == pytest.approx(0.025079, abs=1e-6)
        assert ru.electron_wavelength_angstrom(300.0) == pytest.approx(0.019687, abs=1e-6)


class TestSignalScale:
    def test_every_labelling_reports_the_same_pixel_size(self):
        for name, signal in _equivalent_signals().items():
            assert ru.inverse_angstrom_scale(signal) == \
                pytest.approx(INVERSE_ANGSTROM_PER_PIXEL, rel=1e-9), name

    def test_every_labelling_reports_the_same_extent(self):
        expected = INVERSE_ANGSTROM_PER_PIXEL * DETECTOR_SIZE / 2.0
        for name, signal in _equivalent_signals().items():
            assert ru.reciprocal_extent(signal) == pytest.approx(expected, rel=1e-9), name

    def test_milliradian_without_a_beam_energy_is_unresolvable(self):
        signal = _signal(0.2508, "mrad", beam_energy=None)
        assert ru.inverse_angstrom_scale(signal) is None
        assert ru.available_units(signal)[ru.MILLIRADIAN] == "needs a beam energy"

    def test_wavelength_prefers_a_recorded_value(self):
        signal = _signal(INVERSE_ANGSTROM_PER_PIXEL, "A^-1")
        signal.metadata.set_item("Acquisition_instrument.TEM.wavelength", 0.0197)
        assert ru.wavelength_angstrom(signal) == pytest.approx(0.0197)


class TestToggle:
    def test_round_trip_through_every_unit_preserves_the_calibration(self):
        signal = _signal(INVERSE_ANGSTROM_PER_PIXEL * 10.0, "nm^-1")
        for unit in ("A^-1", "mrad", "px", "nm^-1", "mrad", "A^-1"):
            assert ru.convert_signal_axes(signal, unit), unit
            assert ru.inverse_angstrom_scale(signal) == \
                pytest.approx(INVERSE_ANGSTROM_PER_PIXEL, rel=1e-9), unit

    def test_the_toggle_rescales_rather_than_relabels(self):
        signal = _signal(INVERSE_ANGSTROM_PER_PIXEL * 10.0, "nm^-1")
        ru.convert_signal_axes(signal, "A^-1")
        axis = signal.axes_manager.signal_axes[0]
        assert axis.units == "A^-1"
        assert float(axis.scale) == pytest.approx(INVERSE_ANGSTROM_PER_PIXEL)
        # The offset moves with the scale, so the marked origin stays put.
        assert float(axis.offset) == pytest.approx(
            -INVERSE_ANGSTROM_PER_PIXEL * DETECTOR_SIZE / 2.0)

    def test_pixels_keep_the_origin_and_stay_reversible(self):
        signal = _signal(INVERSE_ANGSTROM_PER_PIXEL * 10.0, "nm^-1")
        ru.convert_signal_axes(signal, "px")
        axis = signal.axes_manager.signal_axes[0]
        assert float(axis.scale) == 1.0
        assert float(axis.offset) == pytest.approx(-DETECTOR_SIZE / 2.0)
        # Nothing about the detector was forgotten, only how it is written.
        assert ru.inverse_angstrom_scale(signal) == \
            pytest.approx(INVERSE_ANGSTROM_PER_PIXEL, rel=1e-9)

    def test_an_off_centre_origin_survives_conversion(self):
        signal = _signal(0.1, "nm^-1")
        for axis in signal.axes_manager.signal_axes:
            axis.offset = -0.1 * 30.0          # direct beam at pixel 30, not 32
        ru.convert_signal_axes(signal, "A^-1")
        axis = signal.axes_manager.signal_axes[0]
        assert -float(axis.offset) / float(axis.scale) == pytest.approx(30.0)

    def test_normalize_converts_only_a_real_reciprocal_calibration(self):
        converted = _signal(0.1, "nm^-1")
        assert ru.normalize_to_canonical(converted) is True
        assert converted.axes_manager.signal_axes[0].units == "A^-1"

        already = _signal(0.01, "A^-1")
        assert ru.normalize_to_canonical(already) is False

        # A scan axis in nm is not a detector axis and must be left alone.
        import hyperspy.api as hs
        scan = hs.signals.Signal2D(np.zeros((8, 8), np.float32))
        for axis in scan.axes_manager.signal_axes:
            axis.scale, axis.units = 0.9, "nm"
        assert ru.normalize_to_canonical(scan) is False
        assert scan.axes_manager.signal_axes[0].scale == pytest.approx(0.9)


class TestReciprocalRadiusParity:
    """The number that sizes every simulated template library."""

    def test_identical_whatever_the_axes_are_labelled(self):
        from spyde.actions._common import reciprocal_radius

        expected = INVERSE_ANGSTROM_PER_PIXEL * DETECTOR_SIZE / 2.0
        for name, signal in _equivalent_signals().items():
            assert reciprocal_radius(signal) == pytest.approx(expected, rel=1e-9), name

    def test_says_so_instead_of_guessing_when_uncalibrated(self):
        from spyde.actions._common import reciprocal_radius

        signal = _signal(0.2508, "mrad", beam_energy=None)
        with pytest.raises(ValueError, match="no reciprocal calibration"):
            reciprocal_radius(signal)


class TestPolarGridParity:
    """The radial axis simulated template rings are placed on."""

    def test_radial_range_identical_whatever_the_axes_are_labelled(self):
        from spyde.actions.orientation_compute import polar_radial_range

        ranges = {}
        for name, signal in _equivalent_signals().items():
            _sl, _f, _fs, radial_range = signal.calibration.get_slices2d(100, 360)
            ranges[name] = polar_radial_range(signal, radial_range)
        reference = ranges["A^-1"]
        for name, value in ranges.items():
            assert value == pytest.approx(reference, rel=1e-6), name


class TestMeasuredVectorParity:
    """Vectors are stored in the detector's units and fitted in Å⁻¹."""

    def test_the_peaks_adapter_converts_to_inverse_angstrom(self):
        """The matcher works entirely in Å⁻¹ and the vectors are stored in the
        detector's own units, so the adapter that presents one to the other is
        where the two meet."""
        from spyde.actions.vector_orientation_quantem import PeaksAdapter

        class _Vectors:
            n_time = 0
            nav_shape = (1, 2)

            def at(self, row, column):
                rows = np.zeros((1, 6), np.float64)
                rows[:, 2] = 3.9 if column == 0 else -3.9   # kx in nm⁻¹
                return rows

        peaks = PeaksAdapter(_Vectors(), inverse_angstrom_factor=0.1)
        assert peaks[0, 0].array[0, 0] == pytest.approx(0.39)
        assert peaks[0, 1].array[0, 0] == pytest.approx(-0.39)

    def test_detector_pixels_places_inverse_angstrom_spots_correctly(self):
        from spyde.actions.vector_overlay import DetectorPixels

        class _Axis:
            def __init__(self, scale, offset, size, units):
                self.scale, self.offset = scale, offset
                self.size, self.units = size, units

        # The same detector in both units: a spot at 0.39 Å⁻¹ must land on the
        # same image pixel either way, or the matched-template overlay is drawn
        # somewhere the data is not.
        in_angstrom = DetectorPixels.from_axes(
            [_Axis(0.01, -0.32, 64, "A^-1"), _Axis(0.01, -0.32, 64, "A^-1")])
        in_nanometre = DetectorPixels.from_axes(
            [_Axis(0.1, -3.2, 64, "nm^-1"), _Axis(0.1, -3.2, 64, "nm^-1")])
        assert in_nanometre.inverse_angstrom_factor == pytest.approx(0.1)

        spots = np.array([[0.39, 0.0], [0.0, -0.2]])
        assert np.allclose(in_angstrom.inverse_angstrom_to_pixels(spots),
                           in_nanometre.inverse_angstrom_to_pixels(spots))


class TestCifFamilyParity:
    """Crystallography is done in 1/Å; the detector may not be."""

    @staticmethod
    def _aluminium():
        from diffpy.structure import Atom, Lattice, Structure
        from orix.crystal_map import Phase

        structure = Structure(atoms=[Atom("Al", [0, 0, 0])],
                              lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))
        return Phase(name="Al", space_group=225, structure=structure)

    def test_families_are_expressed_in_the_detector_units(self):
        from spyde.actions.strain_mapping import cif_g_families, cif_g_families_for

        class _Axis:
            def __init__(self, units):
                self.units = units

        phase = self._aluminium()
        in_angstrom = cif_g_families(phase)
        assert np.allclose(cif_g_families_for(phase, [_Axis("A^-1")]), in_angstrom)
        assert np.allclose(cif_g_families_for(phase, [_Axis("nm^-1")]),
                           in_angstrom * 10.0)
        # No axis records at all: taken at face value, as documented.
        assert np.allclose(cif_g_families_for(phase, ()), in_angstrom)

    def test_a_nanometre_reference_snaps_instead_of_coming_back_empty(self):
        # The regression this exists for: 1/Å families against nm⁻¹ magnitudes
        # fall outside snap_reference_to_cif's tolerance, so EVERY reflection is
        # rejected, the reference is empty, and the strain map silently never
        # appears.
        from spyde.actions.strain_mapping import (
            cif_g_families, cif_g_families_for, snap_reference_to_cif,
        )

        phase = self._aluminium()
        first = float(cif_g_families(phase)[0])           # Å⁻¹
        directions = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        measured_nm = directions * (first * 10.0)         # the same spots, nm⁻¹

        naive = snap_reference_to_cif(measured_nm, cif_g_families(phase))
        assert len(naive) == 0, "the un-converted comparison should match nothing"

        snapped = snap_reference_to_cif(measured_nm,
                                        cif_g_families_for(phase, [type("A", (), {"units": "nm^-1"})()]))
        assert len(snapped) == 4
        assert np.linalg.norm(snapped[0]) == pytest.approx(first * 10.0, rel=1e-6)

    def test_a_millradian_detector_says_so_rather_than_answering_wrongly(self):
        # An axis record carries no beam energy, so mrad cannot be turned into
        # a spacing here. Returning the 1/Å families unconverted would match
        # nothing and produce no map — the exact silent failure this replaces.
        from spyde.actions.strain_mapping import cif_g_families_for

        class _Axis:
            units = "mrad"

        with pytest.raises(ValueError, match="reciprocal unit"):
            cif_g_families_for(self._aluminium(), [_Axis()])
