"""
tutorial_data.py — TutorialDataMixin: curated, ALWAYS-AVAILABLE small in-memory
tutorial datasets (Phase 1 of the docs/walkthroughs overhaul).

Unlike ``TestHarnessMixin`` (spyde/backend/_session_testharness.py), which is
gated behind ``_TEST_ACTIONS_ENABLED`` and disappears in a packaged build,
these loaders are user-reachable in EVERY build — a "Tutorial Data" menu lets
a real user (not just Playwright) load a small, fast, no-download dataset that
demonstrates one workflow end-to-end. Phase 2+ guided walkthroughs drive these
same loaders via the ``tutorial_load`` action.

Every loader here follows the same shape as the test-harness loaders it's
modelled on: ``ensure_heavy_imports()`` FIRST (before ``set_signal_type`` —
racing the startup prewarm's pyxem import poisons the cast), generate the
signal in-memory (no file, no pooch download), ``set_signal_type(...)``, then
``self._add_signal(sig, source_path="tutorial_<name>")``. Keep every dataset
SMALL: a tutorial must load in a couple of seconds, not download or
materialise gigabytes (see ``simulated_strain``'s huge default, deliberately
downsized below).

The mixin only USES ``self.<attr>``/``self.<method>`` (``self._add_signal``,
``self._await_dask``) provided by the final Session.
"""
from __future__ import annotations

import logging

import numpy as np
import hyperspy.api as hs

log = logging.getLogger(__name__)


def _build_movie_frames(n: int, n_frames: int):
    """Build the synthetic in-situ movie frame stack shared by the test-only
    ``_load_test_data_movie`` (large, 2048² default, for the GPU/tile specs)
    and the tutorial-sized ``tutorial_movie`` (small, fast) loaders — same
    asymmetric content (corner blocks, per-frame index band, fine
    checkerboard patch) so both exercise the same coordinate/parity properties,
    just at different resolutions. Returns a ``(n_frames, n, n)`` uint16 array.
    """
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    base = (xx / n) * 250.0 + (yy / n) * 250.0
    base[(yy < n // 6) & (xx < n // 6)] = 1000.0          # TOP-LEFT block
    base[(yy > 5 * n // 6) & (xx > 5 * n // 6)] = 800.0   # BOTTOM-RIGHT block
    # Fine checkerboard patch (2-px pitch) in the centre quarter.
    cb = slice(3 * n // 8, 5 * n // 8)
    checker = (((xx[cb, cb] // 2).astype(np.int32)
                + (yy[cb, cb] // 2).astype(np.int32)) % 2) * 400.0 + 200.0
    frames = np.empty((n_frames, n, n), dtype=np.uint16)
    for t in range(n_frames):
        f = base.copy()
        f[cb, cb] = checker
        x0 = (t + 1) * n // (n_frames + 2)
        f[:, x0:x0 + max(1, n // 32)] = 900.0             # frame-index band
        frames[t] = f.astype(np.uint16)
    return frames


class TutorialDataMixin:
    """Curated tutorial-dataset loaders — always reachable (NOT test-gated)."""

    def tutorial_navigation(self) -> None:
        """Tutorial: Navigation & Virtual Imaging. 10x10 nav x 50x50 signal,
        a disk + ring whose intensity/position vary per probe position
        (``pyxem.data.dummy_data.generate_4d_data``) — clear real-space
        contrast so a virtual image / navigator crosshair drag is obviously
        meaningful. NO download, in-memory, eager (loads in well under a
        second)."""
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()   # don't race the startup prewarm's pyxem import
        from pyxem.data.dummy_data.make_diffraction_test_data import generate_4d_data
        probe = 10
        # Disk position drifts diagonally across the probe grid, and disk/ring
        # intensity ramps with probe position, so BOTH a disk-VI and a
        # ring-VI show clear per-pixel real-space contrast (not a uniform
        # image) when integrated over the whole scan.
        px, py = np.mgrid[0:probe, 0:probe]
        disk_x = 20 + (px / (probe - 1)) * 10.0
        disk_y = 20 + (py / (probe - 1)) * 10.0
        disk_I = 15 + (px + py) * 1.5
        ring_I = 4 + (px * py) * 0.1
        s = generate_4d_data(
            probe_size_x=probe, probe_size_y=probe,
            image_size_x=50, image_size_y=50,
            disk_x=disk_x, disk_y=disk_y, disk_r=5, disk_I=disk_I,
            ring_x=25, ring_y=25, ring_r=20, ring_I=ring_I,
            add_noise=True, show_progressbar=False,
        )
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on tutorial_navigation failed: %s", e)
        s.metadata.set_item("General.title", "Tutorial: Navigation & VI")
        self._add_signal(s, source_path="tutorial_navigation")

    def tutorial_find_vectors(self) -> None:
        """Tutorial: Find Diffraction Vectors. 6x6 nav x 128x128 signal
        (``pyxem.data.si_grains``) — a crisp real reciprocal lattice (not a
        featureless disk), so spot-finding actually lands on real peaks.
        Mirrors ``TestHarnessMixin._load_test_data_si_grains``'s beam-centring
        offset so the direct beam sits mid-detector."""
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()   # don't race the startup prewarm's pyxem import
        import pyxem.data as pxd
        s = pxd.si_grains()
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on tutorial_find_vectors failed: %s", e)
        # Centre the beam in pixel space (bundled data carries offset=0, i.e.
        # k=0 at pixel 0); a centred offset puts the direct beam mid-detector
        # like a real scan (see _load_test_data_si_grains for the same fix).
        for ax in s.axes_manager.signal_axes:
            ax.offset = -(ax.size / 2.0) * float(ax.scale)
        s.metadata.set_item("General.title", "Tutorial: Find Vectors (Si grains)")
        self._add_signal(s, source_path="tutorial_find_vectors")

    def tutorial_orientation(self) -> None:
        """Tutorial: Orientation Mapping. Same bundled Si-grains scan as
        ``tutorial_find_vectors`` (single-phase, crisp lattice) — the simplest
        base case for the OM workflow: find vectors, match a template, view
        the IPF. Kept as its own loader/action (rather than reusing
        tutorial_find_vectors) so a guide can address it by name and stamp a
        distinct title."""
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        import pyxem.data as pxd
        s = pxd.si_grains()
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on tutorial_orientation failed: %s", e)
        for ax in s.axes_manager.signal_axes:
            ax.offset = -(ax.size / 2.0) * float(ax.scale)
        s.metadata.set_item("General.title", "Tutorial: Orientation Mapping")
        self._add_signal(s, source_path="tutorial_orientation")

    def tutorial_multiphase(self) -> None:
        """Tutorial: multi-phase Orientation Mapping. 20x20 nav x 128x128
        signal, BCC + FCC iron grains (``pyxem.data.fe_multi_phase_grains``) —
        demonstrates phase discrimination, not just single-phase indexing."""
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        import pyxem.data as pxd
        s = pxd.fe_multi_phase_grains(num_grains=2, seed=2, size=20, recip_pixels=128)
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on tutorial_multiphase failed: %s", e)
        for ax in s.axes_manager.signal_axes:
            ax.offset = -(ax.size / 2.0) * float(ax.scale)
        s.metadata.set_item("General.title", "Tutorial: Multi-Phase OM (BCC+FCC Fe)")
        self._add_signal(s, source_path="tutorial_multiphase")

    def tutorial_strain(self) -> None:
        """Tutorial: Strain Mapping. ``pyxem.data.simulated_strain``,
        DOWNSIZED from its huge default (32x32 nav x 512x512 signal x 1e5
        electrons/px, ~2 GB) to 16x16 nav x 128x128 signal x 1e3 electrons/px
        — a snappy in-memory dataset with a genuine strained region (a soft
        elliptical strain field applied via a simulated diffraction pattern +
        transformation matrix) so the strain-map workflow has real signal to
        recover."""
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()   # don't race the startup prewarm's pyxem import
        import pyxem.data as pxd
        s = pxd.simulated_strain(
            navigation_shape=(16, 16), signal_shape=(128, 128),
            disk_radius=8, num_electrons=1e3,
        )
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on tutorial_strain failed: %s", e)
        s.metadata.set_item("General.title", "Tutorial: Strain Mapping")
        self._add_signal(s, source_path="tutorial_strain")

    def tutorial_spectroscopy(self) -> None:
        """Tutorial: 1-D spectroscopy workflow. ``hs.data.two_gaussians`` — a
        32x32-probe Signal1D (1024-channel EELS/EDS-like spectrum) built from
        two per-pixel-varying Gaussian components + Poissonian noise, the
        best bundled 1-D-spectroscopy proxy (peak fitting, line profiles,
        per-pixel component maps). Already the right signal_type (Signal1D);
        no cast needed."""
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        s = hs.data.two_gaussians(add_noise=True)
        s.metadata.set_item("General.title", "Tutorial: Spectroscopy (1D)")
        self._add_signal(s, source_path="tutorial_spectroscopy")

    def tutorial_movie(self) -> None:
        """Tutorial: in-situ movie playback. A DOWNSIZED variant of
        ``TestHarnessMixin._load_test_data_movie``'s synthetic movie — 512²
        frames (vs. the test loader's 2048²) x 6 frames, still chunked 1
        frame/chunk lazy like a real .mrc in-situ movie, still carrying the
        same asymmetric content (corner blocks / frame-index band /
        checkerboard patch — see ``_build_movie_frames``, shared with the
        test loader) so Play/Fast-Forward + frame scrubbing are pixel-visibly
        correct without needing a 2048² tile-mode dataset for a tutorial."""
        import dask.array as da
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()   # don't race the startup prewarm's pyxem import
        n, n_frames = 512, 6
        frames = _build_movie_frames(n, n_frames)
        stack = da.from_array(frames, chunks=(1, n, n))   # 1 frame/chunk
        s = hs.signals.Signal2D(stack).as_lazy()
        for ax in s.axes_manager.signal_axes:
            ax.scale = 0.5
            ax.units = "nm"
        tax = s.axes_manager.navigation_axes[0]
        tax.name, tax.units, tax.scale = "time", "s", 0.05
        s.set_signal_type("insitu")   # gates the Play/Fast Forward toolbar buttons
        s.metadata.set_item("General.title", "Tutorial: In-Situ Movie")
        self._add_signal(s, source_path="tutorial_movie")


# name -> bound-method lookup used by the (ungated) `tutorial_load` action in
# _session_actions.py. Keys are the same names used as tutorial_<name> testids
# in the renderer's Tutorial Data menu.
TUTORIAL_LOADERS = {
    "navigation": TutorialDataMixin.tutorial_navigation,
    "find_vectors": TutorialDataMixin.tutorial_find_vectors,
    "orientation": TutorialDataMixin.tutorial_orientation,
    "multiphase": TutorialDataMixin.tutorial_multiphase,
    "strain": TutorialDataMixin.tutorial_strain,
    "spectroscopy": TutorialDataMixin.tutorial_spectroscopy,
    "movie": TutorialDataMixin.tutorial_movie,
}
