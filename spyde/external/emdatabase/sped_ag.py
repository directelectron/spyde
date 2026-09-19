"""
Offer the CALIBRATED SPED-Ag scan as an em-database dataset.

WHAT
    Adds a ``SPEDAgCalibrated`` entry to ``em_database.data`` at runtime,
    pointing at Zenodo record 21790591 ("Pyxem 4D STEM Demo Data", v10). The
    Examples menu lists whatever that namespace holds, so the dataset appears
    beside the others with no change to SpyDE's loading path.

WHY
    ``em_database.data.SPEDAg`` resolves to record 15490547, whose copy of the
    scan carries EXACTLY HALF the true reciprocal calibration — 0.013364
    against 0.026728 Å⁻¹ per pixel. Nothing raises: every measured vector
    simply reaches a template matcher at half its length, Ag {111} reads as
    0.21 Å⁻¹ instead of 0.42, and no crystal can be fitted to it. Measured on
    the vector-orientation live preview, the difference is a residual of
    0.0191 Å⁻¹ with 41 peaks explained against 0.2159 with 7.

    em-database 0.4.0 migrated several datasets to a calibrated record but not
    this one; its ``SPEDAg.yaml`` still pins 15490547 and the old md5.
    ``pyxem.data.sped_ag()`` is stale the same way — ``pyxem/data/_registry.py``
    pins the same record, commented "version 0.9.0".

    Added ALONGSIDE rather than rewritten over ``SPEDAg``, so nothing that
    already refers to the old entry changes meaning under it.

WHEN TO REMOVE
    When em-database repoints ``SPEDAg`` at 21790591 — at which point this
    entry becomes a duplicate of it. That is a two-line change to
    ``em_database/datasets/SPEDAg.yaml``:

        source: https://zenodo.org/records/21790591/files
        checksum: md5:964e638c1476e0337f0d474f2e691748

    The checksum is verified: downloaded, hashed, loaded, and the reciprocal
    scale read back as 0.02672830388733737 Å⁻¹/px.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_applied = False

#: The em-database dataset record, in the YAML schema its own datasets are
#: declared in — so this block is what moves upstream verbatim. Public because
#: the test harness reads the same URL and checksum from here rather than
#: keeping a second copy to drift from.
NAME = "SPEDAgCalibrated"
DATASET = {
    "description": (
        "A 4D-STEM scan of polycrystalline Ag including twins and grain "
        "boundaries, 208 x 64 probe positions of 112 x 112 diffraction "
        "patterns, acquired with precession. Reciprocal space is calibrated at "
        "0.0267283 1/Angstrom per pixel, centred on the direct beam — the same "
        "scan as SPEDAg, from the record that carries the corrected "
        "calibration rather than one at half scale."
    ),
    "source": "https://zenodo.org/records/21790591/files",
    "checksum": "md5:964e638c1476e0337f0d474f2e691748",
    "file": "SPED-Ag.zspy",
    "data_size": "214.0 MB",
    "detector_manufacturer": "Quantum Detectors",
    "detector": "Merlin",
    "microscope_vendor": "JEOL",
    "voltage": "200 kV",
    "license": "CC-BY-4.0",
    "technique": "4D-STEM",
    "tags": ["Orientation Mapping", "Grain Boundaries", "Twins", "Precession"],
}


def apply() -> bool:
    """Add the dataset to ``em_database.data``. Idempotent and defensive."""
    global _applied
    if _applied:
        return True
    try:
        import em_database
        import em_database.data as data
        from em_database.downloadable_dataset import DownloadableDataset
    except Exception as e:
        # em-database is a marked dependency (it needs Python >= 3.12), so its
        # absence is expected rather than an error.
        log.debug("em-database not importable, not adding %s: %s", NAME, e)
        return False

    if getattr(data, NAME, None) is not None:
        _applied = True
        return True
    try:
        def _init(self):
            DownloadableDataset.__init__(self, **DATASET)

        dataset = type(NAME, (DownloadableDataset,),
                       {"__init__": _init, "__doc__": DATASET["description"]})
        setattr(data, NAME, dataset)
        names = getattr(data, "__all__", None)
        if isinstance(names, list) and NAME not in names:
            names.append(NAME)
        # Instantiating is what would fail if the upstream signature moved, so
        # it is worth doing here rather than at first use from the menu.
        dataset()
    except Exception as e:
        log.warning("could not add the calibrated SPED-Ag dataset: %s", e)
        return False
    _applied = True
    log.debug("added em-database dataset %s (record 21790591)", NAME)
    return True
