"""The CSB reader's registration and header parsing.

`csb_format.py` already claims "test_csb_reader.py asserts the two agree" about
its `_SPEC` and `specifications.yaml`; this is that file.

Scope is deliberate. The event decoder needs a real multi-gigabyte stream to
exercise (`electron/tests/csb_movie.spec.ts` drives it, and skips itself when
that file is absent — so it never runs in CI), but everything ABOVE the events
can be covered from a synthesised file: a valid CSB with an empty payload is
just a 108-byte header plus a zero-filled block table.

That covers the parts most likely to break silently rather than loudly:
  * the plugin not registering, so `.csb` quietly will not open at all
  * `.csb` missing from SUPPORTED_EXTS, so the Open dialog will not offer it
    (a real bug on this branch — "let Open actually pick a .csb")
  * `_SPEC` drifting from the yaml that becomes authoritative the day this
    moves into rosettasciio
  * header field offsets and the block-grid arithmetic
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from spyde.external.rosettasciio import csb_format
from spyde.external.rsciio_csb._core import CSBFile, CSB_MAGIC


# ── synthesising a CSB ────────────────────────────────────────────────────────

def _csb_bytes(*, width=64, height=48, frames=3, block_w=16, block_h=16,
               magic=CSB_MAGIC, us_per_frame=400.0, kv=200, order=0):
    """A structurally valid CSB carrying zero events.

    data_offset == lengths_offset, so the payload region is empty and the
    block table (all zeros) sits immediately after the header — which is what
    the reader's own "table sums to N events, payload holds M words" check
    demands. Everything up to and including the table is therefore real.
    """
    hdr = bytearray(108)
    struct.pack_into("<H", hdr, 0, magic)
    struct.pack_into("<H", hdr, 2, 1)             # file_version
    struct.pack_into("<H", hdr, 4, width)
    struct.pack_into("<H", hdr, 6, height)
    struct.pack_into("<I", hdr, 8, frames)
    struct.pack_into("<f", hdr, 12, 0.025)        # ang_per_pix
    struct.pack_into("<f", hdr, 16, us_per_frame)
    struct.pack_into("<H", hdr, 20, block_w)
    struct.pack_into("<H", hdr, 22, block_h)
    struct.pack_into("<Q", hdr, 24, 108)          # csb_data_offset
    struct.pack_into("<Q", hdr, 32, 108)          # csb_lengths_offset
    struct.pack_into("<H", hdr, 40, order)
    struct.pack_into("<H", hdr, 42, 4242)         # camera_sn
    struct.pack_into("<H", hdr, 62, kv)           # microscope_kv

    import math
    n_table = 0 if min(width, height, frames, block_w, block_h) < 1 else (
        frames * (math.ceil(width / block_w) * math.ceil(height / block_h)))
    tail = b"\x00" * (n_table * 2)
    # A degenerate dimension leaves an empty table and a 108-byte file, which
    # trips the length guard BEFORE the dimension validation we want to reach.
    # Pad past it so those cases fail for the reason under test.
    if len(hdr) + len(tail) < 110:
        tail += b"\x00" * (110 - len(hdr) - len(tail))
    return bytes(hdr) + tail


@pytest.fixture
def csb_path(tmp_path):
    p = tmp_path / "synthetic.csb"
    p.write_bytes(_csb_bytes())
    return str(p)


# ── 1. registration ───────────────────────────────────────────────────────────

class TestRegistration:
    def test_spec_matches_the_yaml(self):
        """The yaml is authoritative the day this moves upstream; _SPEC is a
        hand copy, so they must not drift."""
        import yaml
        from pathlib import Path
        import spyde.external.rsciio_csb as pkg

        spec_file = Path(pkg.__file__).parent / "specifications.yaml"
        on_disk = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
        # Guard the loop below against passing vacuously on an empty/missing yaml.
        assert set(on_disk) >= {"name", "file_extensions", "writes"}, on_disk
        for key, value in on_disk.items():
            assert csb_format._SPEC[key] == value, (
                f"_SPEC[{key!r}] is {csb_format._SPEC.get(key)!r} but the yaml "
                f"says {value!r}")

    def test_apply_registers_and_is_idempotent(self):
        rsciio = pytest.importorskip("rsciio")
        assert csb_format.apply() is True
        names = [p.get("name") for p in rsciio.IO_PLUGINS if isinstance(p, dict)]
        assert names.count("CSB") == 1, "CSB registered zero or twice"
        # Calling again must not append a duplicate.
        assert csb_format.apply() is True
        names = [p.get("name") for p in rsciio.IO_PLUGINS if isinstance(p, dict)]
        assert names.count("CSB") == 1

    def test_csb_is_an_openable_extension(self):
        """Without this the reader works and the Open dialog still refuses the
        file — which is exactly how it shipped broken once."""
        from spyde.backend._session_files import SUPPORTED_EXTS
        assert ".csb" in SUPPORTED_EXTS


# ── 1b. the toolbar gate that reveals the CSB-only actions ────────────────────

class TestOriginalMetadataGate:
    """``requires_original_metadata:`` — the gate "To Frames" is hidden behind.

    Its ``except`` branch is the interesting one. It logs and returns False,
    which is the safe answer; but the module had no logger, so a signal whose
    ``has_item`` raised turned a hidden button into a ``NameError`` raised
    inside ``get_toolbar_actions_for_plot`` — i.e. the whole toolbar failing to
    build, for a signal that merely was not a CSB. Cheap to assert, and the
    kind of thing only an odd signal in the wild would ever reach.
    """

    def test_a_matching_path_shows_the_action(self):
        from spyde.drawing.toolbars.plot_control_toolbar import (
            _has_original_metadata)
        from hyperspy.misc.utils import DictionaryTreeBrowser as DTB

        class Sig:
            original_metadata = DTB({"csb": {"us_per_frame": 390.0}})

        assert _has_original_metadata(Sig(), "csb.us_per_frame") is True
        assert _has_original_metadata(Sig(), "csb.nope") is False

    def test_no_gate_means_always_visible(self):
        from spyde.drawing.toolbars.plot_control_toolbar import (
            _has_original_metadata)
        assert _has_original_metadata(object(), None) is True
        assert _has_original_metadata(object(), "") is True

    def test_a_signal_without_original_metadata_is_refused(self):
        from spyde.drawing.toolbars.plot_control_toolbar import (
            _has_original_metadata)
        assert _has_original_metadata(object(), "csb.us_per_frame") is False

    def test_a_raising_lookup_is_refused_not_propagated(self):
        from spyde.drawing.toolbars.plot_control_toolbar import (
            _has_original_metadata)

        class Boom:
            class _OM:
                def has_item(self, dotted):
                    raise RuntimeError("no such tree")
            original_metadata = _OM()

        assert _has_original_metadata(Boom(), "csb.us_per_frame") is False


# ── 2. header parsing + geometry ──────────────────────────────────────────────

class TestHeader:
    def test_fields_come_back_off_the_documented_offsets(self, csb_path):
        f = CSBFile(csb_path)
        assert f.file_specifier == CSB_MAGIC
        assert (f.frame_width, f.frame_height) == (64, 48)
        assert f.frame_count == 3
        assert f.csb_block_width == 16 and f.csb_block_height == 16
        assert f.microsec_per_frame == pytest.approx(400.0)
        assert f.microscope_kv == 200
        assert f.camera_sn == 4242

    def test_block_grid_and_padding(self, csb_path):
        f = CSBFile(csb_path)
        assert (f.blocks_per_width, f.blocks_per_height) == (4, 3)
        assert f.blocks_per_frame == 12
        assert f.tile_size == 256
        # 64x48 divides evenly by 16, so no padding here.
        assert (f.padded_width, f.padded_height) == (64, 48)

    def test_a_partial_edge_block_keeps_its_stride(self, tmp_path):
        """Edge blocks are partial but keep the full stride — the accumulator
        is allocated padded and cropped at readout."""
        p = tmp_path / "odd.csb"
        p.write_bytes(_csb_bytes(width=70, height=50, block_w=16, block_h=16))
        f = CSBFile(str(p))
        assert (f.blocks_per_width, f.blocks_per_height) == (5, 4)
        assert (f.padded_width, f.padded_height) == (80, 64)

    def test_an_empty_payload_reads_as_zero_events(self, csb_path):
        f = CSBFile(csb_path)
        assert f.n_events == 0
        assert f.counts.shape == (3 * 12,)
        assert f.counts.sum() == 0
        # starts/ends stay consistent even with nothing in them.
        assert f.starts.shape == f.counts.shape
        assert int(f.starts.max()) == 0


# ── 3. rejecting what is not a CSB ────────────────────────────────────────────

class TestRejects:
    def test_a_short_file(self, tmp_path):
        p = tmp_path / "stub.csb"
        p.write_bytes(b"\x00" * 32)
        with pytest.raises(ValueError, match="too short"):
            CSBFile(str(p))

    def test_a_wrong_magic(self, tmp_path):
        """A .mrc renamed to .csb must say so, not crash somewhere downstream."""
        p = tmp_path / "notcsb.csb"
        p.write_bytes(_csb_bytes(magic=CSB_MAGIC + 1))
        with pytest.raises(ValueError, match="not a CSB file"):
            CSBFile(str(p))

    @pytest.mark.parametrize("kw", [
        {"width": 0}, {"height": 0}, {"frames": 0},
        {"block_w": 0}, {"block_h": 0},
    ])
    def test_a_zero_dimension(self, tmp_path, kw):
        p = tmp_path / "bad.csb"
        p.write_bytes(_csb_bytes(**kw))
        with pytest.raises(ValueError, match="invalid"):
            CSBFile(str(p))

    def test_a_truncated_block_table(self, tmp_path):
        p = tmp_path / "cut.csb"
        raw = _csb_bytes()
        p.write_bytes(raw[:-10])            # lose the tail of the table
        with pytest.raises(ValueError, match="block table is incomplete"):
            CSBFile(str(p))
