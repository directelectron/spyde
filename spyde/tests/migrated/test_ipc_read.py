"""
Stdin message decoding (the Electron→backend line protocol).

`read_messages` reads the BINARY stdin stream and decodes each line as UTF-8
itself, rather than trusting `sys.stdin`'s text layer — whose encoding is the
platform default (cp1252 on Windows). A UTF-8 payload carrying a non-ASCII
character (e.g. the strain labels ``εxx``/``εyy``, the ``Å`` unit) was mojibake'd
on Windows: the bytes ``0xCE 0xB5`` (UTF-8 ``ε``) decoded to ``Îµ``, and
downstream string matching (tile_views' view-label lookup) silently dropped the
view. This guards the decode end-to-end without an Electron round-trip.
"""
from __future__ import annotations

import asyncio
import io
import json


class _BinStdin:
    """A stdin double exposing a ``.buffer`` with UTF-8-encoded bytes, mirroring
    the real ``sys.stdin`` shape the pump reads from."""

    def __init__(self, lines: list[str]) -> None:
        payload = "".join(line + "\n" for line in lines).encode("utf-8")
        self.buffer = io.BytesIO(payload)


def _collect(stdin) -> list[dict]:
    from spyde.backend import ipc

    async def _run() -> list[dict]:
        out: list[dict] = []
        async for msg in ipc.read_messages(asyncio.get_event_loop()):
            out.append(msg)
        return out

    orig = ipc.sys.stdin
    ipc.sys.stdin = stdin
    try:
        return asyncio.run(_run())
    finally:
        ipc.sys.stdin = orig


class TestReadMessages:
    def test_utf8_non_ascii_labels_decode_intact(self):
        # ε is U+03B5 (one codepoint). A cp1252 decode of its UTF-8 bytes would
        # yield "Îµxx" (codepoints 206, 181, …) — the Windows mojibake bug.
        msgs = _collect(_BinStdin([
            json.dumps({"type": "action", "action": "tile_views",
                        "labels": ["εxx", "εyy", "Å⁻¹"]}),
        ]))
        assert len(msgs) == 1
        labels = msgs[0]["labels"]
        assert labels == ["εxx", "εyy", "Å⁻¹"]
        assert [ord(labels[0][0])] == [0x03B5]     # a single ε, not 0xCE 0xB5

    def test_blank_and_non_json_lines_skipped(self):
        msgs = _collect(_BinStdin([
            "", "  ", "not json at all",
            json.dumps({"type": "quit"}),
        ]))
        assert msgs == [{"type": "quit"}]

    def test_text_stdin_without_buffer_still_works(self):
        # Some harnesses replace sys.stdin with a plain text stream (no .buffer);
        # the pump must fall back to reading it as text.
        text = io.StringIO(json.dumps({"type": "action", "action": "x"}) + "\n")
        # StringIO has no .buffer attribute → the text fallback path.
        assert not hasattr(text, "buffer")
        msgs = _collect(text)
        assert msgs == [{"type": "action", "action": "x"}]
