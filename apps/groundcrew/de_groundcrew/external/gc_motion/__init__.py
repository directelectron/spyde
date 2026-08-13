"""
gc_motion — Ground Crew's motion-correction compute, vendored VERBATIM.

Copied unchanged from ``CSSFrancis/de_ground_crew`` at
``e9e21de25bc10fcdc6135f34bd8ca7e736c37c6f`` (origin/main, 2026-08-11). See
MANIFEST.md for the file list and the one permitted edit.

## Why vendored rather than ported

The compute layer over there is **already Qt-free** — only the ``*_worker.py``
QThread wrappers import PySide6. So there is nothing to rewrite, and rewriting
would be actively harmful:

* This code is **numerically delicate and externally validated**. v3 was
  co-designed against cryoSPARC as an oracle and RELION 5.0.1's source, and the
  CTF port is held to the CTFFIND 5.0.2 binary to ±0.0005 µm. A transcription
  is a chance to silently break that, and no test on this side would catch it.
* Ground Crew's own QThread-removal spec explicitly **rejects** a parallel
  package: *"duplicates code, invites drift, larger diff"*. A hand-port here is
  exactly that parallel package.
* Upstream fixes become a re-copy, not a re-implementation.

A first attempt DID hand-port this, from a snapshot that had been deleted from
`main` — and reproduced an already-fixed bug while missing the entire v3
aligner. That is the failure mode this arrangement exists to prevent.

## The one edit

``_motion_correction_v3.py`` does ``from _motion_correction_v2 import …`` — a
top-level module import, which cannot resolve inside a package. That single
line is rewritten to an absolute import of the vendored sibling. Nothing else
is touched; `tests/test_vendor_pristine.py` proves it.

## What is NOT here

The ``*_worker.py`` QThread wrappers. Those are the Qt layer this app exists to
replace — `de_groundcrew.motion.driver` is their Qt-free counterpart, and is
the only thing that should call into this package.
"""
