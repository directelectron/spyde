"""
test_movie_model.py — the MovieSpec data model + the ``.spyde-report`` movie-cell
round-trip.

A movie cell mirrors a figure cell in the container: its recipe serializes to
``movies/<id>.yaml`` (that ``movies/`` sibling — vs ``figures/`` — is what marks it
a movie on reload), with a baked poster at ``assets/<id>.png`` and an invisible
``<!-- spyde:movie <id> -->`` marker in ``report.md``. SCHEMA_VERSION stays 1, so an
older report with no movie cells loads unchanged.
"""
from __future__ import annotations

import zipfile

from spyde.actions.report.model import (
    Cell, MovieSpec, ReportDoc, SignalRef, SCHEMA_VERSION,
    parse_report_md, read_report, serialize_report_md, write_report,
)


def _full_movie_spec() -> MovieSpec:
    return MovieSpec(
        source=SignalRef(file_path="x.hspy", tree_uid="t1", tree_node="movie",
                         title="movie", shape=[6, 2048, 2048]),
        params={"fps": 12, "downsample": 2, "stride": 1, "cmap": "viridis",
                "clim": [0.0, 100.0], "timestamp": True, "scalebar": False,
                "t_start": 0, "t_end": 5},
        annotations=[
            {"kind": "text", "time_range": [0.0, 2.0], "text": "Label", "xy": [10, 20]},
            {"kind": "rect", "time_range": [1.0, 3.0], "xy": [5, 5], "wh": [50, 50],
             "color": "#ff0000"}],
        text_overlays=[{"label": "T", "fmt": "{label} = {value:.1f} C",
                        "xy": [10, 40], "source": {"tree_uid": "temp"}}],
        freezes=[{"t": 3, "hold_s": 1.5}],
        crop=[100, 100, 1900, 1900],
        out_size=[900, 900])


class TestMovieSpecYaml:
    def test_to_from_yaml_roundtrips(self):
        ms = _full_movie_spec()
        ms2 = MovieSpec.from_yaml(ms.to_yaml())
        assert ms2.params == ms.params
        assert ms2.annotations == ms.annotations
        assert ms2.text_overlays == ms.text_overlays
        assert ms2.freezes == ms.freezes
        assert ms2.crop == ms.crop
        assert ms2.out_size == ms.out_size
        assert ms2.source is not None
        assert ms2.source.tree_uid == "t1"
        assert ms2.source.shape == [6, 2048, 2048]

    def test_empty_spec_tolerant(self):
        ms = MovieSpec.from_yaml("")
        assert ms.source is None
        assert ms.params == {}
        assert ms.annotations == [] and ms.freezes == []
        assert ms.crop is None and ms.out_size is None

    def test_partial_dict_tolerant(self):
        # A hand-authored / older partial spec: only params, no source/crop/etc.
        ms = MovieSpec.from_dict({"params": {"fps": 8}})
        assert ms.params == {"fps": 8}
        assert ms.source is None and ms.crop is None


class TestReportRoundTrip:
    def _doc(self):
        doc = ReportDoc(title="Movie Test")
        doc.cells = [
            Cell(id="cmd1", cell_type="markdown", source="# Hello"),
            Cell(id="cmov1", cell_type="movie", caption="My movie",
                 movie=_full_movie_spec(), placeholder=False),
            Cell(id="cmov2", cell_type="movie", caption="",
                 movie=MovieSpec(), placeholder=True),
        ]
        return doc

    def test_zip_layout(self, tmp_path):
        doc = self._doc()
        poster = b"\x89PNG\r\n\x1a\n" + b"poster"
        p = str(tmp_path / "t.spyde-report")
        write_report(doc, p, {"cmov1": poster})
        with zipfile.ZipFile(p) as zf:
            names = set(zf.namelist())
        # The movie sibling yaml (both movies) + the poster for the non-placeholder.
        assert "movies/cmov1.yaml" in names
        assert "movies/cmov2.yaml" in names
        assert "assets/cmov1.png" in names
        # NO figures yaml for a movie cell.
        assert "figures/cmov1.yaml" not in names

    def test_full_roundtrip(self, tmp_path):
        doc = self._doc()
        poster = b"\x89PNG\r\n\x1a\n" + b"poster"
        p = str(tmp_path / "t.spyde-report")
        write_report(doc, p, {"cmov1": poster})
        doc2, assets2 = read_report(p)

        movies = [c for c in doc2.cells if c.cell_type == "movie"]
        assert len(movies) == 2
        m, ph = movies
        # The non-placeholder movie: caption, spec + poster all survive.
        assert m.id == "cmov1" and m.caption == "My movie" and not m.placeholder
        assert m.movie is not None
        assert m.movie.params["fps"] == 12 and m.movie.params["cmap"] == "viridis"
        assert m.movie.crop == [100, 100, 1900, 1900]
        assert m.movie.out_size == [900, 900]
        assert len(m.movie.annotations) == 2
        assert m.movie.freezes == [{"t": 3, "hold_s": 1.5}]
        assert m.movie.source is not None and m.movie.source.tree_uid == "t1"
        assert assets2.get("cmov1") == poster
        # The placeholder movie: still a placeholder movie, empty spec, no poster.
        assert ph.id == "cmov2" and ph.placeholder
        assert ph.movie is not None and ph.movie.source is None
        assert "cmov2" not in assets2

    def test_marker_is_invisible_comment(self):
        doc = self._doc()
        md = serialize_report_md(doc)
        assert "<!-- spyde:movie cmov1 -->" in md
        assert "<!-- spyde:movie cmov2 -->" in md
        # A standalone poster ref for the non-placeholder; none for the placeholder.
        assert "![My movie](assets/cmov1.png)" in md
        assert "assets/cmov2.png" not in md

    def test_schema_version_unchanged(self):
        assert SCHEMA_VERSION == 1

    def test_older_report_without_movies_still_loads(self):
        # A report.md with no movie markers parses with zero movie cells.
        md = ("---\nversion: 1\ntitle: Old\ntemplate: false\ntype: report\n"
              "created: '2020'\nmodified: '2020'\n---\n\n# Just markdown\n")
        doc = parse_report_md(md)
        assert all(c.cell_type != "movie" for c in doc.cells)
        assert any(c.cell_type == "markdown" for c in doc.cells)

    def test_placeholder_movie_survives_without_poster(self, tmp_path):
        # A placeholder movie (no source, no poster) round-trips as a placeholder.
        doc = ReportDoc(title="ph")
        doc.cells = [Cell(id="conly", cell_type="movie", movie=MovieSpec(),
                          placeholder=True)]
        p = str(tmp_path / "ph.spyde-report")
        write_report(doc, p, {})
        doc2, _ = read_report(p)
        m = doc2.cells[0]
        assert m.cell_type == "movie" and m.placeholder
        assert m.movie is not None and m.movie.source is None
