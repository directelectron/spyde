"""test_report_theme.py — the deck THEME: colours, type, footer bar, logo.

The theme belongs to the DOCUMENT (a talk keeps its look when reopened or handed
to someone), with a separate "set as default" in settings.json that seeds every
NEW deck. Those are deliberately different scopes and the tests pin both.

Covered:
  * normalize_theme fills gaps, coerces types, drops unknown keys
  * an untouched deck serializes NO theme front matter (byte-identical to a
    pre-theme document — opening and saving must not produce a diff)
  * a customised theme round-trips through serialize/parse
  * report_set_theme MERGES (a partial patch must not reset the other fields)
  * report_theme_set_default → settings.json → seeds the next report_new
  * report_theme_reset returns the BUILT-IN look, not the saved default
"""
from __future__ import annotations

import spyde.actions.report.handlers as h
from spyde.actions.report.model import (
    THEME_DEFAULTS, ReportDoc, normalize_theme, parse_report_md,
    serialize_report_md, theme_is_default,
)


class TestNormalizeTheme:
    def test_absent_gives_defaults(self):
        assert normalize_theme(None) == dict(THEME_DEFAULTS)
        assert normalize_theme({}) == dict(THEME_DEFAULTS)
        assert normalize_theme("nonsense") == dict(THEME_DEFAULTS)

    def test_fills_missing_keys(self):
        t = normalize_theme({"accent": "#ff0000"})
        assert t["accent"] == "#ff0000"
        # Every other key still present, at its default.
        assert t["bg"] == THEME_DEFAULTS["bg"]
        assert set(t) == set(THEME_DEFAULTS)

    def test_drops_unknown_keys(self):
        t = normalize_theme({"accent": "#ff0000", "evil": "payload"})
        assert "evil" not in t

    def test_coerces_bools_and_ints(self):
        t = normalize_theme({"footer_show": 0, "slide_numbers": 1,
                             "logo_height": "44"})
        assert t["footer_show"] is False
        assert t["slide_numbers"] is True
        assert t["logo_height"] == 44

    def test_clamps_logo_height(self):
        assert normalize_theme({"logo_height": 5000})["logo_height"] == 200
        assert normalize_theme({"logo_height": 1})["logo_height"] == 8
        # Junk falls back to the default rather than raising.
        assert normalize_theme({"logo_height": "abc"})["logo_height"] == \
            THEME_DEFAULTS["logo_height"]

    def test_long_text_is_capped_but_logo_is_not(self):
        t = normalize_theme({"footer_name": "x" * 5000, "logo": "d" * 5000})
        assert len(t["footer_name"]) == 200
        # A logo is a data: URL — capping it would corrupt the image.
        assert len(t["logo"]) == 5000


class TestSerialization:
    def test_untouched_deck_writes_no_theme_front_matter(self):
        doc = ReportDoc(doc_type="presentation")
        assert theme_is_default(doc.theme)
        assert "theme:" not in serialize_report_md(doc)

    def test_custom_theme_round_trips(self):
        doc = ReportDoc(doc_type="presentation")
        doc.theme = normalize_theme({
            "accent": "#00ff88", "bg": "#000000",
            "footer_name": "Carter Francis",
            "footer_email": "cartsfrancis@gmail.com",
            "footer_show": True, "slide_numbers": False,
            "logo_height": 42,
        })
        back = parse_report_md(serialize_report_md(doc))
        assert back.theme["accent"] == "#00ff88"
        assert back.theme["bg"] == "#000000"
        assert back.theme["footer_name"] == "Carter Francis"
        assert back.theme["footer_email"] == "cartsfrancis@gmail.com"
        assert back.theme["slide_numbers"] is False
        assert back.theme["logo_height"] == 42

    def test_pre_theme_document_loads_with_the_builtin_look(self):
        md = "---\nversion: 1\ntitle: Old\ntype: presentation\n---\n\nhello\n"
        doc = parse_report_md(md)
        assert doc.theme == dict(THEME_DEFAULTS)


class TestThemeHandlers:
    def test_state_always_carries_a_full_theme(self, window):
        session = window["window"]
        h.report_new(session, None, {"type": "presentation"})
        state = h._manager(session).state()
        assert set(state["theme"]) == set(THEME_DEFAULTS)

    def test_set_theme_merges_rather_than_replaces(self, window):
        session = window["window"]
        h.report_new(session, None, {"type": "presentation"})
        h.report_set_theme(session, None, {"theme": {"footer_name": "Carter"}})
        h.report_set_theme(session, None, {"theme": {"accent": "#abcdef"}})
        theme = h._manager(session).doc.theme
        # The SECOND patch must not have wiped the first.
        assert theme["footer_name"] == "Carter"
        assert theme["accent"] == "#abcdef"

    def test_set_theme_marks_dirty(self, window):
        session = window["window"]
        h.report_new(session, None, {"type": "presentation"})
        mgr = h._manager(session)
        mgr.dirty = False
        h.report_set_theme(session, None, {"theme": {"accent": "#111111"}})
        assert mgr.dirty is True

    def test_set_default_seeds_the_next_new_deck(self, window):
        session = window["window"]
        h.report_new(session, None, {"type": "presentation"})
        h.report_set_theme(session, None, {"theme": {
            "footer_name": "Carter Francis", "accent": "#00ff88"}})
        h.report_theme_set_default(session, None, {})

        # A brand-new deck starts from that default.
        h.report_new(session, None, {"type": "presentation"})
        theme = h._manager(session).doc.theme
        assert theme["footer_name"] == "Carter Francis"
        assert theme["accent"] == "#00ff88"

    def test_reset_returns_the_builtin_look_not_the_saved_default(self, window):
        session = window["window"]
        h.report_new(session, None, {"type": "presentation"})
        h.report_set_theme(session, None, {"theme": {"accent": "#00ff88"}})
        h.report_theme_set_default(session, None, {})
        # Still on the customised deck; reset must go to the STOCK look, or a
        # customised default would leave no way back to it.
        h.report_theme_reset(session, None, {})
        assert h._manager(session).doc.theme == dict(THEME_DEFAULTS)

    def test_use_default_applies_the_saved_default(self, window):
        session = window["window"]
        h.report_new(session, None, {"type": "presentation"})
        h.report_set_theme(session, None, {"theme": {"accent": "#00ff88"}})
        h.report_theme_set_default(session, None, {})
        h.report_theme_reset(session, None, {})
        h.report_theme_use_default(session, None, {})
        assert h._manager(session).doc.theme["accent"] == "#00ff88"

    def test_handlers_error_cleanly_with_no_open_report(self, window, captured_messages):
        session = window["window"]
        h._manager(session).doc = None
        for fn in (h.report_set_theme, h.report_theme_set_default,
                   h.report_theme_reset, h.report_theme_use_default):
            fn(session, None, {"theme": {"accent": "#fff"}})
        errs = [m for m in captured_messages if m.get("type") == "error"]
        assert len(errs) >= 4
