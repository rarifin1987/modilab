"""Consistency tests for the translation files."""

import string

import pytest

from modilab.i18n import LANGUAGES, Translator, load


def _placeholders(value):
    text = "\n".join(value) if isinstance(value, list) else value
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


@pytest.mark.parametrize("code", [c for c in LANGUAGES if c != "en"])
def test_same_keys_as_english(code):
    assert set(load(code)) == set(load("en"))


@pytest.mark.parametrize("code", list(LANGUAGES))
def test_same_placeholders_as_english(code):
    en, other = load("en"), load(code)
    for key in en:
        assert _placeholders(other[key]) == _placeholders(en[key]), key


def test_decimal_separator():
    assert Translator("id").num(1234.5, 2) == "1.234,50"
    assert Translator("en").num(1234.5, 2) == "1,234.50"


def test_unknown_language_falls_back_to_default():
    assert Translator("xx").code == "id"


# --------------------------------------------------------------------------- learning content
from modilab.i18n import CONTENT_DIR, SIM_MARKER, quiz  # noqa: E402

MODULES = ["m01", "m02", "m03", "m04", "m05", "m06"]


@pytest.mark.parametrize("code", list(LANGUAGES))
@pytest.mark.parametrize("name", MODULES)
def test_every_module_has_text_with_one_simulation_marker(code, name):
    text = (CONTENT_DIR / code / f"{name}.md").read_text(encoding="utf-8")
    assert text.count(SIM_MARKER) == 1


@pytest.mark.parametrize("code", [c for c in LANGUAGES if c != "en"])
def test_quiz_matches_english_structure(code):
    en, other = quiz("en"), quiz(code)
    assert set(en) == set(other) == set(MODULES)
    for m in MODULES:
        assert len(en[m]) == len(other[m])
        for a, b in zip(en[m], other[m]):
            assert a["answer"] == b["answer"]
            assert len(a["options"]) == len(b["options"])
            assert 0 <= b["answer"] < len(b["options"])


def test_navigation_titles_exist_for_every_module():
    for code in LANGUAGES:
        t = Translator(code)
        for m in MODULES:
            assert t(f"nav.{m}") != f"nav.{m}"
