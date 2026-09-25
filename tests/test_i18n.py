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
