"""Minimal translation layer.

Every user facing string lives in modilab/locales/<code>.json. A value can be
a string or a list of paragraphs (joined with blank lines). Placeholders use
str.format syntax, for example "{n} particles". To add a language, copy
en.json, translate the values, and register it in LANGUAGES.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LOCALE_DIR = Path(__file__).parent / "locales"
LANGUAGES = {"id": "Bahasa Indonesia", "en": "English"}
DEFAULT = "id"


@lru_cache(maxsize=None)
def load(code: str) -> dict:
    with open(LOCALE_DIR / f"{code}.json", encoding="utf-8") as fh:
        return json.load(fh)


class Translator:
    def __init__(self, code: str = DEFAULT):
        self.code = code if code in LANGUAGES else DEFAULT
        self._strings = load(self.code)
        self._fallback = load("en")

    def __call__(self, key: str, **kwargs) -> str:
        value = self._strings.get(key, self._fallback.get(key, key))
        if isinstance(value, list):
            value = "\n\n".join(value)
        return value.format(**kwargs) if kwargs else value

    def num(self, x: float, digits: int = 3) -> str:
        """Format a number with the decimal separator of the active language."""
        s = f"{x:,.{digits}f}"
        if self.code == "id":
            s = s.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")
        return s

    @property
    def plotly_separators(self) -> str:
        return ",." if self.code == "id" else ".,"
