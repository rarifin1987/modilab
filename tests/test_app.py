"""Smoke test: the full multipage application starts in both languages."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


@pytest.mark.parametrize("lang", ["id", "en"])
def test_app_starts(lang):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["lang"] = lang
    at.run()
    assert not at.exception
    assert at.title[0].value == "MoDiLab"
