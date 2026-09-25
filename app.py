"""MoDiLab: bilingual virtual laboratory for teaching molecular dynamics.

Run with:  streamlit run app.py
"""

import streamlit as st

from modilab import __version__
from modilab.i18n import DEFAULT, LANGUAGES, Translator
from modilab.ui import pages as P
from modilab.ui.common import units_sidebar
from modilab.ui.lab import lab

st.set_page_config(page_title="MoDiLab", page_icon="⚛️", layout="wide")
st.session_state.setdefault("lang", DEFAULT)

with st.sidebar:
    st.radio("Bahasa / Language", list(LANGUAGES), key="lang",
             format_func=LANGUAGES.get, horizontal=True)
    t = Translator(st.session_state.lang)
    units_sidebar(t)
    st.caption(f"MoDiLab v{__version__}")
    st.divider()

navigation = st.navigation({
    t("nav.sec_start"): [st.Page(P.home, title=t("nav.home"), url_path="home", default=True)],
    t("nav.sec_modules"): [
        st.Page(getattr(P, m), title=t(f"nav.{m}"), url_path=m)
        for m in ("m01", "m02", "m03", "m04", "m05", "m06")],
    t("nav.sec_lab"): [st.Page(lab, title=t("nav.lab"), url_path="lab")],
})
navigation.run()
