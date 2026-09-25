"""Shared interface helpers: translation, cached simulation, plots and quizzes."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..core import Parameters, run
from ..i18n import DEFAULT, Translator, module_text, quiz
from ..units_io import element, element_symbols, to_real

BLUE, RED, GREEN, GREY, ORANGE = "#1F5AA6", "#C0392B", "#2E8B57", "#6B7785", "#D98E04"


def tr() -> Translator:
    return Translator(st.session_state.get("lang", DEFAULT))


@st.cache_data(max_entries=48, show_spinner=False)
def cached_run(**kwargs):
    """Simulations are cached, so identical settings from many students run only once."""
    return run(Parameters(**kwargs))


def simulate(t: Translator, **kwargs):
    with st.spinner(t("common.running")):
        return cached_run(**kwargs)


def el_name(t: Translator, symbol: str) -> str:
    return element(symbol)[f"name_{t.code}"]


def element_picker(t: Translator, key: str, default: str = "Ar", container=st) -> str:
    symbols = element_symbols()
    sym = container.selectbox(t("common.element"), symbols, index=symbols.index(default),
                              key=key, format_func=lambda k: f"{k} ({el_name(t, k)})")
    if element(sym)["quantum_warning"]:
        container.warning(t("common.quantum_warning", el=el_name(t, sym)))
    return sym


def style(fig: go.Figure, t: Translator, height: int = 420, **kw) -> go.Figure:
    fig.update_layout(height=height, separators=t.plotly_separators,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), **kw)
    return fig


def show(fig: go.Figure, container=st):
    container.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- page layout
def module_page(name: str, sim_fn):
    t = tr()
    before, after = module_text(t.code, name)
    st.title(t(f"nav.{name}"))
    st.markdown(before)
    st.divider()
    st.subheader(t("common.sim_header"))
    sim_fn(t)
    st.divider()
    st.markdown(after)
    render_quiz(t, name)


def render_quiz(t: Translator, name: str):
    questions = quiz(t.code)[name]
    st.markdown(f"### {t('quiz.header')}")
    answers = []
    for i, q in enumerate(questions):
        answers.append(st.radio(f"{i + 1}. {q['q']}", range(len(q["options"])),
                                format_func=lambda k, q=q: q["options"][k],
                                index=None, key=f"quiz_{name}_{i}"))
    if st.button(t("quiz.check"), key=f"quiz_{name}_check"):
        if any(a is None for a in answers):
            st.warning(t("quiz.unanswered"))
            return
        score = 0
        for i, (q, a) in enumerate(zip(questions, answers)):
            if a == q["answer"]:
                score += 1
                st.success(f"{i + 1}. " + t("quiz.correct", explain=q["explain"]))
            else:
                st.error(f"{i + 1}. " + t("quiz.wrong", ans=q["options"][q["answer"]],
                                          explain=q["explain"]))
        st.info(t("quiz.score", s=score, n=len(questions)))


# --------------------------------------------------------------------------- figures
def box_axes(box, grey=GREY):
    axis = dict(showgrid=False, zeroline=False, mirror=True, showline=True, linecolor=grey)
    return (dict(axis, range=[0, box[0]], constrain="domain"),
            dict(axis, range=[0, box[1]], scaleanchor="x"))


def animation_controls(t: Translator, labels, names, x_slider=0.2, prefix="t* = "):
    menus = [dict(type="buttons", direction="left", x=0.0, y=-0.02,
                  xanchor="left", yanchor="top",
                  buttons=[dict(label=t("anim.play"), method="animate",
                                args=[None, dict(frame=dict(duration=60, redraw=True),
                                                 fromcurrent=True)]),
                           dict(label=t("anim.pause"), method="animate",
                                args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                   mode="immediate")])])]
    sliders = [dict(x=x_slider, len=1 - x_slider, y=-0.02, currentvalue=dict(prefix=prefix),
                    steps=[dict(method="animate", label=lab,
                                args=[[nm], dict(mode="immediate", frame=dict(duration=0))])
                           for lab, nm in zip(labels, names)])]
    return menus, sliders


def time_controls(t: Translator, D, r, idx):
    labels = [D.fmt("time", r.frame_time[k], 2) for k in idx]
    return animation_controls(t, labels, [str(k) for k in idx], prefix=f"{t('q.time')}: ")


def speed_animation_2d(t: Translator, r, max_frames: int = 150, height: int = 600) -> go.Figure:
    """2D animation of atoms coloured by speed."""
    D = display(t, 2)
    stride = max(1, len(r.positions) // max_frames)
    idx = np.arange(0, len(r.positions), stride)
    speeds = D("speed", np.linalg.norm(r.velocities, axis=-1))
    cmax = float(np.percentile(speeds, 99))
    size = max(4.0, 520 / r.box.max() * 0.9)

    def trace(k):
        return go.Scatter(
            x=r.positions[k, :, 0], y=r.positions[k, :, 1], mode="markers",
            marker=dict(size=size, color=speeds[k], cmin=0, cmax=cmax, colorscale="Viridis",
                        line=dict(width=0.5, color="white"),
                        colorbar=dict(title=D.unit("speed") if D.real else t("anim.speed_bar"))),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>")

    fig = go.Figure(data=[trace(idx[0])],
                    frames=[go.Frame(data=[trace(k)], name=str(k)) for k in idx])
    xa, ya = box_axes(r.box)
    menus, sliders = time_controls(t, D, r, idx)
    return style(fig, t, height=height, plot_bgcolor="#F7F9FC", xaxis=xa, yaxis=ya,
                 updatemenus=menus, sliders=sliders, showlegend=False)


def snapshot_3d(t: Translator, r, frame: int = -1, height: int = 520) -> go.Figure:
    D = display(t, 3)
    pos = r.positions[frame]
    speed = D("speed", np.linalg.norm(r.velocities[frame], axis=1))
    L = r.box
    edges = []
    for a in (0, L[0]):
        for b in (0, L[1]):
            edges.append(([a, a], [b, b], [0, L[2]]))
    for a in (0, L[0]):
        for c in (0, L[2]):
            edges.append(([a, a], [0, L[1]], [c, c]))
    for b in (0, L[1]):
        for c in (0, L[2]):
            edges.append(([0, L[0]], [b, b], [c, c]))
    fig = go.Figure([go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=False,
                                  line=dict(color=GREY, width=2), hoverinfo="skip")
                     for x, y, z in edges])
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers", showlegend=False,
        marker=dict(size=6, color=speed, colorscale="Viridis",
                    colorbar=dict(title=D.unit("speed") if D.real else t("anim.speed_bar")),
                    line=dict(width=0))))
    fig.update_layout(scene=dict(aspectmode="cube", xaxis_title="x/σ",
                                 yaxis_title="y/σ", zaxis_title="z/σ"))
    return style(fig, t, height=height)


# --------------------------------------------------------------------------- display units
# quantity key -> (to_real key 2D, to_real key 3D, reduced axis symbol, reduced unit 2D, 3D)
QUANTITIES = {
    "time": ("time", "time", "t*", "τ", "τ"),
    "temperature": ("temperature", "temperature", "T*", "ε/k_B", "ε/k_B"),
    "energy": ("energy", "energy", "E/ε", "ε", "ε"),
    "pair_energy": ("energy", "energy", "E/ε", "ε", "ε"),
    "pressure": ("pressure2d", "pressure3d", "P*", "ε/σ²", "ε/σ³"),
    "length": ("length", "length", "r/σ", "σ", "σ"),
    "msd": ("length2", "length2", "MSD/σ²", "σ²", "σ²"),
    "speed": ("velocity", "velocity", "|v|*", "√(ε/m)", "√(ε/m)"),
    "velocity": ("velocity", "velocity", "v*", "√(ε/m)", "√(ε/m)"),
    "speed_pdf": ("speed_pdf", "speed_pdf", "f(v*)", "", ""),
    "density": ("density2d", "density3d", "ρ*", "σ⁻²", "σ⁻³"),
    "diffusion": ("diffusion", "diffusion", "D*", "σ²/τ", "σ²/τ"),
}


class Display:
    """Converts reduced simulation output to the units chosen in the sidebar."""

    def __init__(self, t: Translator, dim: int = 2):
        self.t = t
        self.dim = dim
        self.real = st.session_state.get("units", "reduced") == "real"
        self.symbol = st.session_state.get("ref_el", "Ar")

    def _key(self, q: str) -> str:
        return QUANTITIES[q][0 if self.dim == 2 else 1]

    def unit(self, q: str) -> str:
        if self.real:
            return to_real(self._key(q), 1.0, self.symbol)[1]
        return QUANTITIES[q][3 if self.dim == 2 else 4]

    def __call__(self, q: str, x):
        """Value(s) in display units."""
        if not self.real:
            return x
        return to_real(self._key(q), np.asarray(x, dtype=float), self.symbol)[0]

    def label(self, q: str) -> str:
        name = self.t(f"q.{q}")
        return f"{name} ({self.unit(q)})" if self.real else f"{name} {QUANTITIES[q][2]}"

    def fmt(self, q: str, x: float, digits: int = 3) -> str:
        v = float(self(q, x))
        if self.real and (abs(v) >= 1e5 or 0 < abs(v) < 1e-2):
            mant, exp = f"{v:.2e}".split("e")
            return f"{self.t.num(float(mant), 2)} × 10^{int(exp)} {self.unit(q)}"
        return f"{self.t.num(v, digits)} {self.unit(q)}".strip()

    def other(self, q: str, x: float, digits: int = 3) -> str:
        """The same value in the other unit system (for secondary display)."""
        self.real = not self.real
        try:
            return self.fmt(q, x, digits)
        finally:
            self.real = not self.real


    def equivalents(self, items) -> str:
        """Caption listing real equivalents of reduced inputs, e.g. [(label, q, x)]."""
        if not self.real:
            return ""
        parts = [f"{lab.replace(' T*', '')} ≈ {self.fmt(q, x, 1 if q == 'temperature' else 3)}"
                 for lab, q, x in items]
        return self.t("units.equiv", el=el_name(self.t, self.symbol), items="; ".join(parts))


def display(t: Translator, dim: int = 2) -> Display:
    return Display(t, dim)


def units_sidebar(t: Translator):
    """Global unit controls shown in the sidebar on every page."""
    st.session_state.setdefault("units", "reduced")
    st.session_state.setdefault("ref_el", "Ar")
    st.radio(t("units.label"), ["reduced", "real"], key="units", horizontal=True,
             format_func=lambda k: t(f"units.{k}"), help=t("units.help"))
    symbols = element_symbols()
    st.selectbox(t("units.element"), symbols, key="ref_el",
                 format_func=lambda k: f"{k} ({el_name(t, k)})")
    if element(st.session_state.ref_el)["quantum_warning"]:
        st.warning(t("common.quantum_warning", el=el_name(t, st.session_state.ref_el)))
