"""Home page and learning modules 1 to 6."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import analysis as A
from ..core import build_lattice, dimer, lj_force, lj_potential
from ..i18n import module_text
from ..units_io import element, element_source, element_symbols, to_real, unit_scales
from .common import (BLUE, GREEN, GREY, ORANGE, RED, display, el_name, module_page,
                     show, simulate, snapshot_3d, speed_animation_2d, style,
                     time_controls, tr)

ELEMENT_COLOURS = {"He": "#8E44AD", "Ne": RED, "Ar": BLUE, "Kr": GREEN, "Xe": ORANGE}


def element_table(t, extended: bool = False) -> pd.DataFrame:
    rows = []
    for s in element_symbols():
        e, u = element(s), unit_scales(s)
        row = {t("col.symbol"): s, t("col.name"): el_name(t, s),
               t("col.mass"): t.num(e["mass_u"], 3),
               t("col.sigma"): t.num(e["sigma_nm"] * 10, 2),
               t("col.eps_K"): t.num(e["epsilon_K"], 1),
               t("col.eps_meV"): t.num(u["epsilon_eV"] * 1e3, 2)}
        if extended:
            row[t("col.tau")] = t.num(u["tau_s"] * 1e12, 2)
            row[t("col.vel")] = t.num(u["velocity_m_s"], 0)
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== home
def home():
    t = tr()
    st.title("MoDiLab")
    st.markdown(module_text(t.code, "home")[0])
    st.subheader(t("home.elements_title"))
    st.dataframe(element_table(t), hide_index=True, width="stretch")
    src = element_source()
    st.caption(t("home.source", cite=f"{src['citation']} https://doi.org/{src['doi']}"))
    st.caption(t("home.variability"))


# =========================================================================== module 1
def _m01_sim(t):
    c1, c2 = st.columns([3, 1])
    chosen = c1.multiselect(t("m01.pick"), element_symbols(), default=["Ne", "Ar", "Kr", "Xe"],
                            format_func=lambda k: f"{k} ({el_name(t, k)})")
    unit = c2.radio(t("m01.units"), ["real", "reduced"], horizontal=False,
                    index=0 if st.session_state.get("units") == "real" else 1,
                    format_func=lambda k: t(f"common.{k}"))
    if not chosen:
        return
    r = np.linspace(0.92, 3.0, 400)
    fu, ff = go.Figure(), go.Figure()
    eps_max = 0.0
    for s in chosen:
        u = unit_scales(s)
        sig_A, eps_meV = element(s)["sigma_nm"] * 10, u["epsilon_eV"] * 1e3
        if unit == "real":
            x, U, F = r * sig_A, lj_potential(r) * eps_meV, lj_force(r) * eps_meV / sig_A
            eps_max = max(eps_max, eps_meV)
        else:
            x, U, F = r, lj_potential(r), lj_force(r)
            eps_max = 1.0
        col = ELEMENT_COLOURS.get(s)
        fu.add_trace(go.Scatter(x=x, y=U, name=s, line=dict(color=col, width=2)))
        ff.add_trace(go.Scatter(x=x, y=F, name=s, line=dict(color=col, width=2)))
    real = unit == "real"
    ymax_f = 3.0 * (max(unit_scales(s)["epsilon_eV"] * 1e3 / (element(s)["sigma_nm"] * 10)
                        for s in chosen) if real else 1.0)
    for fig in (fu, ff):
        fig.add_hline(y=0, line_color=GREY, line_width=1)
    style(fu, t, title=t("m01.u_title"),
          xaxis_title=t("m01.r_axis_real" if real else "m01.r_axis_red"),
          yaxis_title=t("m01.u_axis_real" if real else "m01.u_axis_red"),
          yaxis_range=[-1.2 * eps_max, 1.5 * eps_max])
    style(ff, t, title=t("m01.f_title"),
          xaxis_title=t("m01.r_axis_real" if real else "m01.r_axis_red"),
          yaxis_title=t("m01.f_axis_real" if real else "m01.f_axis_red"),
          yaxis_range=[-ymax_f, ymax_f])
    a, b = st.columns(2)
    show(fu, a)
    show(ff, b)
    rows = [{t("col.symbol"): s, t("col.sigma"): t.num(element(s)["sigma_nm"] * 10, 2),
             t("col.rmin"): t.num(2 ** (1 / 6) * element(s)["sigma_nm"] * 10, 2),
             t("col.eps_meV"): t.num(unit_scales(s)["epsilon_eV"] * 1e3, 2),
             t("col.eps_K"): t.num(element(s)["epsilon_K"], 1)} for s in chosen]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def m01():
    module_page("m01", _m01_sim)


# =========================================================================== module 2
def _m02_sim(t):
    D = display(t)
    with st.form("m02_form"):
        c = st.columns(3)
        r0 = c[0].slider(t("m02.r0"), 1.05, 2.5, 1.5, 0.05)
        dt = c[1].select_slider(t("m02.dt"), [0.001, 0.002, 0.005, 0.01, 0.02], 0.01)
        n = c[2].number_input(t("m02.steps"), 200, 10000, 2000, 200)
        st.form_submit_button(t("common.run"), type="primary")
    if D.real:
        st.caption(D.equivalents([("r₀", "length", r0), ("Δt", "time", dt)]))
    res = {m: dimer(r0, 0.0, dt, int(n), m) for m in ("euler", "verlet")}
    names = {"euler": t("m02.euler"), "verlet": t("m02.verlet")}
    cols = {"euler": RED, "verlet": BLUE}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(t("m02.plot_r"), t("m02.plot_e")))
    for m, d in res.items():
        tt = D("time", d["t"])
        fig.add_trace(go.Scatter(x=tt, y=D("length", d["r"]), name=names[m],
                                 line=dict(color=cols[m])), 1, 1)
        fig.add_trace(go.Scatter(x=tt, y=D("pair_energy", d["energy"]), name=names[m],
                                 showlegend=False, line=dict(color=cols[m])), 2, 1)
    fig.update_xaxes(title_text=D.label("time"), row=2, col=1)
    fig.update_yaxes(title_text=D.label("length"), row=1, col=1)
    fig.update_yaxes(title_text=D.label("pair_energy"), row=2, col=1)
    ph = go.Figure([go.Scatter(x=D("length", d["r"]), y=D("velocity", d["v"]), name=names[m],
                               mode="lines", line=dict(color=cols[m])) for m, d in res.items()])
    a, b = st.columns([3, 2])
    show(style(fig, t, height=560), a)
    show(style(ph, t, height=560, title=t("m02.plot_phase"),
               xaxis_title=D.label("length"), yaxis_title=D.label("velocity")), b)

    def drift(e):
        return float(np.nanmax(np.abs(e - e[0])))

    st.markdown(t("m02.drift", e=D.fmt("pair_energy", drift(res["euler"]["energy"]), 5),
                  v=D.fmt("pair_energy", drift(res["verlet"]["energy"]), 5)))
    bad = np.flatnonzero(np.isnan(res["euler"]["r"]))
    if len(bad):
        st.warning(t("m02.broke", t=D.fmt("time", res["euler"]["t"][bad[0]], 2)))


def m02():
    module_page("m02", _m02_sim)


# =========================================================================== module 3
def _periodic_animation(t, r, images: bool, tracked: int = 0, max_frames: int = 100):
    D = display(t)
    stride = max(1, len(r.positions) // max_frames)
    idx = np.arange(0, len(r.positions), stride)
    L = r.box
    shifts = [np.array([i, j]) * L for i in (-1, 0, 1) for j in (-1, 0, 1)
              if images or (i == 0 and j == 0)]
    size = max(4.0, 520 / (L.max() * (3 if images else 1)) * 0.9)

    def traces(k):
        pos = r.positions[k]
        main = go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers", hoverinfo="skip",
                          marker=dict(size=size, color=BLUE, line=dict(width=0.5, color="white")))
        others = [s for s in shifts if s.any()]
        img = np.concatenate([pos + s for s in others]) if others else np.empty((0, 2))
        ghost = go.Scatter(x=img[:, 0], y=img[:, 1], mode="markers", hoverinfo="skip",
                           marker=dict(size=size, color="#B8C4D6",
                                       line=dict(width=0.5, color="white")))
        tr_pts = np.array([pos[tracked] + s for s in shifts])
        red = go.Scatter(x=tr_pts[:, 0], y=tr_pts[:, 1], mode="markers", hoverinfo="skip",
                         marker=dict(size=size * 1.1, color=RED, line=dict(width=1, color="white")))
        return [ghost, main, red]

    fig = go.Figure(data=traces(idx[0]),
                    frames=[go.Frame(data=traces(k), name=str(k)) for k in idx])
    lo, hi = (-L, 2 * L) if images else (np.zeros(2), L)
    fig.add_shape(type="rect", x0=0, y0=0, x1=L[0], y1=L[1], line=dict(color="black", width=2))
    menus, sliders = time_controls(t, D, r, idx)
    axis = dict(showgrid=False, zeroline=False, showticklabels=False)
    return style(fig, t, height=620, showlegend=False, plot_bgcolor="#F7F9FC",
                 xaxis=dict(axis, range=[lo[0], hi[0]], constrain="domain"),
                 yaxis=dict(axis, range=[lo[1], hi[1]], scaleanchor="x"),
                 updatemenus=menus, sliders=sliders)


def _m03_sim(t):
    st.markdown(f"#### {t('m03.anim_title')}")
    r = simulate(t, dim=2, lattice="square", cells=6, density=0.35, temperature=1.2,
                 n_steps=1500, save_every=15, seed=7)
    images = st.checkbox(t("m03.show_images"), value=True)
    show(_periodic_animation(t, r, images))
    st.caption(t("m03.tracked"))

    st.markdown(f"#### {t('m03.conv_title')}")
    sym = st.session_state.get("ref_el", "Ar")
    st.caption(t("m03.conv_hint", el=el_name(t, sym)))
    c = st.columns(5)
    vals = {"temperature": c[0].number_input(t("m03.T"), value=1.0, step=0.1, format="%.3f"),
            "density3d": c[1].number_input(t("m03.rho"), value=0.8, step=0.05, format="%.3f"),
            "time": c[2].number_input(t("m03.t"), value=15.0, step=1.0, format="%.3f"),
            "pressure3d": c[3].number_input(t("m03.P"), value=1.0, step=0.1, format="%.3f"),
            "diffusion": c[4].number_input(t("m03.D"), value=0.05, step=0.01, format="%.4f")}
    out = st.columns(5)
    for col, (q, v) in zip(out, vals.items()):
        real, unit = to_real(q, v, sym)
        if q == "diffusion":
            mant, exp = f"{real:.2e}".split("e")
            txt = f"{t.num(float(mant), 2)} × 10^{int(exp)}"
        else:
            txt = t.num(real, 2)
        col.metric(unit, txt)
    st.markdown(f"#### {t('m03.table_title')}")
    st.dataframe(element_table(t, extended=True), hide_index=True, width="stretch")


def m03():
    module_page("m03", _m03_sim)


# =========================================================================== module 4
def _m04_sim(t):
    D = display(t)
    with st.form("m04_form"):
        c = st.columns(4)
        T_init = c[0].slider(t("m04.T_init"), 0.2, 3.0, 2.0, 0.1)
        T_target = c[1].slider(t("m04.T_target"), 0.2, 3.0, 0.8, 0.1)
        tau = c[2].select_slider(t("m04.tau"), [0.02, 0.05, 0.1, 0.5, 1.0], 0.1)
        cells = c[3].slider(t("m04.size"), 4, 14, 8)
        st.form_submit_button(t("common.run"), type="primary")
    if D.real:
        st.caption(D.equivalents([(t("m04.T_init"), "temperature", T_init),
                                  (t("m04.T_target"), "temperature", T_target),
                                  ("τ_T", "time", tau)]))
    common = dict(dim=2, lattice="square", cells=cells, density=0.6, n_steps=3000,
                  dt=0.005, save_every=50, seed=11)
    nve = simulate(t, thermostat="nve", temperature=T_init, **common)
    nvt = simulate(t, thermostat="berendsen", temperature=T_target,
                   initial_temperature=T_init, tau_t=tau, **common)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(t("m04.plot_T"), t("m04.plot_E")))
    for r, name, col in ((nve, t("m04.nve"), RED), (nvt, t("m04.nvt"), BLUE)):
        fig.add_trace(go.Scatter(x=D("time", r.time), y=D("temperature", r.temperature),
                                 name=name, line=dict(color=col, width=1)), 1, 1)
        fig.add_trace(go.Scatter(x=D("time", r.time), y=D("energy", r.total_energy),
                                 name=name, showlegend=False,
                                 line=dict(color=col, width=1.5)), 1, 2)
    fig.add_hline(y=float(D("temperature", T_target)), line_dash="dash", line_color=GREY,
                  row=1, col=1, annotation_text=t("m04.target"))
    fig.update_xaxes(title_text=D.label("time"))
    fig.update_yaxes(title_text=D.label("temperature"), row=1, col=1)
    fig.update_yaxes(title_text=D.label("energy"), row=1, col=2)
    show(style(fig, t, height=460))
    half = len(nve.temperature) // 2

    def fluct(r):
        x = r.temperature[half:]
        return 100 * np.std(x) / np.mean(x)

    st.markdown(t("m04.fluct", a=t.num(fluct(nve), 2), b=t.num(fluct(nvt), 2),
                  n=nve.params.n_particles))


def m04():
    module_page("m04", _m04_sim)


# =========================================================================== module 5
def _m05_sim(t):
    with st.form("m05_form"):
        c = st.columns(2)
        dim = c[0].radio(t("m05.dim"), [2, 3], horizontal=True,
                         format_func=lambda d: t(f"m05.dim{d}"))
        T0 = c[1].slider(t("m05.T"), 0.5, 2.0, 1.0, 0.1)
        st.form_submit_button(t("common.run"), type="primary")
    D = display(t, dim)
    if D.real:
        st.caption(D.equivalents([(t("m05.T"), "temperature", T0)]))
    geo = dict(lattice="square", cells=12) if dim == 2 else dict(lattice="fcc", cells=3)
    r = simulate(t, dim=dim, density=0.5, temperature=T0, thermostat="nve",
                 velocity_init="equal_speed", n_steps=1500, save_every=10, dt=0.005,
                 seed=5, **geo)
    speeds = np.linalg.norm(r.velocities, axis=-1)
    vmax = float(np.percentile(speeds, 99.5)) * 1.15
    bins = np.linspace(0, vmax, 26)
    v = np.linspace(0, vmax, 200)

    def frame_T(k):
        return float(np.sum(r.velocities[k] ** 2) / (dim * len(r.velocities[k]) - dim))

    deviation = []
    for k in range(len(r.positions)):
        x, h = A.speed_histogram(r.velocities[k], bins)
        deviation.append(np.mean(np.abs(h - A.maxwell_boltzmann_speed(x, frame_T(k), dim))))

    k = st.select_slider(t("m05.frame"), options=list(range(len(r.frame_time))), value=0,
                         format_func=lambda i: D.fmt("time", r.frame_time[i], 2))
    x, h = A.speed_histogram(r.velocities[k], bins)
    Tk = frame_T(k)
    mb = A.maxwell_boltzmann_speed(v, Tk, dim)
    fh = go.Figure()
    fh.add_trace(go.Bar(x=D("speed", x), y=D("speed_pdf", h), name=t("m05.sim"),
                        marker_color=BLUE, opacity=0.7))
    fh.add_trace(go.Scatter(x=D("speed", v), y=D("speed_pdf", mb),
                            name=t("m05.mb", d=t(f"m05.dim{dim}"),
                                   T=D.fmt("temperature", Tk, 2 if not D.real else 0)),
                            line=dict(color=RED, width=2)))
    ytop = max(1.2 * float(np.max(mb)), min(float(h.max()) * 1.05, 6.0))
    style(fh, t, height=440, bargap=0.03,
          title=t("m05.hist_title", t=D.fmt("time", r.frame_time[k], 2)),
          xaxis_title=D.label("speed"), yaxis_title=D.label("speed_pdf"),
          yaxis_range=[0, float(D("speed_pdf", ytop))])
    ft = D("time", r.frame_time)
    fd = go.Figure(go.Scatter(x=ft, y=deviation, line=dict(color=GREEN)))
    fd.add_vline(x=float(ft[k]), line_color=RED, line_dash="dash")
    style(fd, t, height=440, title=t("m05.dev_title"), xaxis_title=D.label("time"),
          yaxis_title=t("m05.dev_axis"), showlegend=False)
    a, b = st.columns([3, 2])
    show(fh, a)
    show(fd, b)


def m05():
    module_page("m05", _m05_sim)


# =========================================================================== module 6
PHASES = {2: {"solid": (0.90, 0.30), "liquid": (0.75, 1.00), "gas": (0.10, 1.50)},
          3: {"solid": (1.00, 0.50), "liquid": (0.80, 1.00), "gas": (0.05, 1.50)}}


def _m06_sim(t):
    with st.form("m06_form"):
        c = st.columns(3)
        dim = c[0].radio(t("m06.system"), [2, 3], format_func=lambda d: t(f"m06.sys{d}"))
        phase = c[1].radio(t("m06.phase"), ["solid", "liquid", "gas"],
                           format_func=lambda k: t(f"m06.{k}"))
        size = c[2].radio(t("m06.size"), ["small", "large"],
                          format_func=lambda k: {"small": "2D: N = 80 · 3D: N = 108",
                                                 "large": "2D: N = 168 · 3D: N = 256"}[k])
        st.form_submit_button(t("common.run"), type="primary")
    D = display(t, dim)
    sym = D.symbol
    lattice = "hex" if dim == 2 else "fcc"
    cells = {2: {"small": 8, "large": 12}, 3: {"small": 3, "large": 4}}[dim][size]
    rho, T = PHASES[dim][phase]
    st.caption(t("m06.conditions", rho=D.fmt("density", rho), T=D.fmt("temperature", T, 2)))
    box = build_lattice(lattice, cells, rho)[1]
    r_cut = float(min(2.5, 0.49 * box.min()))
    r = simulate(t, dim=dim, lattice=lattice, cells=cells, density=rho, temperature=T,
                 r_cut=r_cut, n_steps=2000 if dim == 2 else 1500, save_every=20, seed=3)

    x, g = A.rdf(r, 0.4, n_bins=120)
    n_r = A.coordination_number(x, g, rho, dim)
    r_min = A.first_minimum_after_peak(x, g)
    cn = float(np.interp(r_min, x, n_r))
    ideal = 6 if dim == 2 else 12
    xd, r_min_d = D("length", x), float(D("length", r_min))

    snap = snapshot_3d(t, r) if dim == 3 else speed_animation_2d(t, r, height=520)
    fg = go.Figure(go.Scatter(x=xd, y=g, line=dict(color=BLUE, width=2)))
    fg.add_hline(y=1, line_dash="dot", line_color=GREY)
    fg.add_vline(x=r_min_d, line_dash="dash", line_color=ORANGE,
                 annotation_text=t("m06.first_shell"))
    style(fg, t, height=320, title=t("m06.gr"), xaxis_title=D.label("length"),
          yaxis_title="g(r)", showlegend=False)
    fn = go.Figure(go.Scatter(x=xd, y=n_r, line=dict(color=GREEN, width=2)))
    fn.add_vline(x=r_min_d, line_dash="dash", line_color=ORANGE)
    fn.add_hline(y=ideal, line_dash="dot", line_color=GREY)
    style(fn, t, height=320, title=t("m06.cn"), xaxis_title=D.label("length"),
          yaxis_title="n(r)", showlegend=False, yaxis_range=[0, ideal * 2])
    a, b = st.columns([1, 1])
    with a:
        st.markdown(f"**{t('m06.snapshot')}**")
        show(snap)
    with b:
        show(fg)
        show(fn)
    r_peak = float(x[np.argmax(g)])
    st.markdown(t("m06.result", r=t.num(r_peak, 2),
                  ra=t.num(to_real("length", r_peak, sym)[0], 2),
                  el=el_name(t, sym), cn=t.num(cn, 1), ideal=ideal))
    if dim == 3:
        st.markdown(t("m06.density", rho=t.num(rho, 2),
                      d=t.num(to_real("density3d", rho, sym)[0], 3), el=el_name(t, sym)))


def m06():
    module_page("m06", _m06_sim)
