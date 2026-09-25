"""Free laboratory: the open ended 2D simulator from earlier versions."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import analysis as A
from ..core import Parameters, run
from ..units_io import element, thermo_csv, to_real, trajectory_extxyz, unit_scales
from .common import (BLUE, GREEN, GREY, RED, display, el_name, show,
                     speed_animation_2d, style, tr)

PRESETS = {
    "gas": dict(density=0.05, temperature=1.50, dt=0.005),
    "liquid": dict(density=0.70, temperature=1.00, dt=0.005),
    "solid": dict(density=0.95, temperature=0.20, dt=0.005),
}
THERMOSTAT_KEYS = ["nve", "berendsen", "rescale"]


def _apply_preset():
    values = PRESETS.get(st.session_state.lab_preset)
    if values:
        for k, v in values.items():
            st.session_state[f"lab_{k}"] = v


def lab():
    t = tr()
    for k, v in {"lab_preset": "liquid", **{f"lab_{k}": v for k, v in PRESETS["liquid"].items()}}.items():
        st.session_state.setdefault(k, v)

    with st.sidebar:
        st.header(t("sidebar.header"))
        st.selectbox(t("preset.label"), list(PRESETS) + ["custom"], key="lab_preset",
                     format_func=lambda k: t(f"preset.{k}"), on_change=_apply_preset,
                     help=t("preset.help"))
        st.subheader(t("sec.system"))
        lattice = st.radio(t("side.lattice_type"), ["square", "hex"], horizontal=True,
                           format_func=lambda k: t(f"lattice.{k}"))
        cells = st.slider(t("side.cells"), 4, 20, 10, help=t("side.lattice_help"))
        n_atoms = Parameters(lattice=lattice, cells=cells).n_particles
        st.caption(t("side.n_caption", n=n_atoms))
        density = st.slider(t("side.density"), 0.02, 1.00, key="lab_density", step=0.01)

        st.subheader(t("sec.thermo"))
        thermostat = st.radio(t("side.ensemble"), THERMOSTAT_KEYS, index=1,
                              format_func=lambda k: t(f"thermo.{k}"))
        temperature = st.slider(t("side.temperature"), 0.05, 3.00, key="lab_temperature", step=0.05)
        ramp = st.checkbox(t("side.ramp"), disabled=thermostat == "nve")
        final_temperature = None
        if ramp and thermostat != "nve":
            final_temperature = st.slider(t("side.final_temperature"), 0.05, 3.00, 1.50, 0.05)
        tau_t = 0.1
        if thermostat == "berendsen":
            tau_t = st.select_slider(t("side.tau"), [0.02, 0.05, 0.1, 0.5, 1.0, 5.0], 0.1)

        st.subheader(t("sec.integration"))
        dt = st.select_slider(t("side.dt"), [0.001, 0.002, 0.005, 0.01, 0.02, 0.03], key="lab_dt")
        n_steps = st.number_input(t("side.n_steps"), 500, 20000, 3000, 500)
        save_every = st.number_input(t("side.save_every"), 5, 200, 20, 5)
        with st.expander(t("side.advanced")):
            r_cut = st.slider(t("side.rcut"), 2.0, 4.0, 2.5, 0.1)
            seed = st.number_input(t("side.seed"), 0, 99999, 42)
            discard = st.slider(t("side.discard"), 0.0, 0.8, 0.3, 0.05)
        start = st.button(t("side.run"), type="primary", width="stretch")

    st.title(t("nav.lab"))
    st.markdown(t("app.subtitle"))
    st.caption(t("lab.dim_note"))

    if start:
        p = Parameters(dim=2, lattice=lattice, cells=cells, density=density,
                       temperature=temperature, final_temperature=final_temperature, dt=dt,
                       n_steps=int(n_steps), r_cut=r_cut, thermostat=thermostat, tau_t=tau_t,
                       save_every=int(save_every), seed=int(seed))
        half = float(p.box.min()) / 2
        if r_cut > half:
            st.error(t("msg.rcut_error", rc=t.num(r_cut, 1), half=t.num(half, 2)))
            st.stop()
        bar = st.progress(0.0, text=t("msg.progress", pct="0 %"))
        st.session_state.lab_result = run(p, callback=lambda f: bar.progress(
            f, text=t("msg.progress", pct=f"{f * 100:.0f} %")))
        bar.empty()

    if "lab_result" not in st.session_state:
        st.info(t("msg.empty"))
        return

    r = st.session_state.lab_result
    p = r.params
    D = display(t, 2)
    sym = D.symbol
    el = el_name(t, sym)
    if r.stable:
        st.success(t("msg.done"))
    else:
        st.error(t("msg.unstable", step=r.failed_step))

    k0 = A.equilibrated_index(r, discard) * p.save_every
    T_mean = float(np.mean(r.temperature[k0:]))
    c = st.columns(4)
    c[0].metric(t("metric.particles"), p.n_particles)
    P_mean = float(np.mean(r.pressure[k0:]))
    c[1].metric(t("q.temperature"), D.fmt("temperature", T_mean, 1 if D.real else 3),
                D.other("temperature", T_mean, 3 if D.real else 1), delta_color="off")
    c[2].metric(t("q.pressure"), D.fmt("pressure", P_mean, 2 if D.real else 3),
                D.other("pressure", P_mean, 3 if D.real else 2), delta_color="off")
    c[3].metric(t("q.time"), D.fmt("time", r.time[-1], 2),
                D.other("time", r.time[-1], 2), delta_color="off")
    st.caption(t("units.equiv", el=el, items=f"{t('q.density')} ≈ {D.fmt('density', p.density)}") if D.real
               else f"{t('q.density')}: {D.fmt('density', p.density)}")
    if D.real:
        st.caption(t("units.note2d"))

    tabs = st.tabs([t(f"tab.{k}") for k in ("anim", "energy", "rdf", "msd", "speed",
                                            "theory", "worksheet", "download")])
    with tabs[0]:
        left, right = st.columns([3, 2])
        show(speed_animation_2d(t, r), left)
        with right:
            st.markdown(t("anim.box", Lx=t.num(r.box[0], 2), Ly=t.num(r.box[1], 2),
                          Ax=t.num(to_real("length", r.box[0], sym)[0], 1),
                          Ay=t.num(to_real("length", r.box[1], sym)[0], 1), el=el))
            st.markdown(t("anim.colour"))

    with tabs[1]:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=(t("energy.title_e"), t("energy.title_t"),
                                            t("energy.title_p")))
        tt = D("time", r.time)
        fig.add_trace(go.Scatter(x=tt, y=D("energy", r.kinetic_energy), name=t("energy.kin"),
                                 line=dict(color=RED)), 1, 1)
        fig.add_trace(go.Scatter(x=tt, y=D("energy", r.potential_energy), name=t("energy.pot"),
                                 line=dict(color=BLUE)), 1, 1)
        fig.add_trace(go.Scatter(x=tt, y=D("energy", r.total_energy), name=t("energy.tot"),
                                 line=dict(color="black", width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=tt, y=D("temperature", r.temperature),
                                 name=t("energy.t_inst"), line=dict(color=RED)), 2, 1)
        if p.thermostat != "nve":
            fig.add_trace(go.Scatter(x=tt, y=D("temperature", r.target_temperature),
                                     name=t("energy.t_target"),
                                     line=dict(color=GREY, dash="dash")), 2, 1)
        fig.add_trace(go.Scatter(x=tt, y=D("pressure", r.pressure), name=t("energy.p_inst"),
                                 line=dict(color=GREEN)), 3, 1)
        fig.add_vrect(x0=0, x1=float(tt[k0]), fillcolor=GREY, opacity=0.12, line_width=0,
                      annotation_text=t("energy.equil"), annotation_position="top left")
        fig.update_xaxes(title_text=D.label("time"), row=3, col=1)
        fig.update_yaxes(title_text=D.unit("energy"), row=1, col=1)
        fig.update_yaxes(title_text=D.unit("temperature"), row=2, col=1)
        fig.update_yaxes(title_text=D.unit("pressure"), row=3, col=1)
        show(style(fig, t, height=780))
        if p.thermostat == "nve":
            e = r.total_energy[k0:]
            st.markdown(t("energy.nve_note", pct=t.num(np.std(e) / abs(np.mean(e)) * 100)))
        else:
            st.markdown(t("energy.nvt_note"))

    with tabs[2]:
        x, g = A.rdf(r, discard)
        fig = go.Figure(go.Scatter(x=D("length", x), y=g, line=dict(color=BLUE, width=2)))
        fig.add_hline(y=1, line_dash="dot", line_color=GREY, annotation_text=t("rdf.ideal"))
        show(style(fig, t, xaxis_title=D.label("length"), yaxis_title="g(r)", showlegend=False))
        st.markdown(t("rdf.note", r=t.num(x[np.argmax(g)], 2), g=t.num(g.max(), 2)))

    with tabs[3]:
        lag, m = A.msd(r, discard)
        if len(lag) < 6:
            st.warning(t("msd.too_few"))
        else:
            D_coef, slope, icpt, (ta, tb) = A.diffusion_coefficient(lag, m, 2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=D("time", lag), y=D("msd", m), name=t("msd.label"),
                                     line=dict(color=BLUE)))
            tt = np.linspace(ta, tb, 20)
            fig.add_trace(go.Scatter(x=D("time", tt), y=D("msd", slope * tt + icpt),
                                     name=t("msd.fit"), line=dict(color=RED, dash="dash")))
            show(style(fig, t, xaxis_title=D.label("time"), yaxis_title=D.label("msd")))
            if D_coef < 1e-3:
                st.markdown(t("msd.solid"))
            else:
                D_real, unit = to_real("diffusion", D_coef, sym)
                mant, exp = f"{D_real:.2e}".split("e")
                st.markdown(t("msd.note", D=t.num(D_coef, 4), el=el, unit=unit,
                              Dar=f"{t.num(float(mant), 2)} × 10^{int(exp)}"))

    with tabs[4]:
        x, y, v, mb, T = A.speed_distribution(r, discard)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=D("speed", x), y=D("speed_pdf", y), name=t("speed.sim"),
                             marker_color=BLUE, opacity=0.65))
        fig.add_trace(go.Scatter(x=D("speed", v), y=D("speed_pdf", mb),
                                 name=t("speed.mb", T=D.fmt("temperature", T, 0 if D.real else 2)),
                                 line=dict(color=RED, width=2)))
        show(style(fig, t, bargap=0.02, xaxis_title=D.label("speed"),
                   yaxis_title=D.label("speed_pdf")))
        st.markdown(t("speed.note"))

    with tabs[5]:
        st.markdown(f"#### {t('theory.lj_title')}")
        st.latex(r"U(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}"
                 r" - \left(\frac{\sigma}{r}\right)^{6}\right]")
        st.markdown(t("theory.lj_text"))
        st.markdown(f"#### {t('theory.eom_title')}")
        st.latex(r"\mathbf v_i(t+\tfrac{\Delta t}{2}) = \mathbf v_i(t) + "
                 r"\tfrac{\Delta t}{2m}\mathbf F_i(t),\quad "
                 r"\mathbf r_i(t+\Delta t) = \mathbf r_i(t) + \Delta t\,"
                 r"\mathbf v_i(t+\tfrac{\Delta t}{2})")
        st.latex(r"\mathbf v_i(t+\Delta t) = \mathbf v_i(t+\tfrac{\Delta t}{2}) "
                 r"+ \tfrac{\Delta t}{2m}\mathbf F_i(t+\Delta t)")
        st.markdown(f"#### {t('theory.tp_title')}")
        st.latex(r"k_B T = \frac{1}{N_f}\sum_i m v_i^2,\qquad N_f = 2N-2")
        st.latex(r"P = \frac{1}{A}\left(N k_B T + \frac{1}{2}\sum_{i<j} "
                 r"\mathbf r_{ij}\cdot\mathbf F_{ij}\right)")
        st.markdown(f"#### {t('theory.rdf_title')}")
        st.latex(r"g(r) = \frac{\langle n(r)\rangle}{\rho\,2\pi r\,\Delta r},"
                 r"\qquad \langle |\mathbf r(t)-\mathbf r(0)|^2\rangle "
                 r"\xrightarrow{t\to\infty} 2dDt")
        st.markdown(f"#### {t('theory.units_title')}")
        st.markdown(t("theory.units_text", el=el,
                      sigma=t.num(element(sym)["sigma_nm"] * 10, 2),
                      epsK=t.num(element(sym)["epsilon_K"], 1),
                      tau=t.num(unit_scales(sym)["tau_s"] * 1e12, 3)))

    with tabs[6]:
        st.markdown(t("worksheet"))

    with tabs[7]:
        st.markdown(t("download.text"))
        a, b = st.columns(2)
        a.download_button(t("download.csv"), thermo_csv(r), "modilab_thermo.csv",
                          "text/csv", width="stretch")
        b.download_button(t("download.xyz"), trajectory_extxyz(r, sym), "modilab_trajectory.xyz",
                          "chemical/x-xyz", width="stretch")
