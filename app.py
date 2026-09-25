"""MoDiLab: bilingual graphical interface for a virtual MD laboratory.

Run with:  streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from modilab import __version__
from modilab import analysis as A
from modilab.core import Parameters, run
from modilab.i18n import DEFAULT, LANGUAGES, Translator
from modilab.units_io import argon_tau_ps, thermo_csv, to_argon, trajectory_extxyz

st.set_page_config(page_title="MoDiLab", page_icon="⚛️", layout="wide")

BLUE, RED, GREEN, GREY = "#1F5AA6", "#C0392B", "#2E8B57", "#6B7785"

PRESETS = {
    "gas": dict(density=0.05, temperature=1.50, dt=0.005),
    "liquid": dict(density=0.70, temperature=1.00, dt=0.005),
    "solid": dict(density=0.95, temperature=0.20, dt=0.005),
}
THERMOSTAT_KEYS = ["nve", "berendsen", "rescale"]


def apply_preset():
    values = PRESETS.get(st.session_state.preset)
    if values:
        for k, v in values.items():
            st.session_state[k] = v


for k, v in {"lang": DEFAULT, "preset": "liquid", **PRESETS["liquid"]}.items():
    st.session_state.setdefault(k, v)

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.radio("Bahasa / Language", list(LANGUAGES), key="lang",
             format_func=LANGUAGES.get, horizontal=True)
    t = Translator(st.session_state.lang)

    st.header(t("sidebar.header"))
    st.selectbox(t("preset.label"), list(PRESETS) + ["custom"], key="preset",
                 format_func=lambda k: t(f"preset.{k}"),
                 on_change=apply_preset, help=t("preset.help"))

    st.subheader(t("sec.system"))
    side = st.slider(t("side.lattice"), 4, 20, 10, help=t("side.lattice_help"))
    st.caption(t("side.n_caption", n=side * side))
    density = st.slider(t("side.density"), 0.02, 1.00, key="density", step=0.01)

    st.subheader(t("sec.thermo"))
    thermostat = st.radio(t("side.ensemble"), THERMOSTAT_KEYS, index=1,
                          format_func=lambda k: t(f"thermo.{k}"))
    temperature = st.slider(t("side.temperature"), 0.05, 3.00,
                            key="temperature", step=0.05)
    ramp = st.checkbox(t("side.ramp"), disabled=thermostat == "nve")
    final_temperature = None
    if ramp and thermostat != "nve":
        final_temperature = st.slider(t("side.final_temperature"),
                                      0.05, 3.00, 1.50, 0.05)
    tau_t = 0.1
    if thermostat == "berendsen":
        tau_t = st.select_slider(t("side.tau"),
                                 [0.02, 0.05, 0.1, 0.5, 1.0, 5.0], 0.1)

    st.subheader(t("sec.integration"))
    dt = st.select_slider(t("side.dt"),
                          [0.001, 0.002, 0.005, 0.01, 0.02, 0.03], key="dt")
    n_steps = st.number_input(t("side.n_steps"), 500, 20000, 3000, 500)
    save_every = st.number_input(t("side.save_every"), 5, 200, 20, 5)

    with st.expander(t("side.advanced")):
        r_cut = st.slider(t("side.rcut"), 2.0, 4.0, 2.5, 0.1)
        seed = st.number_input(t("side.seed"), 0, 99999, 42)
        discard = st.slider(t("side.discard"), 0.0, 0.8, 0.3, 0.05)

    start = st.button(t("side.run"), type="primary", width="stretch")
    st.caption(f"MoDiLab v{__version__}")

# ------------------------------------------------------------------ Header
st.title("MoDiLab")
st.markdown(t("app.subtitle"))

if start:
    p = Parameters(lattice_side=side, density=density, temperature=temperature,
                   final_temperature=final_temperature, dt=dt,
                   n_steps=int(n_steps), r_cut=r_cut, thermostat=thermostat,
                   tau_t=tau_t, save_every=int(save_every), seed=int(seed))
    if r_cut > p.box_length / 2:
        st.error(t("msg.rcut_error", rc=t.num(r_cut, 1),
                   half=t.num(p.box_length / 2, 2)))
        st.stop()
    bar = st.progress(0.0, text=t("msg.progress", pct="0 %"))
    result = run(p, callback=lambda f: bar.progress(
        f, text=t("msg.progress", pct=f"{f * 100:.0f} %")))
    bar.empty()
    st.session_state.result = result

if "result" not in st.session_state:
    st.info(t("msg.empty"))
    st.stop()

r = st.session_state.result
p = r.params
if r.stable:
    st.success(t("msg.done"))
else:
    st.error(t("msg.unstable", step=r.failed_step))


def style(fig, height=450, **kw):
    fig.update_layout(height=height, separators=t.plotly_separators,
                      margin=dict(l=10, r=10, t=30, b=10), **kw)
    return fig


k0 = A.equilibrated_index(r, discard) * p.save_every
T_mean = float(np.mean(r.temperature[k0:]))
c = st.columns(4)
c[0].metric(t("metric.particles"), p.n_particles)
c[1].metric(t("metric.temp"), t.num(T_mean),
            t("metric.argon_K", v=t.num(to_argon("temperature", T_mean)[0], 0)),
            delta_color="off")
c[2].metric(t("metric.pressure"), t.num(float(np.mean(r.pressure[k0:]))))
c[3].metric(t("metric.time"), t.num(r.time[-1], 2),
            t("metric.argon_ps", v=t.num(to_argon("time", r.time[-1])[0], 1)),
            delta_color="off")

tabs = st.tabs([t(f"tab.{k}") for k in ("anim", "energy", "rdf", "msd", "speed",
                                        "theory", "worksheet", "download")])

# ------------------------------------------------------------------ Animation
with tabs[0]:
    stride = max(1, len(r.positions) // 150)
    idx = np.arange(0, len(r.positions), stride)
    speeds = np.linalg.norm(r.velocities, axis=-1)
    cmax = float(np.percentile(speeds, 99))
    size = max(4.0, 520 / r.L * 0.9)

    def trace(k):
        return go.Scatter(
            x=r.positions[k, :, 0], y=r.positions[k, :, 1], mode="markers",
            marker=dict(size=size, color=speeds[k], cmin=0, cmax=cmax,
                        colorscale="Viridis", line=dict(width=0.5, color="white"),
                        colorbar=dict(title=t("anim.speed_bar"))),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>")

    fig = go.Figure(data=[trace(idx[0])],
                    frames=[go.Frame(data=[trace(k)], name=str(k)) for k in idx])
    axis = dict(range=[0, r.L], showgrid=False, zeroline=False, mirror=True,
                showline=True, linecolor=GREY)
    style(fig, height=600, plot_bgcolor="#F7F9FC",
          xaxis=dict(axis, constrain="domain"),
          yaxis=dict(axis, scaleanchor="x"),
          updatemenus=[dict(
              type="buttons", direction="left", x=0.0, y=-0.02,
              xanchor="left", yanchor="top",
              buttons=[
                  dict(label=t("anim.play"), method="animate",
                       args=[None, dict(frame=dict(duration=60, redraw=True),
                                        fromcurrent=True)]),
                  dict(label=t("anim.pause"), method="animate",
                       args=[[None], dict(frame=dict(duration=0, redraw=False),
                                          mode="immediate")])])],
          sliders=[dict(
              x=0.2, len=0.78, y=-0.02, currentvalue=dict(prefix="t* = "),
              steps=[dict(method="animate", label=t.num(r.frame_time[k], 2),
                          args=[[str(k)], dict(mode="immediate",
                                               frame=dict(duration=0))])
                     for k in idx])])
    left, right = st.columns([3, 2])
    left.plotly_chart(fig, width="stretch")
    with right:
        st.markdown(t("anim.box", L=t.num(r.L, 2),
                      A=t.num(to_argon("length", r.L)[0], 1)))
        st.markdown(t("anim.colour"))

# ------------------------------------------------------------------ Energy
with tabs[1]:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=(t("energy.title_e"), t("energy.title_t"),
                                        t("energy.title_p")))
    fig.add_trace(go.Scatter(x=r.time, y=r.kinetic_energy, name=t("energy.kin"),
                             line=dict(color=RED)), 1, 1)
    fig.add_trace(go.Scatter(x=r.time, y=r.potential_energy, name=t("energy.pot"),
                             line=dict(color=BLUE)), 1, 1)
    fig.add_trace(go.Scatter(x=r.time, y=r.total_energy, name=t("energy.tot"),
                             line=dict(color="black", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=r.time, y=r.temperature, name=t("energy.t_inst"),
                             line=dict(color=RED)), 2, 1)
    if p.thermostat != "nve":
        fig.add_trace(go.Scatter(x=r.time, y=r.target_temperature,
                                 name=t("energy.t_target"),
                                 line=dict(color=GREY, dash="dash")), 2, 1)
    fig.add_trace(go.Scatter(x=r.time, y=r.pressure, name=t("energy.p_inst"),
                             line=dict(color=GREEN)), 3, 1)
    fig.add_vrect(x0=0, x1=r.time[k0], fillcolor=GREY, opacity=0.12,
                  line_width=0, annotation_text=t("energy.equil"),
                  annotation_position="top left")
    fig.update_xaxes(title_text=t("axis.time"), row=3, col=1)
    st.plotly_chart(style(fig, height=780), width="stretch")
    if p.thermostat == "nve":
        e = r.total_energy[k0:]
        st.markdown(t("energy.nve_note",
                      pct=t.num(np.std(e) / abs(np.mean(e)) * 100)))
    else:
        st.markdown(t("energy.nvt_note"))

# ------------------------------------------------------------------ g(r)
with tabs[2]:
    x, g = A.rdf(r, discard)
    fig = go.Figure(go.Scatter(x=x, y=g, line=dict(color=BLUE, width=2)))
    fig.add_hline(y=1, line_dash="dot", line_color=GREY,
                  annotation_text=t("rdf.ideal"))
    st.plotly_chart(style(fig, xaxis_title="r / σ", yaxis_title="g(r)"),
                    width="stretch")
    st.markdown(t("rdf.note", r=t.num(x[np.argmax(g)], 2), g=t.num(g.max(), 2)))

# ------------------------------------------------------------------ MSD
with tabs[3]:
    lag, m = A.msd(r, discard)
    if len(lag) < 6:
        st.warning(t("msd.too_few"))
    else:
        D, slope, icpt, (ta, tb) = A.diffusion_coefficient(lag, m)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lag, y=m, name=t("msd.label"),
                                 line=dict(color=BLUE)))
        tt = np.linspace(ta, tb, 20)
        fig.add_trace(go.Scatter(x=tt, y=slope * tt + icpt, name=t("msd.fit"),
                                 line=dict(color=RED, dash="dash")))
        st.plotly_chart(style(fig, xaxis_title=t("msd.xaxis"),
                              yaxis_title="MSD (σ²)"), width="stretch")
        if D < 1e-3:
            st.markdown(t("msd.solid"))
        else:
            D_ar, unit = to_argon("diffusion", D)
            mant, exp = f"{D_ar:.2e}".split("e")
            D_ar_txt = f"{t.num(float(mant), 2)} × 10^{int(exp)}"
            st.markdown(t("msd.note", D=t.num(D, 4), Dar=D_ar_txt, unit=unit))

# ------------------------------------------------------------------ Speeds
with tabs[4]:
    x, y, v, mb, T = A.speed_distribution(r, discard)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, name=t("speed.sim"), marker_color=BLUE,
                         opacity=0.65))
    fig.add_trace(go.Scatter(x=v, y=mb, name=t("speed.mb", T=t.num(T, 2)),
                             line=dict(color=RED, width=2)))
    st.plotly_chart(style(fig, bargap=0.02, xaxis_title=t("speed.x"),
                          yaxis_title=t("speed.y")), width="stretch")
    st.markdown(t("speed.note"))

# ------------------------------------------------------------------ Theory
with tabs[5]:
    st.markdown(f"#### {t('theory.lj_title')}")
    st.latex(r"U(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}"
             r" - \left(\frac{\sigma}{r}\right)^{6}\right]")
    st.markdown(t("theory.lj_text"))
    st.markdown(f"#### {t('theory.eom_title')}")
    st.latex(r"m\,\ddot{\mathbf r}_i = \mathbf F_i = -\nabla_i "
             r"\sum_{j\neq i} U(r_{ij})")
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
    st.markdown(t("theory.units_text", tau=t.num(argon_tau_ps(), 3)))

# ------------------------------------------------------------------ Worksheet
with tabs[6]:
    st.markdown(t("worksheet"))

# ------------------------------------------------------------------ Download
with tabs[7]:
    st.markdown(t("download.text"))
    a, b = st.columns(2)
    a.download_button(t("download.csv"), thermo_csv(r),
                      "modilab_thermo.csv", "text/csv", width="stretch")
    b.download_button(t("download.xyz"), trajectory_extxyz(r),
                      "modilab_trajectory.xyz", "chemical/x-xyz", width="stretch")
