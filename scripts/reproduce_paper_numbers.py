"""Reproduce every number reported in section 4 of the MoDiLab paper.

Run from the repository root:  python scripts/reproduce_paper_numbers.py
"""
import sys, time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modilab.core import Parameters, run, dimer, build_lattice
from modilab import analysis as A

print("== A. NVE energy conservation ==")
for dim, lat, cells, rho in ((2, "square", 10, 0.70), (3, "fcc", 3, 0.80)):
    for dt in (0.001, 0.002, 0.005, 0.01):
        nsteps = int(10.0 / dt)
        r = run(Parameters(dim=dim, lattice=lat, cells=cells, density=rho, temperature=1.0,
                           dt=dt, n_steps=nsteps, thermostat="nve", save_every=max(1, nsteps // 200), seed=1))
        e = r.total_energy
        rel = np.std(e) / abs(np.mean(e))
        slope = np.polyfit(r.time, e, 1)[0]
        print(f"{dim}D N={r.params.n_particles} dt={dt}: rel.std={rel:.2e} drift/t={slope:.2e} stable={r.stable} Tmean={r.temperature.mean():.3f}")

print("== B. Dimer energy error vs dt (t_total = 20) ==")
for m in ("verlet", "euler"):
    errs = []
    dts = (0.02, 0.01, 0.005, 0.0025)
    for dt in dts:
        d = dimer(1.5, 0.0, dt, int(20 / dt), m)
        errs.append(np.nanmax(np.abs(d["energy"] - d["energy"][0])))
    p = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    print(m, ["%.2e" % e for e in errs], "slope=%.2f" % p)

print("== C. Structure ==")
PH = {2: {"solid": (0.90, 0.30), "liquid": (0.75, 1.00), "gas": (0.10, 1.50)},
      3: {"solid": (1.00, 0.50), "liquid": (0.80, 1.00), "gas": (0.05, 1.50)}}
for dim, lat, cells in ((2, "hex", 12), (3, "fcc", 4)):
    for ph, (rho, T) in PH[dim].items():
        box = build_lattice(lat, cells, rho)[1]; rc = float(min(2.5, 0.49 * box.min()))
        r = run(Parameters(dim=dim, lattice=lat, cells=cells, density=rho, temperature=T, r_cut=rc,
                           n_steps=2000 if dim == 2 else 1500, save_every=20, seed=3))
        x, g = A.rdf(r, 0.4, 120); n = A.coordination_number(x, g, rho, dim); rm = A.first_minimum_after_peak(x, g)
        print(f"{dim}D N={r.params.n_particles} {ph} rho={rho} T={r.temperature[len(r.temperature)//2:].mean():.3f} peak={x[np.argmax(g)]:.2f} gmax={g.max():.2f} rmin={rm:.2f} CN={np.interp(rm, x, n):.1f} rc={rc:.2f}")

print("== D. Maxwell-Boltzmann relaxation (NVE, equal speeds) ==")
for dim, lat, cells in ((2, "square", 12), (3, "fcc", 3)):
    r = run(Parameters(dim=dim, lattice=lat, cells=cells, density=0.5, temperature=1.0, thermostat="nve",
                       velocity_init="equal_speed", n_steps=1500, save_every=10, dt=0.005, seed=5))
    sp = np.linalg.norm(r.velocities, axis=-1); vmax = np.percentile(sp, 99.5) * 1.15; bins = np.linspace(0, vmax, 26)
    dev = []
    for k in range(len(r.positions)):
        Tk = np.sum(r.velocities[k]**2) / (dim * len(r.velocities[k]) - dim)
        x, h = A.speed_histogram(r.velocities[k], bins)
        dev.append(np.mean(np.abs(h - A.maxwell_boltzmann_speed(x, Tk, dim))))
    dev = np.array(dev); late = dev[len(dev)//2:].mean()
    k_half = np.argmax(dev < dev[0] * 0.5); k_late = np.argmax(dev < 1.5 * late)
    print(f"{dim}D N={r.params.n_particles}: dev0={dev[0]:.3f} late={late:.3f} t(half)={r.frame_time[k_half]:.2f} t(<1.5*late)={r.frame_time[k_late]:.2f} T_final={r.temperature[-300:].mean():.3f}")

print("== E. Berendsen thermostat ==")
for tau in (0.05, 0.1, 1.0):
    r = run(Parameters(cells=8, density=0.6, temperature=0.8, initial_temperature=2.0, tau_t=tau, n_steps=3000, save_every=50, seed=11))
    T = r.temperature; half = len(T)//2
    k = np.argmax(np.abs(T - 0.8) < 0.1 * (2.0 - 0.8))
    print(f"tau={tau}: <T>_2nd half={T[half:].mean():.3f} rel.fluct={np.std(T[half:])/T[half:].mean()*100:.2f}% t_90%={r.time[k]:.2f}")

print("== F. Cost per step ==")
for dim, lat, cells in ((2, "square", 10), (2, "square", 15), (2, "square", 20), (3, "fcc", 3), (3, "fcc", 4), (3, "fcc", 5)):
    p = Parameters(dim=dim, lattice=lat, cells=cells, density=0.8, n_steps=200, save_every=50)
    t0 = time.time(); run(p); print(f"{dim}D N={p.n_particles}: {(time.time()-t0)/200*1e3:.1f} ms/step")
  
