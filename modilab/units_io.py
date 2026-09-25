"""Element parameters, conversion from reduced to real units, and data export."""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Result

KB = 1.380649e-23          # J/K
U = 1.66053906660e-27      # kg
EV = 1.602176634e-19       # J
NA = 6.02214076e23         # 1/mol


@lru_cache(maxsize=None)
def _db() -> dict:
    with open(Path(__file__).parent / "data" / "elements.json", encoding="utf-8") as fh:
        return json.load(fh)


def element_symbols() -> list[str]:
    return [k for k in _db() if not k.startswith("_")]


def element(symbol: str) -> dict:
    return _db()[symbol]


def element_source() -> dict:
    return _db()["_source"]


def unit_scales(symbol: str) -> dict:
    """SI values of the natural units for an element."""
    e = element(symbol)
    sigma = e["sigma_nm"] * 1e-9
    eps = e["epsilon_K"] * KB
    m = e["mass_u"] * U
    tau = sigma * np.sqrt(m / eps)
    return {"sigma_m": sigma, "epsilon_J": eps, "mass_kg": m, "tau_s": tau,
            "epsilon_eV": eps / EV, "velocity_m_s": np.sqrt(eps / m)}


def to_real(quantity: str, value, symbol: str):
    """Convert a reduced value to real units. Returns (value, unit)."""
    s = unit_scales(symbol)
    if quantity == "temperature":
        return value * element(symbol)["epsilon_K"], "K"
    if quantity == "length":
        return value * s["sigma_m"] * 1e10, "Å"
    if quantity == "energy":
        return value * s["epsilon_eV"] * 1e3, "meV"
    if quantity == "time":
        return value * s["tau_s"] * 1e12, "ps"
    if quantity == "velocity":
        return value * s["velocity_m_s"], "m/s"
    if quantity == "length2":
        return value * (s["sigma_m"] * 1e10) ** 2, "Å²"
    if quantity == "density2d":
        return value / (s["sigma_m"] * 1e9) ** 2, "nm⁻²"
    if quantity == "pressure2d":
        return value * s["epsilon_J"] / s["sigma_m"] ** 2 * 1e3, "mN/m"
    if quantity == "speed_pdf":
        return value / s["velocity_m_s"], "s/m"
    if quantity == "diffusion":
        return value * (s["sigma_m"] * 1e2) ** 2 / s["tau_s"], "cm²/s"
    if quantity == "density3d":
        return value * s["mass_kg"] * 1e3 / (s["sigma_m"] * 1e2) ** 3, "g/cm³"
    if quantity == "pressure3d":
        return value * s["epsilon_J"] / s["sigma_m"] ** 3 / 1e6, "MPa"
    raise ValueError(f"unknown quantity: {quantity}")


def thermo_table(r: Result) -> pd.DataFrame:
    return pd.DataFrame({
        "time": r.time, "kinetic_energy": r.kinetic_energy,
        "potential_energy": r.potential_energy, "total_energy": r.total_energy,
        "temperature": r.temperature, "target_temperature": r.target_temperature,
        "pressure": r.pressure,
    })


def thermo_csv(r: Result) -> bytes:
    return thermo_table(r).to_csv(index=False).encode("utf-8")


def trajectory_extxyz(r: Result, symbol: str = "Ar") -> bytes:
    """Trajectory in extended XYZ format, readable by OVITO and ASE."""
    buf = io.StringIO()
    b = np.ones(3)
    b[: r.dim] = r.box
    pbc = "T T T" if r.dim == 3 else "T T F"
    for k, pos in enumerate(r.positions):
        speed = np.linalg.norm(r.velocities[k], axis=1)
        buf.write(f"{len(pos)}\n")
        buf.write(f'Lattice="{b[0]:.6f} 0 0 0 {b[1]:.6f} 0 0 0 {b[2]:.6f}" '
                  'Properties=species:S:1:pos:R:3:speed:R:1 '
                  f'pbc="{pbc}" Time={r.frame_time[k]:.5f}\n')
        for xyz, s in zip(pos, speed):
            z = xyz[2] if r.dim == 3 else 0.0
            buf.write(f"{symbol} {xyz[0]:.6f} {xyz[1]:.6f} {z:.6f} {s:.6f}\n")
    return buf.getvalue().encode("utf-8")
