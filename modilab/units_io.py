"""Conversion from reduced units to argon, and data export (CSV, extended XYZ for OVITO)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .core import Result

# Lennard-Jones parameters of argon commonly used in textbooks
ARGON = {"sigma_angstrom": 3.405, "epsilon_over_kB_K": 119.8, "mass_u": 39.948}
_KB = 1.380649e-23
_U = 1.66053907e-27


def argon_tau_ps() -> float:
    """Reduced time unit tau = sigma * sqrt(m / epsilon), in picoseconds."""
    s = ARGON["sigma_angstrom"] * 1e-10
    m = ARGON["mass_u"] * _U
    e = ARGON["epsilon_over_kB_K"] * _KB
    return s * np.sqrt(m / e) * 1e12


def to_argon(quantity: str, value: float) -> tuple[float, str]:
    """Convert a reduced value to argon units."""
    if quantity == "temperature":
        return value * ARGON["epsilon_over_kB_K"], "K"
    if quantity == "length":
        return value * ARGON["sigma_angstrom"], "Å"
    if quantity == "time":
        return value * argon_tau_ps(), "ps"
    if quantity == "diffusion":
        s_cm = ARGON["sigma_angstrom"] * 1e-8
        return value * s_cm ** 2 / (argon_tau_ps() * 1e-12), "cm²/s"
    raise ValueError(f"unknown quantity: {quantity}")


def thermo_table(r: Result) -> pd.DataFrame:
    return pd.DataFrame({
        "time": r.time,
        "kinetic_energy": r.kinetic_energy,
        "potential_energy": r.potential_energy,
        "total_energy": r.total_energy,
        "temperature": r.temperature,
        "target_temperature": r.target_temperature,
        "pressure": r.pressure,
    })


def thermo_csv(r: Result) -> bytes:
    return thermo_table(r).to_csv(index=False).encode("utf-8")


def trajectory_extxyz(r: Result) -> bytes:
    """Trajectory in extended XYZ format, readable by OVITO and ASE."""
    buf = io.StringIO()
    L = r.L
    for k, pos in enumerate(r.positions):
        speed = np.linalg.norm(r.velocities[k], axis=1)
        buf.write(f"{len(pos)}\n")
        buf.write(f'Lattice="{L:.6f} 0 0 0 {L:.6f} 0 0 0 1.0" '
                  'Properties=species:S:1:pos:R:3:speed:R:1 '
                  f'pbc="T T F" Time={r.frame_time[k]:.5f}\n')
        for (x, y), s in zip(pos, speed):
            buf.write(f"Ar {x:.6f} {y:.6f} 0.000000 {s:.6f}\n")
    return buf.getvalue().encode("utf-8")
