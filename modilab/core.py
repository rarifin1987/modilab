"""Core of a 2D molecular dynamics engine with the Lennard-Jones potential.

All quantities are in reduced units (sigma = epsilon = m = k_B = 1).
The implementation is deliberately simple and vectorised with NumPy so
that students can read it; it is not optimised for large systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

DIM = 2
THERMOSTATS = ("nve", "berendsen", "rescale")


@dataclass
class Parameters:
    """Simulation input parameters."""

    lattice_side: int = 10            # number of particles N = lattice_side ** 2
    density: float = 0.70             # rho* = N / A
    temperature: float = 1.00         # initial (and target) T*
    final_temperature: Optional[float] = None  # if set, target T* changes linearly
    dt: float = 0.005                 # reduced time step
    n_steps: int = 3000
    r_cut: float = 2.5
    thermostat: str = "berendsen"     # one of THERMOSTATS
    tau_t: float = 0.1                # Berendsen coupling time
    save_every: int = 20              # store a configuration every n steps
    initial_noise: float = 0.05       # random lattice displacement (fraction of spacing)
    seed: int = 42

    def __post_init__(self):
        if self.thermostat not in THERMOSTATS:
            raise ValueError(f"thermostat must be one of {THERMOSTATS}")

    @property
    def n_particles(self) -> int:
        return self.lattice_side ** 2

    @property
    def box_length(self) -> float:
        return float(np.sqrt(self.n_particles / self.density))


@dataclass
class Result:
    """Simulation output."""

    params: Parameters
    L: float
    time: np.ndarray                  # time at every step
    kinetic_energy: np.ndarray        # per particle
    potential_energy: np.ndarray      # per particle
    temperature: np.ndarray
    pressure: np.ndarray
    target_temperature: np.ndarray    # NaN for NVE
    frame_time: np.ndarray            # time of every stored frame
    positions: np.ndarray             # (n_frames, N, 2), wrapped into the box
    unwrapped_positions: np.ndarray   # (n_frames, N, 2), used for MSD
    velocities: np.ndarray            # (n_frames, N, 2)
    stable: bool = True
    failed_step: Optional[int] = None

    @property
    def total_energy(self) -> np.ndarray:
        return self.kinetic_energy + self.potential_energy


def initial_positions(p: Parameters, rng: np.random.Generator) -> np.ndarray:
    """Square lattice with a small random displacement."""
    L = p.box_length
    a = L / p.lattice_side
    idx = np.arange(p.lattice_side)
    gx, gy = np.meshgrid(idx, idx, indexing="ij")
    pos = (np.column_stack([gx.ravel(), gy.ravel()]) + 0.5) * a
    pos += rng.uniform(-1, 1, pos.shape) * p.initial_noise * a
    return pos % L


def degrees_of_freedom(n: int) -> int:
    # Total momentum is set to zero, which removes DIM degrees of freedom
    return DIM * n - DIM


def instantaneous_temperature(v: np.ndarray) -> float:
    return float(np.sum(v ** 2) / degrees_of_freedom(len(v)))


def initial_velocities(p: Parameters, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(0.0, 1.0, (p.n_particles, DIM))
    v -= v.mean(axis=0)                                       # remove centre of mass motion
    v *= np.sqrt(p.temperature / instantaneous_temperature(v))  # scale to T*
    return v


def compute_forces(pos: np.ndarray, L: float, r_cut: float):
    """Lennard-Jones forces, total potential energy and virial (shifted potential).

    Returns (forces, potential_energy, virial) where
    virial = sum over pairs of r_ij . F_ij.
    """
    dr = pos[:, None, :] - pos[None, :, :]
    dr -= L * np.round(dr / L)                # minimum image convention
    r2 = np.einsum("ijk,ijk->ij", dr, dr)
    np.fill_diagonal(r2, np.inf)
    inside = r2 < r_cut ** 2

    inv_r2 = np.where(inside, 1.0 / r2, 0.0)
    inv_r6 = inv_r2 ** 3
    u_cut = 4.0 * (r_cut ** -12 - r_cut ** -6)

    u = np.where(inside, 4.0 * (inv_r6 ** 2 - inv_r6) - u_cut, 0.0)
    f_scalar = np.where(inside, 24.0 * (2.0 * inv_r6 ** 2 - inv_r6) * inv_r2, 0.0)

    forces = np.einsum("ij,ijk->ik", f_scalar, dr)
    energy = 0.5 * float(np.sum(u))
    virial = 0.5 * float(np.sum(f_scalar * np.where(inside, r2, 0.0)))
    return forces, energy, virial


def instantaneous_pressure(n: int, L: float, T: float, virial: float) -> float:
    return (n * T + virial / DIM) / (L * L)


def run(p: Parameters,
        callback: Optional[Callable[[float], None]] = None) -> Result:
    """Run an MD simulation with the velocity Verlet integrator."""
    rng = np.random.default_rng(p.seed)
    n, L, dt = p.n_particles, p.box_length, p.dt

    pos = initial_positions(p, rng)
    unwrapped = pos.copy()
    v = initial_velocities(p, rng)
    f, ep, vir = compute_forces(pos, L, p.r_cut)

    n_frames = p.n_steps // p.save_every + 1
    ek_arr = np.zeros(p.n_steps + 1)
    ep_arr = np.zeros_like(ek_arr)
    T_arr = np.zeros_like(ek_arr)
    P_arr = np.zeros_like(ek_arr)
    T0_arr = np.full_like(ek_arr, np.nan)
    pos_fr = np.zeros((n_frames, n, DIM))
    unw_fr = np.zeros_like(pos_fr)
    vel_fr = np.zeros_like(pos_fr)

    def record(i: int, T0: float):
        T = instantaneous_temperature(v)
        ek_arr[i] = 0.5 * np.sum(v ** 2) / n
        ep_arr[i] = ep / n
        T_arr[i] = T
        P_arr[i] = instantaneous_pressure(n, L, T, vir)
        T0_arr[i] = T0
        if i % p.save_every == 0:
            k = i // p.save_every
            pos_fr[k], unw_fr[k], vel_fr[k] = pos, unwrapped, v

    thermostatted = p.thermostat != "nve"
    record(0, p.temperature if thermostatted else np.nan)
    stable, failed_step = True, None
    last_step = p.n_steps
    report_every = max(1, p.n_steps // 100)

    for i in range(1, p.n_steps + 1):
        # Velocity Verlet
        v += 0.5 * dt * f
        pos += dt * v
        unwrapped += dt * v
        pos %= L
        f, ep, vir = compute_forces(pos, L, p.r_cut)
        v += 0.5 * dt * f

        # Target temperature (constant or linear ramp)
        if p.final_temperature is None:
            T0 = p.temperature
        else:
            T0 = p.temperature + (p.final_temperature - p.temperature) * i / p.n_steps

        # Thermostat
        if thermostatted:
            T = instantaneous_temperature(v)
            if p.thermostat == "berendsen":
                lam = np.sqrt(max(0.0, 1.0 + dt / p.tau_t * (T0 / T - 1.0)))
            else:  # rescale
                lam = np.sqrt(T0 / T)
            v *= lam

        if not np.all(np.isfinite(v)) or instantaneous_temperature(v) > 1e3:
            stable, failed_step, last_step = False, i, i - 1
            break

        record(i, T0 if thermostatted else np.nan)
        if callback and i % report_every == 0:
            callback(i / p.n_steps)

    m = last_step + 1
    k = last_step // p.save_every + 1
    return Result(
        params=p, L=L,
        time=np.arange(m) * dt,
        kinetic_energy=ek_arr[:m], potential_energy=ep_arr[:m],
        temperature=T_arr[:m], pressure=P_arr[:m],
        target_temperature=T0_arr[:m],
        frame_time=np.arange(k) * p.save_every * dt,
        positions=pos_fr[:k], unwrapped_positions=unw_fr[:k],
        velocities=vel_fr[:k], stable=stable, failed_step=failed_step,
    )
