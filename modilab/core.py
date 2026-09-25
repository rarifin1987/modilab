"""Molecular dynamics engine for Lennard-Jones atoms in 2 or 3 dimensions.

All quantities are in reduced units (sigma = epsilon = m = k_B = 1).
The implementation is vectorised with NumPy and kept deliberately simple so
that students can read it; it is not optimised for large systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

THERMOSTATS = ("nve", "berendsen", "rescale")
LATTICES = {2: ("square", "hex"), 3: ("sc", "fcc")}
VELOCITY_INITS = ("gaussian", "equal_speed")


# --------------------------------------------------------------------------- lattices
def lattice_counts(lattice: str, cells: int) -> tuple[int, tuple[int, ...]]:
    """Number of atoms and cell repetitions for a lattice with `cells` along x."""
    if lattice == "square":
        return cells ** 2, (cells, cells)
    if lattice == "hex":
        ny = max(1, round(cells / np.sqrt(3)))
        return 2 * cells * ny, (cells, ny)
    if lattice == "sc":
        return cells ** 3, (cells,) * 3
    if lattice == "fcc":
        return 4 * cells ** 3, (cells,) * 3
    raise ValueError(f"unknown lattice: {lattice}")


def build_lattice(lattice: str, cells: int, density: float):
    """Perfect lattice positions and box lengths for a given reduced density."""
    n, reps = lattice_counts(lattice, cells)
    if lattice == "square":
        a = density ** -0.5
        cell, basis = np.array([a, a]), np.array([[0.0, 0.0]])
    elif lattice == "hex":
        a = np.sqrt(2.0 / (np.sqrt(3.0) * density))
        cell = np.array([a, a * np.sqrt(3.0)])
        basis = np.array([[0.0, 0.0], [0.5 * a, 0.5 * a * np.sqrt(3.0)]])
    elif lattice == "sc":
        a = density ** (-1 / 3)
        cell, basis = np.array([a, a, a]), np.array([[0.0, 0.0, 0.0]])
    else:  # fcc
        a = (4.0 / density) ** (1 / 3)
        cell = np.array([a, a, a])
        basis = a * np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    grids = np.meshgrid(*[np.arange(r) for r in reps], indexing="ij")
    origins = np.column_stack([g.ravel() for g in grids]) * cell
    pos = (origins[:, None, :] + basis[None, :, :]).reshape(-1, len(cell))
    pos += 0.25 * a / max(1, len(basis))           # keep atoms off the box edge
    box = cell * np.array(reps)
    assert len(pos) == n
    return pos % box, box, a


def nearest_neighbour_distance(lattice: str, a: float) -> float:
    return {"square": a, "hex": a, "sc": a, "fcc": a / np.sqrt(2)}[lattice]


# --------------------------------------------------------------------------- parameters
@dataclass
class Parameters:
    """Simulation input parameters."""

    dim: int = 2
    lattice: str = "square"            # see LATTICES
    cells: int = 10                    # lattice repetitions along x
    density: float = 0.70              # rho* = N / V
    temperature: float = 1.00          # target T* (and initial T* unless set below)
    initial_temperature: Optional[float] = None  # if set, initial velocities use this T*
    final_temperature: Optional[float] = None  # if set, target T* ramps linearly
    dt: float = 0.005
    n_steps: int = 3000
    r_cut: float = 2.5
    thermostat: str = "berendsen"      # see THERMOSTATS
    tau_t: float = 0.1                 # Berendsen coupling time
    save_every: int = 20
    initial_noise: float = 0.05        # random lattice displacement (fraction of spacing)
    velocity_init: str = "gaussian"    # see VELOCITY_INITS
    seed: int = 42

    def __post_init__(self):
        if self.dim not in LATTICES:
            raise ValueError("dim must be 2 or 3")
        if self.lattice not in LATTICES[self.dim]:
            raise ValueError(f"lattice for dim={self.dim} must be one of {LATTICES[self.dim]}")
        if self.thermostat not in THERMOSTATS:
            raise ValueError(f"thermostat must be one of {THERMOSTATS}")
        if self.velocity_init not in VELOCITY_INITS:
            raise ValueError(f"velocity_init must be one of {VELOCITY_INITS}")

    @property
    def n_particles(self) -> int:
        return lattice_counts(self.lattice, self.cells)[0]

    @property
    def box(self) -> np.ndarray:
        return build_lattice(self.lattice, self.cells, self.density)[1]


@dataclass
class Result:
    """Simulation output."""

    params: Parameters
    box: np.ndarray
    lattice_constant: float
    time: np.ndarray
    kinetic_energy: np.ndarray        # per particle
    potential_energy: np.ndarray      # per particle
    temperature: np.ndarray
    pressure: np.ndarray
    target_temperature: np.ndarray    # NaN for NVE
    frame_time: np.ndarray
    positions: np.ndarray             # (n_frames, N, dim), wrapped
    unwrapped_positions: np.ndarray   # (n_frames, N, dim)
    velocities: np.ndarray            # (n_frames, N, dim)
    stable: bool = True
    failed_step: Optional[int] = None

    @property
    def total_energy(self) -> np.ndarray:
        return self.kinetic_energy + self.potential_energy

    @property
    def dim(self) -> int:
        return self.params.dim

    @property
    def volume(self) -> float:
        return float(np.prod(self.box))


# --------------------------------------------------------------------------- physics
def degrees_of_freedom(n: int, dim: int) -> int:
    return dim * n - dim               # total momentum is fixed at zero


def instantaneous_temperature(v: np.ndarray) -> float:
    n, dim = v.shape
    return float(np.sum(v ** 2) / degrees_of_freedom(n, dim))


def initial_velocities(n: int, dim: int, T: float, mode: str,
                       rng: np.random.Generator) -> np.ndarray:
    if mode == "equal_speed":
        # Every atom gets the same speed in a random direction, a state far from
        # equilibrium. Atoms come in pairs with opposite velocities so that the
        # total momentum is exactly zero (if N is odd, the last atom is at rest).
        half = rng.normal(size=(n // 2, dim))
        half /= np.linalg.norm(half, axis=1, keepdims=True)
        v = np.zeros((n, dim))
        v[: 2 * (n // 2)] = np.concatenate([half, -half])
        v = v[rng.permutation(n)]
    else:
        v = rng.normal(size=(n, dim))
        v -= v.mean(axis=0)
    return v * np.sqrt(T / instantaneous_temperature(v))


def lj_potential(r, r_cut: Optional[float] = None, shifted: bool = False):
    """Lennard-Jones pair energy U(r) in reduced units."""
    r = np.asarray(r, dtype=float)
    u = 4.0 * (r ** -12 - r ** -6)
    if shifted and r_cut is not None:
        u = np.where(r < r_cut, u - 4.0 * (r_cut ** -12 - r_cut ** -6), 0.0)
    return u


def lj_force(r):
    """Magnitude of the Lennard-Jones pair force F(r) = -dU/dr (positive = repulsive)."""
    r = np.asarray(r, dtype=float)
    return 24.0 * (2.0 * r ** -13 - r ** -7)


def compute_forces(pos: np.ndarray, box, r_cut: float):
    """Forces, total potential energy and virial (sum of r_ij . F_ij over pairs)."""
    box = np.asarray(box, dtype=float)
    dr = pos[:, None, :] - pos[None, :, :]
    dr -= box * np.round(dr / box)             # minimum image convention
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


def instantaneous_pressure(n: int, volume: float, T: float, virial: float, dim: int) -> float:
    return (n * T + virial / dim) / volume


# --------------------------------------------------------------------------- MD loop
def run(p: Parameters, callback: Optional[Callable[[float], None]] = None) -> Result:
    """Run an MD simulation with the velocity Verlet integrator."""
    rng = np.random.default_rng(p.seed)
    dt, dim = p.dt, p.dim
    pos, box, a = build_lattice(p.lattice, p.cells, p.density)
    n = len(pos)
    pos = (pos + rng.uniform(-1, 1, pos.shape) * p.initial_noise * a) % box
    volume = float(np.prod(box))
    unwrapped = pos.copy()
    T_start = p.temperature if p.initial_temperature is None else p.initial_temperature
    v = initial_velocities(n, dim, T_start, p.velocity_init, rng)
    f, ep, vir = compute_forces(pos, box, p.r_cut)

    n_frames = p.n_steps // p.save_every + 1
    ek_arr = np.zeros(p.n_steps + 1)
    ep_arr, T_arr, P_arr = np.zeros_like(ek_arr), np.zeros_like(ek_arr), np.zeros_like(ek_arr)
    T0_arr = np.full_like(ek_arr, np.nan)
    pos_fr = np.zeros((n_frames, n, dim))
    unw_fr, vel_fr = np.zeros_like(pos_fr), np.zeros_like(pos_fr)

    def record(i: int, T0: float):
        T = instantaneous_temperature(v)
        ek_arr[i] = 0.5 * np.sum(v ** 2) / n
        ep_arr[i] = ep / n
        T_arr[i] = T
        P_arr[i] = instantaneous_pressure(n, volume, T, vir, dim)
        T0_arr[i] = T0
        if i % p.save_every == 0:
            k = i // p.save_every
            pos_fr[k], unw_fr[k], vel_fr[k] = pos, unwrapped, v

    thermostatted = p.thermostat != "nve"
    record(0, p.temperature if thermostatted else np.nan)
    stable, failed_step, last_step = True, None, p.n_steps
    report_every = max(1, p.n_steps // 100)

    for i in range(1, p.n_steps + 1):
        v += 0.5 * dt * f
        pos += dt * v
        unwrapped += dt * v
        pos %= box
        f, ep, vir = compute_forces(pos, box, p.r_cut)
        v += 0.5 * dt * f

        if p.final_temperature is None:
            T0 = p.temperature
        else:
            T0 = p.temperature + (p.final_temperature - p.temperature) * i / p.n_steps

        if thermostatted:
            T = instantaneous_temperature(v)
            if p.thermostat == "berendsen":
                lam = np.sqrt(max(0.0, 1.0 + dt / p.tau_t * (T0 / T - 1.0)))
            else:
                lam = np.sqrt(T0 / T)
            v *= lam

        if not np.all(np.isfinite(v)) or instantaneous_temperature(v) > 1e3:
            stable, failed_step, last_step = False, i, i - 1
            break

        record(i, T0 if thermostatted else np.nan)
        if callback and i % report_every == 0:
            callback(i / p.n_steps)

    m, k = last_step + 1, last_step // p.save_every + 1
    return Result(
        params=p, box=box, lattice_constant=a,
        time=np.arange(m) * dt,
        kinetic_energy=ek_arr[:m], potential_energy=ep_arr[:m],
        temperature=T_arr[:m], pressure=P_arr[:m], target_temperature=T0_arr[:m],
        frame_time=np.arange(k) * p.save_every * dt,
        positions=pos_fr[:k], unwrapped_positions=unw_fr[:k], velocities=vel_fr[:k],
        stable=stable, failed_step=failed_step,
    )


# --------------------------------------------------------------------------- two-atom model
def dimer(r0: float = 1.5, v0: float = 0.0, dt: float = 0.01, n_steps: int = 2000,
          method: str = "verlet") -> dict:
    """Two Lennard-Jones atoms (mass 1 each) moving along a line.

    The relative coordinate r obeys mu * r'' = F(r) with reduced mass mu = 1/2.
    method: "euler" (explicit Euler) or "verlet" (velocity Verlet).
    """
    mu = 0.5
    r = np.empty(n_steps + 1)
    v = np.empty(n_steps + 1)
    r[0], v[0] = r0, v0
    for i in range(n_steps):
        a = lj_force(r[i]) / mu
        if method == "euler":
            r[i + 1] = r[i] + dt * v[i]
            v[i + 1] = v[i] + dt * a
        else:
            vh = v[i] + 0.5 * dt * a
            r[i + 1] = r[i] + dt * vh
            v[i + 1] = vh + 0.5 * dt * lj_force(r[i + 1]) / mu
        if not np.isfinite(r[i + 1]) or r[i + 1] <= 0.5:
            r[i + 2:] = np.nan
            v[i + 2:] = np.nan
            r[i + 1] = v[i + 1] = np.nan
            break
    energy = 0.5 * mu * v ** 2 + lj_potential(r)
    return {"t": np.arange(n_steps + 1) * dt, "r": r, "v": v, "energy": energy}
