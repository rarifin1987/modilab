"""Analysis of simulation output in 2 or 3 dimensions."""

from __future__ import annotations

import numpy as np

from .core import Result


def equilibrated_index(result: Result, discard_fraction: float) -> int:
    """Index of the first stored frame after the equilibration part is discarded."""
    n = len(result.positions)
    return min(n - 2, max(0, int(discard_fraction * n)))


def _shell_measure(edges: np.ndarray, dim: int) -> np.ndarray:
    if dim == 2:
        return np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def rdf(result: Result, discard_fraction: float = 0.3, n_bins: int = 100,
        frames: slice | None = None):
    """Radial distribution function g(r)."""
    box = result.box
    sel = result.positions[equilibrated_index(result, discard_fraction):] \
        if frames is None else result.positions[frames]
    n = sel.shape[1]
    edges = np.linspace(0, box.min() / 2, n_bins + 1)
    hist = np.zeros(n_bins)
    iu = np.triu_indices(n, k=1)
    for pos in sel:
        dr = pos[:, None, :] - pos[None, :, :]
        dr -= box * np.round(dr / box)
        r = np.sqrt(np.sum(dr ** 2, axis=-1))[iu]
        hist += np.histogram(r, bins=edges)[0]
    centres = 0.5 * (edges[1:] + edges[:-1])
    ideal = _shell_measure(edges, result.dim) * (n / result.volume) * n / 2 * len(sel)
    return centres, hist / ideal


def coordination_number(r: np.ndarray, g: np.ndarray, density: float, dim: int) -> np.ndarray:
    """Running coordination number n(r) = rho * integral of g(r') dV'."""
    dr = r[1] - r[0]
    shell = 2 * np.pi * r if dim == 2 else 4 * np.pi * r ** 2
    return density * np.cumsum(g * shell * dr)


def first_minimum_after_peak(r: np.ndarray, g: np.ndarray) -> float:
    """Position of the first minimum of g(r) after its first peak."""
    i_peak = int(np.argmax(g))
    tail = g[i_peak:]
    # smooth a little to avoid noise
    k = np.ones(3) / 3
    tail_s = np.convolve(tail, k, mode="same")
    for i in range(1, len(tail_s) - 1):
        if tail_s[i] < tail_s[i - 1] and tail_s[i] <= tail_s[i + 1] and tail_s[i] < 1.0:
            return float(r[i_peak + i])
    return float(r[min(len(r) - 1, i_peak + len(tail) // 3)])


def msd(result: Result, discard_fraction: float = 0.3):
    """Mean square displacement averaged over multiple time origins."""
    k0 = equilibrated_index(result, discard_fraction)
    x = result.unwrapped_positions[k0:]
    x = x - x.mean(axis=1, keepdims=True)
    max_lag = len(x) // 2
    values = np.zeros(max_lag)
    for lag in range(1, max_lag):
        d = x[lag:] - x[:-lag]
        values[lag] = np.mean(np.sum(d ** 2, axis=-1))
    frame_dt = result.params.dt * result.params.save_every
    return np.arange(max_lag) * frame_dt, values


def diffusion_coefficient(t, msd_values, dim: int, start: float = 0.2, end: float = 0.8):
    """Einstein relation MSD = 2 d D t, fitted over the middle part of the curve."""
    i0 = max(1, int(start * len(t)))
    i1 = max(i0 + 2, int(end * len(t)))
    slope, intercept = np.polyfit(t[i0:i1], msd_values[i0:i1], 1)
    return slope / (2 * dim), slope, intercept, (t[i0], t[i1 - 1])


def maxwell_boltzmann_speed(v: np.ndarray, T: float, dim: int) -> np.ndarray:
    """Maxwell-Boltzmann speed distribution (m = k_B = 1) in 2 or 3 dimensions."""
    if dim == 2:
        return (v / T) * np.exp(-v ** 2 / (2 * T))
    return 4 * np.pi * v ** 2 * (2 * np.pi * T) ** -1.5 * np.exp(-v ** 2 / (2 * T))


def speed_histogram(velocities: np.ndarray, bins):
    speeds = np.linalg.norm(velocities, axis=-1).ravel()
    density, edges = np.histogram(speeds, bins=bins, density=True)
    return 0.5 * (edges[1:] + edges[:-1]), density


def speed_distribution(result: Result, discard_fraction: float = 0.3, n_bins: int = 40):
    """Speed histogram after equilibration and the Maxwell-Boltzmann curve at mean T."""
    k0 = equilibrated_index(result, discard_fraction)
    vmax = float(np.percentile(np.linalg.norm(result.velocities, axis=-1), 99.9)) * 1.1
    centres, density = speed_histogram(result.velocities[k0:], np.linspace(0, vmax, n_bins + 1))
    T = float(np.mean(result.temperature[k0 * result.params.save_every:]))
    v = np.linspace(0, vmax, 200)
    return centres, density, v, maxwell_boltzmann_speed(v, T, result.dim), T
