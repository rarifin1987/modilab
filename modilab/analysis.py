"""Analysis of simulation output: g(r), MSD, diffusion coefficient, speed distribution."""

from __future__ import annotations

import numpy as np

from .core import DIM, Result


def equilibrated_index(result: Result, discard_fraction: float) -> int:
    """Index of the first stored frame after the equilibration part is discarded."""
    n = len(result.positions)
    return min(n - 2, max(0, int(discard_fraction * n)))


def rdf(result: Result, discard_fraction: float = 0.3, n_bins: int = 100):
    """Radial distribution function g(r) for a 2D system."""
    L = result.L
    frames = result.positions[equilibrated_index(result, discard_fraction):]
    n = frames.shape[1]
    edges = np.linspace(0, L / 2, n_bins + 1)
    hist = np.zeros(n_bins)
    iu = np.triu_indices(n, k=1)
    for pos in frames:
        dr = pos[:, None, :] - pos[None, :, :]
        dr -= L * np.round(dr / L)
        r = np.sqrt(np.sum(dr ** 2, axis=-1))[iu]
        hist += np.histogram(r, bins=edges)[0]
    centres = 0.5 * (edges[1:] + edges[:-1])
    ring_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    ideal = ring_area * (n / L ** 2) * n / 2 * len(frames)
    return centres, hist / ideal


def msd(result: Result, discard_fraction: float = 0.3):
    """Mean square displacement averaged over multiple time origins."""
    k0 = equilibrated_index(result, discard_fraction)
    x = result.unwrapped_positions[k0:]
    x = x - x.mean(axis=1, keepdims=True)   # remove centre of mass drift
    max_lag = len(x) // 2
    values = np.zeros(max_lag)
    for lag in range(1, max_lag):
        d = x[lag:] - x[:-lag]
        values[lag] = np.mean(np.sum(d ** 2, axis=-1))
    frame_dt = result.params.dt * result.params.save_every
    return np.arange(max_lag) * frame_dt, values


def diffusion_coefficient(t: np.ndarray, msd_values: np.ndarray,
                          start: float = 0.2, end: float = 0.8):
    """Einstein relation MSD = 2 d D t, fitted over the middle part of the curve."""
    i0 = max(1, int(start * len(t)))
    i1 = max(i0 + 2, int(end * len(t)))
    slope, intercept = np.polyfit(t[i0:i1], msd_values[i0:i1], 1)
    return slope / (2 * DIM), slope, intercept, (t[i0], t[i1 - 1])


def speed_distribution(result: Result, discard_fraction: float = 0.3, n_bins: int = 40):
    """Speed histogram and the 2D Maxwell-Boltzmann curve at the mean temperature."""
    k0 = equilibrated_index(result, discard_fraction)
    speeds = np.linalg.norm(result.velocities[k0:], axis=-1).ravel()
    density, edges = np.histogram(speeds, bins=n_bins, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    T = float(np.mean(result.temperature[k0 * result.params.save_every:]))
    v = np.linspace(0, edges[-1], 200)
    mb = (v / T) * np.exp(-v ** 2 / (2 * T))
    return centres, density, v, mb, T
