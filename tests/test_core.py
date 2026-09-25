"""Basic physical correctness tests for the simulation core."""

import numpy as np
import pytest

from modilab.core import Parameters, compute_forces, initial_positions, run


def test_forces_equal_minus_energy_gradient():
    p = Parameters(lattice_side=5, density=0.6)
    rng = np.random.default_rng(1)
    pos = initial_positions(p, rng) + rng.normal(0, 0.05, (p.n_particles, 2))
    L = p.box_length
    f, _, _ = compute_forces(pos, L, p.r_cut)
    h = 1e-6
    for i, d in [(0, 0), (3, 1), (7, 0)]:
        plus, minus = pos.copy(), pos.copy()
        plus[i, d] += h
        minus[i, d] -= h
        grad = (compute_forces(plus, L, p.r_cut)[1]
                - compute_forces(minus, L, p.r_cut)[1]) / (2 * h)
        assert np.isclose(f[i, d], -grad, rtol=1e-4, atol=1e-6)


def test_total_energy_conserved_in_nve():
    p = Parameters(lattice_side=8, density=0.5, temperature=1.0, dt=0.002,
                   n_steps=1000, thermostat="nve")
    r = run(p)
    e = r.total_energy
    assert r.stable
    assert np.std(e) / abs(np.mean(e)) < 5e-3


def test_total_momentum_zero():
    r = run(Parameters(lattice_side=6, n_steps=300))
    assert np.allclose(r.velocities[-1].sum(axis=0), 0.0, atol=1e-9)


def test_thermostat_reaches_target_temperature():
    r = run(Parameters(lattice_side=8, density=0.5, temperature=1.5,
                       n_steps=2000))
    assert abs(np.mean(r.temperature[-500:]) - 1.5) < 0.1


def test_large_time_step_is_flagged_unstable():
    r = run(Parameters(lattice_side=10, density=0.9, dt=0.03, thermostat="nve"))
    assert not r.stable and r.failed_step is not None


def test_invalid_thermostat_rejected():
    with pytest.raises(ValueError):
        Parameters(thermostat="nose-hoover")
