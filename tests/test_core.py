"""Physical correctness tests for the simulation engine."""

import numpy as np
import pytest

from modilab import analysis as A
from modilab.core import (Parameters, build_lattice, compute_forces, dimer,
                          lj_force, lj_potential, nearest_neighbour_distance, run)


@pytest.mark.parametrize("lattice,cells,dim", [("square", 5, 2), ("fcc", 2, 3)])
def test_forces_equal_minus_energy_gradient(lattice, cells, dim):
    rng = np.random.default_rng(1)
    pos, box, _ = build_lattice(lattice, cells, 0.6)
    pos = pos + rng.normal(0, 0.05, pos.shape)
    f, _, _ = compute_forces(pos, box, 2.5)
    h = 1e-6
    for i, d in [(0, 0), (3, 1), (5, dim - 1)]:
        plus, minus = pos.copy(), pos.copy()
        plus[i, d] += h
        minus[i, d] -= h
        grad = (compute_forces(plus, box, 2.5)[1] - compute_forces(minus, box, 2.5)[1]) / (2 * h)
        assert np.isclose(f[i, d], -grad, rtol=1e-4, atol=1e-6)


def test_pair_force_is_minus_derivative_of_potential():
    r = np.linspace(0.95, 2.5, 50)
    h = 1e-6
    assert np.allclose(lj_force(r), -(lj_potential(r + h) - lj_potential(r - h)) / (2 * h),
                       rtol=1e-5)


@pytest.mark.parametrize("lattice,dim,cells", [("square", 2, 8), ("fcc", 3, 3)])
def test_lattices_have_requested_density(lattice, dim, cells):
    pos, box, _ = build_lattice(lattice, cells, 0.8)
    assert np.isclose(len(pos) / np.prod(box), 0.8)


@pytest.mark.parametrize("lattice,cells,dim", [("hex", 8, 2), ("fcc", 3, 3)])
def test_nearest_neighbour_distance_of_perfect_lattice(lattice, cells, dim):
    pos, box, a = build_lattice(lattice, cells, 0.9)
    dr = pos[:, None] - pos[None]
    dr -= box * np.round(dr / box)
    r = np.linalg.norm(dr, axis=-1)
    np.fill_diagonal(r, np.inf)
    assert np.isclose(r.min(), nearest_neighbour_distance(lattice, a))


@pytest.mark.parametrize("dim,lattice,cells", [(2, "square", 8), (3, "fcc", 3)])
def test_total_energy_conserved_in_nve(dim, lattice, cells):
    p = Parameters(dim=dim, lattice=lattice, cells=cells, density=0.5 if dim == 2 else 0.8,
                   temperature=1.0, dt=0.002, n_steps=800, thermostat="nve")
    r = run(p)
    assert r.stable
    assert np.std(r.total_energy) / abs(np.mean(r.total_energy)) < 5e-3


def test_total_momentum_zero_3d():
    r = run(Parameters(dim=3, lattice="fcc", cells=3, density=0.8, n_steps=200))
    assert np.allclose(r.velocities[-1].sum(axis=0), 0.0, atol=1e-9)


def test_thermostat_reaches_target_temperature():
    r = run(Parameters(cells=8, density=0.5, temperature=1.5, n_steps=2000))
    assert abs(np.mean(r.temperature[-500:]) - 1.5) < 0.1


def test_equal_speed_initialisation_relaxes_to_maxwell_boltzmann():
    p = Parameters(cells=10, density=0.5, temperature=1.0, thermostat="nve",
                   velocity_init="equal_speed", n_steps=3000)
    r = run(p)
    s0 = np.linalg.norm(r.velocities[0], axis=1)
    assert np.std(s0) < 1e-10                      # all speeds equal at t = 0
    x, dens, v, mb, T = A.speed_distribution(r, 0.5)
    assert np.max(np.abs(dens - A.maxwell_boltzmann_speed(x, T, 2))) < 0.25


def test_fcc_solid_first_peak_near_nearest_neighbour():
    p = Parameters(dim=3, lattice="fcc", cells=3, density=1.0, temperature=0.3, n_steps=600)
    r = run(p)
    x, g = A.rdf(r, 0.3)
    assert abs(x[np.argmax(g)] - nearest_neighbour_distance("fcc", r.lattice_constant)) < 0.08


def test_verlet_conserves_dimer_energy_better_than_euler():
    e_v = dimer(1.5, 0.0, 0.01, 2000, "verlet")["energy"]
    e_e = dimer(1.5, 0.0, 0.01, 2000, "euler")["energy"]
    drift = lambda e: np.nanmax(np.abs(e - e[0]))
    assert drift(e_v) < 1e-2
    assert drift(e_e) > 50 * drift(e_v)


def test_large_time_step_is_flagged_unstable():
    r = run(Parameters(cells=10, density=0.9, dt=0.03, thermostat="nve"))
    assert not r.stable and r.failed_step is not None


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        Parameters(thermostat="nose-hoover")
    with pytest.raises(ValueError):
        Parameters(dim=3, lattice="hex")
