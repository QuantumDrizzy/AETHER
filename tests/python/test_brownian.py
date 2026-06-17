"""Validate Brownian motion: Einstein relation, equipartition, fluctuation-dissipation."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.stochastic.brownian import (  # noqa: E402
    diffusion_from_msd,
    einstein_D,
    kinetic_temperature,
    mean_squared_displacement,
    simulate_overdamped,
    simulate_underdamped,
    velocity_autocorrelation,
)


def test_einstein_relation_is_exact():
    assert einstein_D(kT=2.0, gamma=4.0) == 0.5


def test_msd_is_linear_in_time():
    dt = 0.01
    x = simulate_overdamped(D=1.0, dt=dt, seed=0)
    msd = mean_squared_displacement(x)
    t = np.arange(len(msd)) * dt
    # <x^2> = 2 D t -> near-perfect linear correlation
    r = np.corrcoef(t, msd)[0, 1]
    assert r > 0.999
    assert msd[0] == 0.0


def test_diffusion_constant_recovered():
    dt = 0.01
    for D_true in (0.5, 1.5):
        x = simulate_overdamped(D=D_true, dt=dt, n_particles=6000, seed=1)
        assert abs(diffusion_from_msd(x, dt) - D_true) / D_true < 0.10


def test_equipartition():
    # <m v^2> = kT regardless of friction (converges fast with particle averaging)
    for gamma in (1.0, 4.0):
        _, v = simulate_underdamped(gamma=gamma, kT=1.0, m=1.0,
                                    n_particles=2000, n_steps=1500, seed=2)
        assert abs(kinetic_temperature(v, m=1.0) - 1.0) < 0.05


def test_vacf_zero_lag_is_thermal():
    kT, m = 1.0, 2.0
    _, v = simulate_underdamped(gamma=1.0, kT=kT, m=m,
                                n_particles=2000, n_steps=1500, seed=3)
    vacf = velocity_autocorrelation(v, max_lag=10)
    assert abs(vacf[0] - kT / m) / (kT / m) < 0.05


def test_vacf_decays():
    _, v = simulate_underdamped(gamma=2.0, kT=1.0, m=1.0,
                                n_particles=2000, n_steps=1500, seed=4)
    vacf = velocity_autocorrelation(v, max_lag=300)
    assert vacf[200] < vacf[0] * 0.2          # correlation decays away


def test_fluctuation_dissipation_integral():
    # D = integral of VACF = kT/gamma (truncation biases it slightly low)
    gamma, kT, m, dt = 2.0, 1.0, 1.0, 0.005
    _, v = simulate_underdamped(gamma=gamma, kT=kT, m=m, dt=dt,
                                n_particles=3000, n_steps=2500, seed=5)
    vacf = velocity_autocorrelation(v, max_lag=800)
    D = np.trapezoid(vacf, dx=dt)
    assert abs(D - einstein_D(kT, gamma)) / einstein_D(kT, gamma) < 0.10


def test_determinism():
    a = simulate_overdamped(D=1.0, seed=7)
    b = simulate_overdamped(D=1.0, seed=7)
    assert np.array_equal(a, b)
