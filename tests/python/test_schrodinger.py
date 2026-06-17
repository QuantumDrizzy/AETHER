"""Validate the 1D Schrodinger solver: QHO ladder, box levels, barrier tunnelling."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.schrodinger import (  # noqa: E402
    _rect_barrier_grid,
    barrier_transmission_analytic,
    box_levels,
    harmonic_levels,
    solve,
    transmission_transfer_matrix,
)


def test_harmonic_oscillator_ladder():
    x = np.linspace(-12, 12, 2400)
    E, _ = solve(x, 0.5 * x ** 2, n_states=6)
    assert np.allclose(E, harmonic_levels(6, omega=1.0), atol=2e-3)


def test_infinite_square_well_levels():
    L = 1.0
    x = np.linspace(0, L, 1201)[1:-1]          # interior points = hard walls
    E, _ = solve(x, np.zeros_like(x), n_states=4)
    assert np.allclose(E, box_levels(4, L), rtol=2e-3)


def test_wavefunctions_are_normalised():
    x = np.linspace(-12, 12, 2000)
    dx = x[1] - x[0]
    _, psi = solve(x, 0.5 * x ** 2, n_states=3)
    for j in range(3):
        assert abs(np.sum(psi[:, j] ** 2) * dx - 1.0) < 1e-6


def test_ground_state_has_no_nodes():
    x = np.linspace(-12, 12, 2000)
    _, psi = solve(x, 0.5 * x ** 2, n_states=3)
    g = psi[:, 0]
    sign_changes = np.sum(np.diff(np.sign(g[np.abs(g) > 1e-6])) != 0)
    assert sign_changes == 0                    # nodeless ground state
    # first excited state has exactly one node
    e1 = psi[:, 1]
    nodes1 = np.sum(np.diff(np.sign(e1[np.abs(e1) > 1e-3])) != 0)
    assert nodes1 == 1


def test_tunnelling_matches_analytic():
    V0, a = 4.0, 1.0
    x, V = _rect_barrier_grid(V0, a, n=4000)
    dx = x[1] - x[0]
    for E in (1.0, 2.0, 4.0, 6.0, 8.0):
        ta = barrier_transmission_analytic(E, V0, a)
        tn = transmission_transfer_matrix(E, V, dx)
        assert abs(ta - tn) < 5e-3


def test_tunnelling_below_barrier_is_partial():
    V0, a = 5.0, 1.0
    # below the barrier: 0 < T < 1 (classically forbidden, quantum tunnelling)
    T = barrier_transmission_analytic(1.0, V0, a)
    assert 0.0 < T < 1.0


def test_above_barrier_resonance_is_unit():
    # resonance when q a = pi: T = 1 exactly (V0=4, a=1 -> E = V0 + pi^2/2)
    V0, a = 4.0, 1.0
    E_res = V0 + np.pi ** 2 / (2.0 * a ** 2)
    assert abs(barrier_transmission_analytic(E_res, V0, a) - 1.0) < 1e-9
