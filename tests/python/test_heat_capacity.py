"""Validate phonon heat capacity: Dulong-Petit, Debye T^3, Einstein freeze-out."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.phonons.heat_capacity import (  # noqa: E402
    _einstein_function,
    debye_heat_capacity,
    debye_lowT_T3,
    einstein_heat_capacity,
    heat_capacity_from_spectrum,
)


def test_einstein_function_limits():
    assert abs(_einstein_function(np.array([0.0]))[0] - 1.0) < 1e-12   # classical
    assert _einstein_function(np.array([50.0]))[0] < 1e-15             # frozen


def test_spectrum_dulong_petit_high_T():
    # each mode contributes k_B at high T -> C_v -> number of modes
    omegas = np.array([1.0, 2.0, 3.0, 5.0])
    cv = heat_capacity_from_spectrum(omegas, T=1e4)
    assert abs(cv - len(omegas)) < 1e-2


def test_spectrum_freezes_and_is_monotonic():
    omegas = np.array([1.0, 2.0, 3.0])
    temps = [0.05, 0.2, 1.0, 5.0, 50.0]
    cvs = [heat_capacity_from_spectrum(omegas, T) for T in temps]
    assert cvs[0] < 0.1                       # frozen at low T
    assert all(b >= a - 1e-9 for a, b in zip(cvs, cvs[1:]))   # monotone increasing


def test_debye_dulong_petit_high_T():
    cv = debye_heat_capacity(20.0, theta_D=1.0, n_modes=3)[0]
    assert abs(cv - 3.0) < 1e-2               # 3 N k_B


def test_debye_T3_law_low_T():
    T = 0.02
    num = debye_heat_capacity(T, theta_D=1.0, n_modes=3)[0]
    ana = debye_lowT_T3(T, theta_D=1.0, n_modes=3)[0]
    assert abs(num - ana) / ana < 0.02        # T^3 law holds at low T


def test_einstein_high_T_matches_dulong_petit():
    cv = einstein_heat_capacity(20.0, theta_E=1.0, n_modes=3)[0]
    assert abs(cv - 3.0) < 1e-2


def test_einstein_freezes_faster_than_debye():
    # the historical failure: Einstein drops exponentially, Debye as T^3
    Tlow = 0.05
    e = einstein_heat_capacity(Tlow, theta_E=1.0)[0]
    d = debye_heat_capacity(Tlow, theta_D=1.0)[0]
    assert e < d
    assert e >= 0.0 and d > 0.0


def test_zero_temperature():
    assert heat_capacity_from_spectrum(np.array([1.0, 2.0]), 0.0) == 0.0
    assert debye_heat_capacity(0.0)[0] == 0.0
    assert einstein_heat_capacity(0.0)[0] == 0.0
