"""Validate the 2D Ising Monte Carlo against the Onsager critical point."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.criticality.ising2d import (  # noqa: E402
    T_C_ONSAGER,
    critical_temperature,
    simulate,
)


def test_ordered_below_tc():
    m, _ = simulate(L=24, T=1.6, equil=250, measure=300, seed=0)
    assert m > 0.9


def test_disordered_above_tc():
    m, _ = simulate(L=24, T=3.2, equil=250, measure=300, seed=0)
    assert m < 0.3


def test_susceptibility_peaks_near_onsager():
    Tc = critical_temperature(L=24, Ts=np.linspace(1.8, 3.0, 13), equil=300, measure=400)
    # finite-size χ peak sits a little above the L→∞ Onsager value 2.269
    assert abs(Tc - T_C_ONSAGER) < 0.25


def test_magnetization_drops_through_transition():
    m_lo, _ = simulate(L=24, T=2.0, equil=250, measure=300, seed=1)
    m_hi, _ = simulate(L=24, T=2.6, equil=250, measure=300, seed=1)
    assert m_lo > m_hi
