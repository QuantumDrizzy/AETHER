"""Validate the Kuramoto synchronization transition."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.active_matter.kuramoto import Kuramoto, sweep_coupling  # noqa: E402


def test_incoherent_below_threshold():
    r = Kuramoto(n=500, K=0.5, seed=0).run(steps=1500)
    assert r < 0.2          # weak coupling -> drift, only the 1/sqrt(N) floor


def test_synchronized_above_threshold():
    r = Kuramoto(n=500, K=4.0, seed=0).run(steps=1500)
    assert r > 0.8          # strong coupling -> macroscopic sync


def test_order_increases_with_coupling():
    Ks = np.array([0.5, 1.5, 2.5, 4.0])
    r = sweep_coupling(Ks, n=400, steps=1400)
    assert all(r[i] < r[i + 1] for i in range(len(r) - 1))


def test_onset_near_critical_coupling():
    Ks = np.linspace(0.0, 4.0, 17)
    r = sweep_coupling(Ks, n=500, steps=1500)
    onset = Ks[int(np.argmin(np.abs(r - 0.5)))]
    assert 1.4 < onset < 2.3       # Gaussian K_c ~ 1.6 (finite-N shifts it up a touch)
