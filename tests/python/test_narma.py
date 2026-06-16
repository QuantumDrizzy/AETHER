"""Validate the NARMA-10 reservoir benchmark: an optimum below the edge, chaos above."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.reservoir_computing.narma import narma10, narma_nrmse, sweep_spectral_radius  # noqa: E402


def test_narma_series_is_bounded():
    u = np.random.default_rng(0).uniform(0, 0.5, 500)
    y = narma10(u)
    assert np.all(np.isfinite(y)) and y.max() < 2.0


def test_optimum_is_intermediate_not_chaotic():
    radii = np.linspace(0.1, 1.3, 13)
    nr = sweep_spectral_radius(radii, reservoir_size=200, seed=0)
    best_rho = radii[int(np.argmin(nr))]
    assert 0.2 <= best_rho <= 0.95          # optimum below the edge, not at rho->0

def test_chaos_destroys_performance():
    nr_opt = narma_nrmse(0.6, seed=0)
    nr_chaos = narma_nrmse(1.2, seed=0)
    assert nr_chaos > 2.0 * nr_opt          # past the edge the echo-state property is lost


def test_too_stable_forgets():
    nr_opt = narma_nrmse(0.6, seed=0)
    nr_stable = narma_nrmse(0.1, seed=0)
    assert nr_opt < nr_stable               # a too-damped reservoir forgets
