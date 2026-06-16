"""Validate the Vicsek flocking model: order parameter and the noise transition."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.active_matter.vicsek import VicsekModel, sweep_noise  # noqa: E402


def test_order_parameter_in_range():
    m = VicsekModel(n=150, noise=1.0, seed=1)
    phi = m.run(steps=80, average_last=20)
    assert 0.0 <= phi <= 1.0


def test_low_noise_flocks():
    # little noise -> the swarm aligns (high order)
    m = VicsekModel(n=300, noise=0.1, seed=2)
    assert m.run(steps=200, average_last=60) > 0.7


def test_high_noise_disorders():
    # strong noise -> incoherent motion (low order)
    m = VicsekModel(n=300, noise=2 * np.pi, seed=3)
    assert m.run(steps=200, average_last=60) < 0.35


def test_order_decreases_with_noise():
    etas = [0.1, 1.0, 2.0, 4.0, 6.0]
    phis = sweep_noise(etas, steps=150, n=250, seed=4)
    # broadly monotone decreasing: the ordered end beats the disordered end clearly
    assert phis[0] > phis[-1] + 0.3
