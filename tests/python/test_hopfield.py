"""Validate Hopfield associative memory and the storage-capacity collapse."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.neuro.hopfield import Hopfield, capacity_scan, random_patterns  # noqa: E402


def test_few_patterns_are_stable():
    rng = np.random.default_rng(0)
    pat = random_patterns(5, 200, rng)          # alpha = 0.025
    net = Hopfield(200).store(pat)
    for mu in range(5):
        assert np.array_equal(net.update(pat[mu]), pat[mu])   # exact fixed point


def test_recall_from_noisy_pattern():
    rng = np.random.default_rng(1)
    pat = random_patterns(4, 200, rng)
    net = Hopfield(200).store(pat)
    target = pat[0].copy()
    noisy = target.copy()
    flip = rng.choice(200, size=30, replace=False)   # corrupt 15% of bits
    noisy[flip] *= -1
    out = net.recall(noisy, steps=6)
    assert np.mean(out * target) > 0.95              # converges back to the memory


def test_capacity_collapse():
    a, frac, ov = capacity_scan(N=300, alphas=np.array([0.04, 0.25]), trials=3)
    assert frac[0] > 0.9          # well below capacity: patterns stable
    assert frac[1] < 0.2          # well above capacity: collapsed


def test_capacity_crossover_near_alpha_c():
    a = np.linspace(0.04, 0.30, 14)
    _, frac, _ = capacity_scan(N=300, alphas=a, trials=3)
    crossover = a[int(np.argmin(np.abs(frac - 0.5)))]
    assert 0.09 < crossover < 0.17     # near the AGS alpha_c ~ 0.138
