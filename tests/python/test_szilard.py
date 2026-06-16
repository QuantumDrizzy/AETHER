"""Validate the Szilard / Maxwell-demon engine against the known bounds."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.daemons.szilard import (  # noqa: E402
    binary_entropy,
    mutual_information_bits,
    simulate_engine,
    work_bound,
    work_per_bit,
)

K_B = 1.380649e-23
LN2 = math.log(2.0)


def test_work_per_bit_is_kT_ln2():
    assert abs(work_per_bit(300.0) - K_B * 300.0 * LN2) < 1e-30


def test_binary_entropy_endpoints_and_peak():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert abs(binary_entropy(0.5) - 1.0) < 1e-12


def test_mutual_information_limits():
    assert abs(mutual_information_bits(0.0) - 1.0) < 1e-12   # perfect demon: 1 bit
    assert abs(mutual_information_bits(0.5)) < 1e-12          # useless demon: 0 bits


def test_perfect_demon_extracts_one_quantum():
    r = simulate_engine(n_cycles=40000, error_p=0.0, seed=1)
    assert abs(r.mean_work_per_cycle - work_per_bit(300.0)) < 1e-23


def test_useless_demon_extracts_nothing():
    r = simulate_engine(n_cycles=40000, error_p=0.5, seed=2)
    assert abs(r.mean_work_per_cycle) < 0.05 * work_per_bit(300.0)


def test_extracted_work_equals_information_bound():
    # optimal feedback: work = k_B T ln2 * (empirical MI) -> matches the theoretical
    # bound k_B T ln2 (1 - H2(p)) within sampling, and never exceeds it
    for p in (0.0, 0.1, 0.2, 0.35):
        r = simulate_engine(n_cycles=120000, error_p=p, seed=3)
        assert r.mean_work_per_cycle <= r.info_bound + 1e-23
        assert abs(r.mean_work_per_cycle - r.info_bound) < 0.03 * work_per_bit(300.0)


def test_second_law_safe_after_erasure():
    # even a perfect demon nets <= 0 once it pays Landauer to reset its memory
    r = simulate_engine(n_cycles=40000, error_p=0.0, seed=4)
    assert r.net_work_after_erasure <= 1e-23
