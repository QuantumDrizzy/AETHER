"""Validate elementary CA: Rule 90 closed form + fractal dim, Rule 30 randomness."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.cellular_automata.wolfram import (  # noqa: E402
    classify,
    evolve,
    rule30_center_column,
    rule90_dimension,
    rule90_live_cells,
    rule_table,
    shannon_entropy_bits,
    step,
)


def test_rule_table_extremes():
    assert rule_table(0).sum() == 0          # rule 0: everything dies
    assert rule_table(255).sum() == 8        # rule 255: everything lives
    # rule 90 = XOR of neighbours, independent of centre
    t = rule_table(90)
    for left in (0, 1):
        for right in (0, 1):
            idx0 = (left << 2) | (0 << 1) | right
            idx1 = (left << 2) | (1 << 1) | right
            assert t[idx0] == (left ^ right)
            assert t[idx1] == (left ^ right)


def test_rule90_closed_form_matches_simulation():
    # live cells in row n from a single seed == 2**popcount(n) (Pascal mod 2)
    for n in (1, 2, 3, 4, 7, 8, 15, 16, 31):
        st = evolve(90, 4 * n + 3, n + 1)
        assert int(st[n].sum()) == rule90_live_cells(n)


def test_rule90_dimension_is_sierpinski():
    d = rule90_dimension(power=9)
    assert abs(d - np.log2(3)) < 0.12        # log2(3) ~ 1.585, box-counting is noisy


def test_rule30_center_column_is_balanced_and_aperiodic():
    col = rule30_center_column(4000)
    assert abs(col.mean() - 0.5) < 0.05          # unbiased bits
    assert shannon_entropy_bits(col) > 0.99      # ~1 bit of entropy
    # not a short cycle: no period <= 64 over the tail
    tail = col[-512:]
    for p in range(1, 65):
        assert not np.array_equal(tail[p:], tail[:-p] if p else tail)


def test_determinism():
    a = evolve(110, 101, 100)
    b = evolve(110, 101, 100)
    assert np.array_equal(a, b)


def test_step_periodic_boundary():
    row = np.zeros(5, dtype=np.uint8)
    row[0] = 1
    # rule 90: neighbours XOR; wrap-around must light up both edges' neighbours
    nxt = step(row, rule_table(90))
    assert nxt[1] == 1 and nxt[4] == 1 and nxt[0] == 0


def test_classification_poles():
    assert classify(0).startswith("I")          # everything dies -> homogeneous
    assert classify(30).startswith("III")       # chaotic
    assert not classify(110).startswith("I (")  # complex/non-trivial, not dead
