"""Validate inverse photonic-crystal design: round-trip recovery + design-from-spec."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.inverse_design.inverse_photonic import (  # noqa: E402
    analytic_inverse,
    forward_gap,
    numeric_inverse,
)


def test_forward_gap_centres_at_f0():
    c, _w = forward_gap(1.0, 2.0, 1.0)
    assert abs(c - 1.0) < 1e-2                  # quarter-wave gap centres at the design freq


def test_analytic_inverse_round_trip():
    # known structure -> its gap -> invert -> recover the structure
    for n2_true, f0_true in [(2.0, 1.0), (3.0, 1.5), (1.6, 0.8)]:
        c, w = forward_gap(1.0, n2_true, f0_true)
        _n1, n2, f0 = analytic_inverse(c, w)
        assert abs(n2 - n2_true) < 0.02
        assert abs(f0 - f0_true) < 0.02


def test_numeric_inverse_recovers_structure():
    for n2_true, f0_true in [(2.0, 1.0), (2.5, 1.2)]:
        c, w = forward_gap(1.0, n2_true, f0_true)
        (_n1, n2, f0), achieved = numeric_inverse(c, w)
        assert abs(n2 - n2_true) < 0.02 and abs(f0 - f0_true) < 0.02
        assert abs(achieved[0] - c) < 1e-2 and abs(achieved[1] - w) < 1e-2


def test_design_from_spec_hits_target():
    target_c, target_w = 1.0, 0.30
    (_n1, _n2, _f0), achieved = numeric_inverse(target_c, target_w)
    assert abs(achieved[0] - target_c) < 1e-2
    assert abs(achieved[1] - target_w) < 1e-2


def test_higher_bandwidth_needs_more_contrast():
    _, n2_lo, _ = analytic_inverse(1.0, 0.20)
    _, n2_hi, _ = analytic_inverse(1.0, 0.50)
    assert n2_hi > n2_lo > 1.0


def test_zero_bandwidth_means_no_contrast():
    _, n2, _ = analytic_inverse(1.0, 0.0)
    assert abs(n2 - 1.0) < 1e-9                 # no gap width -> no index contrast
