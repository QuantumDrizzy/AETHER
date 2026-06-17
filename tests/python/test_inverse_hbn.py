"""Validate inverse hBN design: target (gap, band top) -> (Δ, t), round-trip."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.inverse_design.inverse_hbn import (  # noqa: E402
    analytic_inverse,
    f_max,
    forward_gap_top,
)


def test_f_max_is_honeycomb_constant():
    assert abs(f_max() - 3.0) < 0.05            # max |structure factor| ~ 3


def test_round_trip_recovers_parameters():
    for d_true, t_true in [(2.3, 2.3), (1.0, 2.0), (3.0, 1.5)]:
        gap, top = forward_gap_top(d_true, t_true)
        d, t = analytic_inverse(gap, top)
        assert abs(d - d_true) < 1e-2
        assert abs(t - t_true) < 1e-2


def test_gap_is_twice_delta():
    d, _t = analytic_inverse(2.0, 5.0)
    assert abs(d - 1.0) < 1e-9                   # Δ = gap/2 exactly


def test_design_from_spec_hits_target():
    d, t = analytic_inverse(2.0, 5.0)
    gap, top = forward_gap_top(d, t)
    assert abs(gap - 2.0) < 1e-2 and abs(top - 5.0) < 1e-2


def test_unreachable_top_raises():
    import pytest
    with pytest.raises(ValueError):
        analytic_inverse(4.0, 1.0)               # E_top below the gap edge (Δ=2 > 1)
