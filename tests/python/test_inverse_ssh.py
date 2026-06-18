"""Validate inverse topological design: target gap + phase -> SSH hoppings (round-trip)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.inverse_design.inverse_ssh import design, forward_check  # noqa: E402
from research.electronic_structure.ssh import bulk_gap, winding_number  # noqa: E402


def test_topological_design_hits_gap_and_phase():
    t1, t2 = design(0.8, topological=True)
    chk = forward_check(t1, t2)
    assert abs(chk["gap"] - 0.8) < 1e-2
    assert chk["winding"] == 1 and chk["topological"]
    assert chk["edge_states"] == 2            # bulk-boundary correspondence


def test_trivial_design_hits_gap_and_phase():
    t1, t2 = design(0.8, topological=False)
    chk = forward_check(t1, t2)
    assert abs(chk["gap"] - 0.8) < 1e-2
    assert chk["winding"] == 0 and not chk["topological"]
    assert chk["edge_states"] == 0            # no mid-gap states in the trivial phase


def test_round_trip_recovers_chain():
    for t1_true, t2_true in [(1.0, 1.6), (1.0, 0.4), (1.0, 1.2)]:
        gap = bulk_gap(t1_true, t2_true)
        topo = winding_number(t1_true, t2_true) == 1
        t1, t2 = design(gap, topo, t1=t1_true)
        assert abs(t2 - t2_true) < 1e-9 and abs(t1 - t1_true) < 1e-9


def test_same_gap_opposite_phase_differ():
    topo = design(1.0, topological=True)
    triv = design(1.0, topological=False)
    assert topo[1] > topo[0]                  # t2 > t1 (topological)
    assert triv[1] < triv[0]                  # t1 > t2 (trivial)


def test_trivial_gap_too_large_raises():
    with pytest.raises(ValueError):
        design(2.5, topological=False, t1=1.0)   # needs t2 = 1 - 1.25 < 0


def test_nonpositive_gap_raises():
    with pytest.raises(ValueError):
        design(0.0, topological=True)
