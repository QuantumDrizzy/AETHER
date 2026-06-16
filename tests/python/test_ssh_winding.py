"""Validate the SSH winding number and bulk-boundary correspondence."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.ssh import count_edge_states, winding_number  # noqa: E402


def test_winding_trivial_phase():
    assert winding_number(1.0, 0.5) == 0       # t1 > t2 -> trivial


def test_winding_topological_phase():
    assert winding_number(0.5, 1.0) == 1       # t2 > t1 -> topological
    assert winding_number(1.0, 1.5) == 1


def test_bulk_boundary_correspondence():
    # the bulk invariant predicts the boundary: an open chain has 2*W edge states
    for t1, t2 in [(1.0, 0.5), (0.5, 1.0), (1.0, 1.5)]:
        assert count_edge_states(25, t1, t2) == 2 * winding_number(t1, t2)
