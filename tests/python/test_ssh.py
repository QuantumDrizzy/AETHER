"""Validation of the SSH model (ADR-0001 capstone — real topological physics on
the general tight-binding solver).

- Bulk gap = 2|t1 - t2|, and closes at t1 = t2.
- Bulk-boundary correspondence: a finite open chain has exactly 2 near-zero edge
  states in the topological phase (t2 > t1) and 0 in the trivial phase (t1 > t2).
- Chiral symmetry: the finite spectrum is symmetric about E = 0.

Run standalone:  python tests/python/test_ssh.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.ssh import (  # noqa: E402
    bulk_gap,
    count_edge_states,
    finite_spectrum,
)


def test_bulk_gap_matches_formula():
    for t1, t2 in [(1.0, 0.6), (1.0, 0.2), (0.7, 1.3)]:
        assert np.isclose(bulk_gap(t1, t2), 2.0 * abs(t1 - t2), atol=2e-3)


def test_bulk_gap_closes_when_hoppings_equal():
    assert bulk_gap(1.0, 1.0) < 2e-3


def test_bulk_boundary_correspondence():
    # Topological phase (t2 > t1): exactly two near-zero edge states.
    assert count_edge_states(25, t1=0.5, t2=1.0) == 2
    # Trivial phase (t1 > t2): clean gap, no mid-gap states.
    assert count_edge_states(25, t1=1.0, t2=0.5) == 0


def test_chiral_symmetry():
    """SSH has sublattice symmetry -> spectrum symmetric about 0."""
    e = np.sort(finite_spectrum(20, t1=0.7, t2=1.0))
    assert np.allclose(e, -e[::-1], atol=1e-9)


def _run_standalone() -> int:
    tests = [
        test_bulk_gap_matches_formula,
        test_bulk_gap_closes_when_hoppings_equal,
        test_bulk_boundary_correspondence,
        test_chiral_symmetry,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
