"""Validate the Kane-Mele model: Z2 = 1 (QSH) vs 0 (trivial), time-reversal safe."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.kane_mele import KaneMele  # noqa: E402


def test_topological_phase_is_z2_one():
    km = KaneMele(lambda_so=0.1, M=0.0)
    c_up, c_dn = km.spin_chern_numbers(36)
    assert abs(c_up) == 1 and abs(c_dn) == 1
    assert km.z2_invariant(36) == 1


def test_trivial_phase_is_z2_zero():
    km = KaneMele(lambda_so=0.1, M=1.0)
    assert km.z2_invariant(36) == 0


def test_time_reversal_total_chern_zero():
    # the two spins carry opposite Chern -> total Chern 0 (time-reversal preserved)
    c_up, c_dn = KaneMele(lambda_so=0.1, M=0.0).spin_chern_numbers(36)
    assert c_up == -c_dn


def test_boundary_matches_haldane():
    assert KaneMele(lambda_so=0.1, M=0.45).is_topological()
    assert not KaneMele(lambda_so=0.1, M=0.60).is_topological()
