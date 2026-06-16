"""Validate the Haldane model: Chern number ±1 (topological) vs 0 (trivial)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.haldane import Haldane  # noqa: E402


def test_topological_phase_has_unit_chern():
    h = Haldane(t=1.0, t2=0.1, phi=np.pi / 2, M=0.0)
    assert h.band_gap() > 1e-3                 # gapped (by the complex NNN term)
    assert abs(h.chern_number(n_k=36)) == 1     # |C| = 1


def test_trivial_phase_has_zero_chern():
    h = Haldane(t=1.0, t2=0.1, phi=np.pi / 2, M=1.0)   # large mass -> trivial
    assert h.band_gap() > 1e-3
    assert h.chern_number(n_k=36) == 0


def test_phase_boundary_matches_analytic():
    # boundary at |M| = 3√3 t2 |sinφ| ≈ 0.5196 for t2=0.1, φ=π/2
    below = Haldane(t2=0.1, phi=np.pi / 2, M=0.45)
    above = Haldane(t2=0.1, phi=np.pi / 2, M=0.60)
    assert below.is_topological() and abs(below.chern_number(36)) == 1
    assert (not above.is_topological()) and above.chern_number(36) == 0


def test_sign_flips_with_phase():
    c_plus = Haldane(t2=0.1, phi=np.pi / 2, M=0.0).chern_number(36)
    c_minus = Haldane(t2=0.1, phi=-np.pi / 2, M=0.0).chern_number(36)
    assert c_plus == -c_minus and abs(c_plus) == 1
