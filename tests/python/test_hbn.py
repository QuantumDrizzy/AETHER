"""Validate the gapped-honeycomb (hBN) model: the gap is 2*delta, graphene at delta=0."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.hbn import GappedHoneycomb  # noqa: E402


def test_gap_at_dirac_point_is_two_delta():
    h = GappedHoneycomb(delta=2.3)
    eK = h.bands(h.K_point)
    assert abs((eK[1] - eK[0]) - 2 * 2.3) < 1e-6


def test_measured_gap_matches_two_delta():
    for d in (0.5, 1.0, 2.3):
        h = GappedHoneycomb(delta=d)
        assert abs(h.measured_gap_eV() - 2 * d) < 1e-3


def test_delta_zero_recovers_graphene_gapless():
    h = GappedHoneycomb(delta=0.0)
    assert h.measured_gap_eV() < 1e-2          # Dirac point closes -> ~gapless


def test_electron_hole_symmetry():
    h = GappedHoneycomb(delta=1.5)
    for k in (np.array([0.1, 0.2]), h.K_point, np.array([0.0, 0.0])):
        em, ep = h.bands(k)
        assert abs(em + ep) < 1e-9


def test_closed_form_matches_diagonalization():
    h = GappedHoneycomb(delta=1.2)
    k = np.array([0.3, -0.4])
    em, ep = h.bands(k)
    eig = h.eigenergies(k)
    assert abs(eig[0] - em) < 1e-9 and abs(eig[1] - ep) < 1e-9
