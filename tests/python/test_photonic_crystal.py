"""Validate the 1D photonic crystal: a structural band gap (Bragg)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.em_simulation.photonic_crystal import PhotonicCrystal  # noqa: E402


def test_first_gap_centred_at_design_frequency():
    pc = PhotonicCrystal(n1=1.0, n2=2.0, f0=1.0)
    lo, hi = pc.first_gap(2.0)
    assert abs((lo + hi) / 2 - 1.0) < 0.02       # quarter-wave gap centred at f0


def test_gap_widens_with_index_contrast():
    def width(n2):
        lo, hi = PhotonicCrystal(1.0, n2, 1.0).first_gap()
        return hi - lo
    assert width(3.0) > width(1.5) > 0.0


def test_uniform_medium_has_no_gap():
    assert PhotonicCrystal(2.0, 2.0, 1.0).band_gaps(3.0) == []


def test_bloch_cos_exceeds_one_in_gap():
    pc = PhotonicCrystal(1.0, 2.0, 1.0)
    assert pc.is_gap(1.0) and abs(pc.bloch_cos(1.0)) > 1.0      # design freq is in the gap
    assert not pc.is_gap(0.5)                                   # below the gap, propagating
