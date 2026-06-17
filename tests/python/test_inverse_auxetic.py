"""Validate inverse auxetic design: target Poisson's ratio -> rib angle (round-trip)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.inverse_design.inverse_auxetic import analytic_inverse, numeric_inverse  # noqa: E402
from research.metamaterials.auxetic import poisson_ratio_12  # noqa: E402


def test_analytic_round_trip():
    for th_true, hl in [(30.0, 1.0), (-25.0, 2.0), (-40.0, 2.0), (45.0, 1.5)]:
        nu = poisson_ratio_12(th_true, hl)
        assert abs(analytic_inverse(nu, hl) - th_true) < 0.5


def test_numeric_round_trip():
    for th_true, hl in [(-25.0, 2.0), (35.0, 1.0)]:
        nu = poisson_ratio_12(th_true, hl)
        assert abs(numeric_inverse(nu, hl) - th_true) < 0.5


def test_design_from_spec_hits_target():
    th = analytic_inverse(-0.5, 2.0)
    assert abs(poisson_ratio_12(th, 2.0) - (-0.5)) < 1e-3


def test_auxetic_target_needs_reentrant_angle():
    assert analytic_inverse(-0.5, 2.0) < 0        # negative ν -> θ < 0 (re-entrant)
    assert analytic_inverse(+0.8, 1.0) > 0        # positive ν -> θ > 0 (normal)
