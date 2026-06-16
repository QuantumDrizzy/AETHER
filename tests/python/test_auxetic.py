"""Validate the auxetic honeycomb Poisson's ratio against Gibson-Ashby limits."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.metamaterials.auxetic import (  # noqa: E402
    is_auxetic,
    poisson_ratio_12,
    poisson_ratio_21,
)


def test_regular_hexagon_poisson_is_one():
    # θ = +30°, h = l : the classic regular honeycomb has ν = +1
    assert abs(poisson_ratio_12(30.0, 1.0) - 1.0) < 1e-9


def test_reentrant_is_auxetic():
    # θ = -30°, h/l = 2 : re-entrant ribs -> ν = -1 (negative => auxetic)
    assert abs(poisson_ratio_12(-30.0, 2.0) - (-1.0)) < 1e-9
    assert is_auxetic(-30.0, 2.0)


def test_normal_geometry_not_auxetic():
    assert not is_auxetic(30.0, 1.0)
    assert poisson_ratio_12(20.0, 1.5) > 0.0


def test_reciprocity_product_is_one():
    for theta, hl in [(25.0, 1.0), (-20.0, 2.0), (40.0, 1.5)]:
        assert abs(poisson_ratio_12(theta, hl) * poisson_ratio_21(theta, hl) - 1.0) < 1e-9


def test_sign_follows_rib_direction():
    # positive rib angle -> ν > 0 ; negative (re-entrant) -> ν < 0 (valid h/l)
    assert poisson_ratio_12(15.0, 2.0) > 0.0
    assert poisson_ratio_12(-15.0, 2.0) < 0.0
