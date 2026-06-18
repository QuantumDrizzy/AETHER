"""Validate that inverse design reports the *family* (degeneracy), honestly forward-checked."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.inverse_design.degeneracy import (  # noqa: E402
    auxetic_family_verified,
    photonic_family_verified,
    ssh_family_verified,
)
from research.inverse_design.inverse_photonic import forward_gap  # noqa: E402
from research.inverse_design.inverse_ssh import forward_check  # noqa: E402
from research.metamaterials.auxetic import poisson_ratio_12  # noqa: E402


def test_photonic_family_is_multimodal_and_all_hit_target():
    fam = photonic_family_verified(1.0, 0.30, [1.0, 1.3, 1.6, 2.0])
    assert len(fam) >= 3                                    # genuinely many-to-one
    n2s = {round(n2, 4) for _, n2, _ in fam}
    assert len(n2s) == len(fam)                            # distinct structures, not relabels
    for n1, n2, f0 in fam:
        c, w = forward_gap(n1, n2, f0)
        assert abs(c - 1.0) < 2e-2 and abs(w - 0.30) < 2e-2


def test_ssh_family_shares_gap_and_phase_across_scales():
    fam = ssh_family_verified(0.8, topological=True, t1_values=[0.5, 1.0, 2.0, 4.0])
    assert len(fam) == 4
    assert len({t1 for t1, _ in fam}) == 4                 # distinct energy scales
    for t1, t2 in fam:
        chk = forward_check(t1, t2)
        assert abs(chk["gap"] - 0.8) < 1e-9 and chk["topological"]


def test_auxetic_family_shares_poisson_ratio_across_aspects():
    fam = auxetic_family_verified(-0.5, [1.5, 2.0, 3.0])
    assert len(fam) >= 2
    thetas = {round(t, 3) for t, _ in fam}
    assert len(thetas) == len(fam)                         # each aspect -> its own angle
    for theta, hl in fam:
        assert abs(poisson_ratio_12(theta, hl) - (-0.5)) < 1e-3
        assert theta < 0                                   # auxetic => re-entrant ribs


def test_verified_filter_drops_unphysical_members():
    """The trivial SSH branch needs t1 > gap/2; too-small scales must be filtered out."""
    fam = ssh_family_verified(0.8, topological=False, t1_values=[0.2, 1.0, 2.0])
    assert all(t1 > 0.4 for t1, _ in fam)                  # t1=0.2 -> t2<0, dropped
    for t1, t2 in fam:
        assert t2 > 0 and not forward_check(t1, t2)["topological"]
