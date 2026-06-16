"""Validate 1D Anderson localization: participation ratio collapses with disorder."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.anderson import (  # noqa: E402
    mean_participation_ratio,
    participation_ratios,
    sweep_disorder,
)


def test_pr_decreases_with_disorder():
    prs = sweep_disorder([0.5, 1.0, 2.0, 4.0, 8.0], L=300, trials=4)
    assert all(prs[i] > prs[i + 1] for i in range(len(prs) - 1))


def test_weak_disorder_is_extended():
    assert mean_participation_ratio(300, 0.5, trials=4) > 50.0


def test_strong_disorder_is_localized():
    assert mean_participation_ratio(300, 8.0, trials=4) < 6.0


def test_pr_in_valid_range():
    pr = participation_ratios(200, 2.0, seed=0)
    assert np.all(pr >= 1.0 - 1e-6) and np.all(pr <= 200 + 1e-6)
