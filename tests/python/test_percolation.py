"""Validate 2D site percolation: the spanning transition near p_c ~ 0.593."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.criticality.percolation import (  # noqa: E402
    estimate_pc,
    has_spanning_cluster,
    spanning_probability,
)


def test_below_threshold_does_not_span():
    assert spanning_probability(64, 0.40, trials=60) < 0.1


def test_above_threshold_spans():
    assert spanning_probability(64, 0.80, trials=60) > 0.9


def test_estimated_pc_matches_known():
    pc = estimate_pc(size=64, trials=60)
    assert 0.56 <= pc <= 0.63          # known 2D square-site threshold ~0.5927


def test_full_and_empty_grids():
    assert has_spanning_cluster(np.ones((10, 10), dtype=bool))
    assert not has_spanning_cluster(np.zeros((10, 10), dtype=bool))
