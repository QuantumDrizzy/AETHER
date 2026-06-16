"""Validate the Watts-Strogatz small-world transition."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.networks.small_world import (  # noqa: E402
    avg_path_length,
    clustering_coefficient,
    sweep_rewiring,
    watts_strogatz,
)


def test_ring_is_clustered_and_long():
    g = watts_strogatz(200, 6, 0.0)
    assert clustering_coefficient(g) > 0.5      # ring lattice C = 3(k-2)/4(k-1) = 0.6
    assert avg_path_length(g) > 10               # long paths around the ring


def test_random_graph_is_unclustered_and_short():
    g = watts_strogatz(200, 6, 1.0, seed=1)
    assert clustering_coefficient(g) < 0.1
    assert avg_path_length(g) < 5                # short paths


def test_small_world_window():
    # a little rewiring: paths already short, clustering still high
    C, L = sweep_rewiring(np.array([0.05]), n=200, k=6)
    assert L[0] < 0.5 and C[0] > 0.7


def test_path_length_drops_with_rewiring():
    C, L = sweep_rewiring(np.array([0.0, 0.1, 1.0]), n=200, k=6)
    assert L[0] > L[1] > L[2]
