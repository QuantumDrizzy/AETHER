"""Validate the Vicsek phase diagram: flocking needs low noise AND enough density."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.active_matter.phase_diagram import order_at  # noqa: E402


def test_flocking_corner_is_ordered():
    assert order_at(0.2, 200, box=7.0, steps=140) > 0.8


def test_noise_destroys_order():
    assert order_at(5.5, 200, box=7.0, steps=140) < 0.35


def test_density_matters():
    dense = order_at(0.2, 200, box=7.0, steps=140)
    sparse = order_at(0.2, 12, box=7.0, steps=140)
    assert sparse < dense          # too few neighbours -> weaker alignment
