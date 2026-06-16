"""Validate that three different substrates each show an order->disorder transition."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.criticality.universal import (  # noqa: E402
    ising_order,
    reconfig_order,
    transition_point,
    vicsek_order,
)


def test_ising_orders_low_T_disorders_high_T():
    assert ising_order(0.3) > 0.8
    assert ising_order(3.0) < 0.4


def test_vicsek_orders_low_noise_disorders_high_noise():
    assert vicsek_order(0.2) > 0.7
    assert vicsek_order(2 * np.pi) < 0.35


def test_reconfig_orders_low_T_disorders_high_T():
    assert reconfig_order(0.05) > 0.7      # cold: forms the shape
    assert reconfig_order(5.0) < 0.4       # hot: cannot hold the shape


def test_transition_point_interpolates():
    controls = np.array([0.0, 1.0, 2.0, 3.0])
    orders = np.array([1.0, 0.8, 0.2, 0.0])
    xc = transition_point(controls, orders, level=0.5)
    assert 1.0 < xc < 2.0
