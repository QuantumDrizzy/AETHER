"""Validate phonon dispersion against closed forms: branches, gap, sound speed."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.phonons.phonon_chain import (  # noqa: E402
    diatomic_dispersion,
    monatomic_closed_form,
    monatomic_dispersion,
    optical_at_gamma,
    sound_speed,
    zone_boundary_gap,
)


def test_monatomic_matches_closed_form():
    ks = np.linspace(-np.pi, np.pi, 401)
    num = monatomic_dispersion(ks, K=1.0, m=1.0, a=1.0)
    cf = monatomic_closed_form(ks, K=1.0, m=1.0, a=1.0)
    assert np.allclose(num, cf, atol=1e-12)


def test_monatomic_omega_max():
    K, m = 2.5, 1.3
    ks = np.linspace(-np.pi, np.pi, 801)
    num = monatomic_dispersion(ks, K=K, m=m, a=1.0)
    assert abs(num.max() - 2.0 * np.sqrt(K / m)) < 1e-6


def test_sound_speed_is_acoustic_slope():
    # numeric slope at small k must match a sqrt(K/m)
    K, m, a = 1.0, 1.0, 1.0
    dk = 1e-4
    w = monatomic_dispersion(np.array([dk]), K, m, a)[0]
    assert abs(w / dk - sound_speed(K, m, a)) < 1e-3


def test_group_velocity_vanishes_at_zone_boundary():
    K, m, a = 1.0, 1.0, 1.0
    kb = np.pi / a
    w = monatomic_dispersion(np.array([kb - 1e-4, kb]), K, m, a)
    vg = (w[1] - w[0]) / 1e-4
    assert abs(vg) < 1e-2          # standing wave: zero group velocity


def test_diatomic_acoustic_vanishes_optical_at_gamma():
    K, m1, m2 = 1.0, 1.0, 2.0
    ks = np.array([0.0])
    ac, op = diatomic_dispersion(ks, K, m1, m2, a=1.0)
    assert ac[0] < 1e-6          # omega = sqrt(eigenvalue); eig ~ machine eps -> ~1e-8
    assert abs(op[0] - optical_at_gamma(K, m1, m2)) < 1e-9


def test_phonon_band_gap_opens_with_mass_contrast():
    K = 1.0
    ac_top, op_bot = zone_boundary_gap(K, m1=1.0, m2=2.0)
    assert op_bot > ac_top                       # a real gap
    assert abs(ac_top - np.sqrt(2 * K / 2.0)) < 1e-12
    assert abs(op_bot - np.sqrt(2 * K / 1.0)) < 1e-12


def test_gap_closes_when_masses_equal():
    # equal masses -> diatomic is just the folded monatomic chain, bands touch
    ac_top, op_bot = zone_boundary_gap(1.0, m1=1.0, m2=1.0)
    assert abs(op_bot - ac_top) < 1e-12


def test_branch_ordering():
    ks = np.linspace(-np.pi, np.pi, 201)
    ac, op = diatomic_dispersion(ks, 1.0, 1.0, 2.0, 1.0)
    assert np.all(op >= ac - 1e-12)              # optical never below acoustic
