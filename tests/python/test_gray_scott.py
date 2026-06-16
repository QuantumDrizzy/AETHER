"""Validate Gray-Scott: a seed self-organises into pattern; uniform stays uniform."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.reaction_diffusion.gray_scott import GrayScott  # noqa: E402


def test_pattern_emerges_from_seed():
    gs = GrayScott()                       # default = maze preset (robust)
    u, v = gs.seeded_init(90, seed=0)
    u, v = gs.run(u, v, 8000)
    assert np.all(np.isfinite(v)) and 0.0 <= v.min() and v.max() <= 1.0
    assert v.std() > 0.07                  # clear spatial structure
    assert (v > 0.2).mean() > 0.15         # a substantial patterned fraction


def test_uniform_field_stays_uniform():
    gs = GrayScott()
    u = np.ones((90, 90))
    v = np.zeros((90, 90))                  # perfectly uniform, no seed
    u, v = gs.run(u, v, 4000)
    assert v.std() < 1e-9                   # no perturbation -> no pattern


def test_values_bounded():
    gs = GrayScott()
    u, v = gs.seeded_init(60, seed=1)
    u, v = gs.run(u, v, 2000)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
