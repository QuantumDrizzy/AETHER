"""Validate Game of Life: still life, oscillator, glider translation, self-organisation."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.cellular_automata.game_of_life import (  # noqa: E402
    blinker,
    block,
    glider,
    run,
    step,
    translated_equal,
)


def test_block_is_still_life():
    assert np.array_equal(step(block()), block())


def test_blinker_period_two():
    b = blinker()
    assert not np.array_equal(step(b), b)
    assert np.array_equal(run(b, 2), b)


def test_glider_translates():
    g = glider(20)
    assert translated_equal(g, run(g, 4), (1, 1))      # period 4, moves by (1,1)
    assert not np.array_equal(run(g, 4), g)            # it is NOT stationary


def test_self_organises_to_low_density():
    rng = np.random.default_rng(0)
    init = (rng.random((100, 100)) < 0.30).astype(np.int8)
    final = run(init, 150)
    d = final.mean()
    assert 0.0 < d < 0.15          # collapses well below the 0.30 seeding, not extinct
