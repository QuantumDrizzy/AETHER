"""Validate self-reconfiguration to a target and graceful self-repair under damage."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.programmable_matter.reconfigure import (  # noqa: E402
    ReconfigurableMatter,
    square_target,
)


def test_reconfigures_to_target():
    # from a random scatter, anneal into the target shape
    tgt = square_target(12, side=4)
    m = ReconfigurableMatter(12, tgt, seed=1)
    m.anneal(sweeps=1000, t_final=0.01)
    assert m.coverage() > 0.9          # nearly the whole shape is filled


def test_units_end_on_target():
    tgt = square_target(12, side=4)
    m = ReconfigurableMatter(12, tgt, seed=2)
    m.anneal(sweeps=1000, t_final=0.01)
    assert m.on_target_fraction() > 0.9


def test_self_repair_after_damage():
    # destroy a quarter of the units; the survivors must re-anneal so that
    # essentially all of them sit on the target (graceful degradation, no waste)
    tgt = square_target(12, side=4)        # 16 units
    m = ReconfigurableMatter(12, tgt, seed=3)
    m.anneal(sweeps=1000, t_final=0.01)
    m.damage(4)
    m.anneal(sweeps=1000, t_final=0.01)
    assert m.on_target_fraction() > 0.9    # remaining units all productive
    # coverage drops to (remaining / target) but no more
    assert m.coverage() >= (12 / 16) - 0.1


def test_energy_is_nonnegative_and_drops():
    tgt = square_target(12, side=4)
    m = ReconfigurableMatter(12, tgt, seed=4)
    e0 = m.energy()
    e1 = m.anneal(sweeps=500)
    assert e1 >= 0
    assert e1 < e0
