"""Validate the target-seeking active swarm: assembly and damage/repair."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.active_matter.swarm_assembly import TargetSeekingSwarm, ring_target  # noqa: E402


def test_swarm_assembles_to_target():
    swarm = TargetSeekingSwarm(ring_target(12), n_agents=12, seed=0)
    assert swarm.run(steps=200) > 0.95


def test_graceful_degradation_without_spares():
    swarm = TargetSeekingSwarm(ring_target(12), n_agents=12, seed=0)
    swarm.run(steps=200)
    swarm.damage(3)
    cov = swarm.run(steps=200)
    # no spare mass -> coverage settles at the honest ceiling (remaining / sites)
    assert abs(cov - 9 / 12) < 0.1


def test_full_self_repair_with_spares():
    swarm = TargetSeekingSwarm(ring_target(12), n_agents=16, seed=1)  # 4 spares
    swarm.run(steps=200)
    swarm.damage(3)                      # 13 agents left, still >= 12 sites
    cov = swarm.run(steps=200)
    assert cov > 0.95                    # spares fill in -> full recovery


def test_coverage_in_range():
    swarm = TargetSeekingSwarm(ring_target(8), n_agents=8, seed=2)
    assert 0.0 <= swarm.coverage() <= 1.0
