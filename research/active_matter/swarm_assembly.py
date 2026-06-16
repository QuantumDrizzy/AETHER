"""Target-seeking active swarm — self-assembly and self-repair, continuous space.

The active-matter + programmable-matter fusion, and the closest thing here to the
Columbia "robot metabolism": N self-propelled agents in continuous 2D, each
assigned (greedily, by distance) to a site of a target shape, steering toward it at
constant speed with angular noise. Coverage = fraction of target sites that have
their agent within a capture radius.

The point is **damage with redundancy**: with a few spare agents, destroying some
units frees their sites, the survivors (including spares) re-assign, and coverage
returns all the way to 1.0 — the swarm heals because it has reserve mass and a
reassignment rule, not a fixed wiring. Without spares it degrades gracefully to
the honest ceiling (remaining / sites).
"""

from __future__ import annotations

import numpy as np


class TargetSeekingSwarm:
    def __init__(self, targets: np.ndarray, n_agents: int | None = None,
                 box: float = 10.0, speed: float = 0.15, noise: float = 0.05,
                 capture_radius: float = 0.4, seed: int = 0):
        self.targets = np.asarray(targets, dtype=float)
        self.n_targets = len(self.targets)
        self.n_agents = n_agents if n_agents is not None else self.n_targets
        self.box = box
        self.speed = speed
        self.noise = noise
        self.capture_radius = capture_radius
        self._rng = np.random.default_rng(seed)
        self.pos = self._rng.random((self.n_agents, 2)) * box

    def _assign(self) -> dict[int, int]:
        """Greedy nearest assignment: target -> agent (each agent used once)."""
        d = np.linalg.norm(self.pos[:, None, :] - self.targets[None, :, :], axis=2)  # (agents, targets)
        assign: dict[int, int] = {}
        free_agents = set(range(self.n_agents))
        free_targets = set(range(self.n_targets))
        while free_agents and free_targets:
            sub = [(d[a, t], a, t) for a in free_agents for t in free_targets]
            _, a, t = min(sub)
            assign[t] = a
            free_agents.discard(a)
            free_targets.discard(t)
        return assign

    def step(self) -> None:
        assign = self._assign()
        for t, a in assign.items():
            direction = self.targets[t] - self.pos[a]
            dist = np.linalg.norm(direction)
            if dist > 1e-9:
                direction = direction / dist
            # active motion: step toward the target, capped at remaining distance, + noise
            angle = np.arctan2(direction[1], direction[0]) + self._rng.uniform(-self.noise, self.noise)
            move = min(self.speed, dist) * np.array([np.cos(angle), np.sin(angle)])
            self.pos[a] = self.pos[a] + move

    def coverage(self) -> float:
        """Fraction of target sites with some agent within the capture radius."""
        d = np.linalg.norm(self.pos[:, None, :] - self.targets[None, :, :], axis=2)
        return float(np.mean(d.min(axis=0) <= self.capture_radius))

    def run(self, steps: int = 200) -> float:
        for _ in range(steps):
            self.step()
        return self.coverage()

    def damage(self, k: int) -> None:
        k = min(k, self.n_agents - 1)
        keep = self._rng.choice(self.n_agents, size=self.n_agents - k, replace=False)
        self.pos = self.pos[keep]
        self.n_agents = len(self.pos)


def ring_target(n: int, radius: float = 3.0, center: float = 5.0) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((center + radius * np.cos(ang), center + radius * np.sin(ang)))
