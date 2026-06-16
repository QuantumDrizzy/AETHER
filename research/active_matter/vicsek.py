"""Active matter — the Vicsek flocking model.

Self-propelled particles, each moving at constant speed, that on every step align
their heading with the average heading of neighbours within a radius, plus angular
noise. From this purely *local* rule a global order emerges: below a critical
noise the swarm spontaneously flocks (a collective direction); above it the motion
is incoherent. Order is measured by

    phi = | (1/N) Σ_j e^{iθ_j} |   ∈ [0, 1]      (1 = aligned, 0 = disordered).

This is "agency in matter" at the mesoscale — no leader, no blueprint, just a rule
and a noise knob — and it is a genuine non-equilibrium phase transition, the
active-matter cousin of the Ising/edge-of-chaos criticality in DRIFT.

Reference: Vicsek et al., PRL 75, 1226 (1995).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VicsekModel:
    n: int = 300
    box: float = 7.0          # density n/box^2 ~ 6 (well above the flocking threshold)
    radius: float = 1.0       # interaction radius
    speed: float = 0.03       # step length
    noise: float = 0.5        # angular noise amplitude eta (radians), in [0, 2pi]
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.pos = rng.random((self.n, 2)) * self.box
        self.theta = rng.uniform(-np.pi, np.pi, self.n)
        self._rng = rng

    def _neighbor_mask(self) -> np.ndarray:
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        diff -= self.box * np.round(diff / self.box)      # periodic minimum image
        dist2 = (diff * diff).sum(-1)
        return dist2 <= self.radius * self.radius

    def step(self) -> None:
        mask = self._neighbor_mask()
        sx = mask @ np.cos(self.theta)
        sy = mask @ np.sin(self.theta)
        mean_angle = np.arctan2(sy, sx)
        noise = self._rng.uniform(-self.noise / 2.0, self.noise / 2.0, self.n)
        self.theta = mean_angle + noise
        self.pos = (self.pos + self.speed * np.column_stack((np.cos(self.theta),
                                                             np.sin(self.theta)))) % self.box

    def order_parameter(self) -> float:
        return float(np.hypot(np.cos(self.theta).sum(), np.sin(self.theta).sum()) / self.n)

    def run(self, steps: int = 200, average_last: int = 60) -> float:
        """Run and return the order parameter averaged over the last `average_last` steps."""
        vals = []
        for t in range(steps):
            self.step()
            if t >= steps - average_last:
                vals.append(self.order_parameter())
        return float(np.mean(vals))


def sweep_noise(noise_values, steps: int = 200, **kw):
    """Mean order parameter vs noise amplitude (the flocking transition)."""
    out = []
    for eta in noise_values:
        m = VicsekModel(noise=float(eta), **kw)
        out.append(m.run(steps=steps))
    return np.asarray(out)
