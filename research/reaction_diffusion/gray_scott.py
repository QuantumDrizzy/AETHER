"""Reaction–diffusion (Gray–Scott) — how matter self-organises into pattern.

Two chemicals U and V diffuse and react on a grid:

    ∂u/∂t = Du ∇²u − u v² + F(1 − u)
    ∂v/∂t = Dv ∇²v + u v² − (F + k) v

From a nearly uniform state plus a tiny seed, a Turing instability grows the
seed into stable spatial structure — spots, stripes, mazes — set entirely by the
feed/kill rates (F, k). It is morphogenesis from local rules: the same theme as
the daemons line (active matter, programmable matter), now in a continuous
chemical substrate. Spatial structure (the variance of V) grows from ~0 and
saturates, and a perfectly uniform field with no seed stays uniform — pattern
needs both the instability and a perturbation.

Toroidal Laplacian; the standard 9-point Gray–Scott kernel.
"""

from __future__ import annotations

import numpy as np

_KERNEL = np.array([[0.05, 0.20, 0.05],
                    [0.20, -1.0, 0.20],
                    [0.05, 0.20, 0.05]])

PRESETS = {            # (F, k)
    "spots": (0.0367, 0.0649),
    "stripes": (0.0545, 0.0620),
    "maze": (0.0290, 0.0570),
}


def _laplacian(a: np.ndarray) -> np.ndarray:
    out = -1.0 * a
    out += 0.20 * (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1))
    out += 0.05 * (np.roll(np.roll(a, 1, 0), 1, 1) + np.roll(np.roll(a, 1, 0), -1, 1)
                   + np.roll(np.roll(a, -1, 0), 1, 1) + np.roll(np.roll(a, -1, 0), -1, 1))
    return out


class GrayScott:
    def __init__(self, F: float = 0.029, k: float = 0.057,
                 Du: float = 0.16, Dv: float = 0.08, dt: float = 1.0):
        # default = the "maze" preset, which patterns robustly from a small seed;
        # spots/stripes are more sensitive to seeding (see PRESETS / [KNOWN_LIMIT]).
        self.F, self.k, self.Du, self.Dv, self.dt = F, k, Du, Dv, dt

    def seeded_init(self, size: int = 100, seed: int = 0):
        rng = np.random.default_rng(seed)
        u = np.ones((size, size))
        v = np.zeros((size, size))
        r = size // 10
        c = size // 2
        u[c - r:c + r, c - r:c + r] = 0.50
        v[c - r:c + r, c - r:c + r] = 0.25
        v += 0.02 * rng.random((size, size))
        return u, v

    def step(self, u, v):
        uvv = u * v * v
        u = u + self.dt * (self.Du * _laplacian(u) - uvv + self.F * (1 - u))
        v = v + self.dt * (self.Dv * _laplacian(v) + uvv - (self.F + self.k) * v)
        return np.clip(u, 0, 1), np.clip(v, 0, 1)

    def run(self, u, v, steps: int):
        for _ in range(steps):
            u, v = self.step(u, v)
        return u, v
