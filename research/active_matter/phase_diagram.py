"""Vicsek phase diagram — the flocking region in the noise-density plane.

The 1-D noise sweep (vicsek.py) showed an order->disorder transition; the full
picture is 2-D. Flocking needs two things at once: low enough angular noise AND
high enough density (so each agent actually has neighbours to align with). Mapping
the order parameter φ over the (noise, density) plane shows a flocking *region*: a
high-order corner at low noise / high density that dissolves as either knob is
pushed. Density here is N / box² (agents per unit area).
"""

from __future__ import annotations

import numpy as np

from research.active_matter.vicsek import VicsekModel


def order_at(noise: float, n_agents: int, box: float = 7.0,
             steps: int = 140, seed: int = 0) -> float:
    return VicsekModel(n=n_agents, box=box, noise=float(noise), seed=seed).run(steps=steps)


def phase_diagram(noises, densities, box: float = 7.0, steps: int = 140, seed: int = 0):
    """Return a (len(densities), len(noises)) grid of the order parameter φ.

    Each density d is realised as N = round(d · box²) agents.
    """
    grid = np.zeros((len(densities), len(noises)))
    for i, d in enumerate(densities):
        n = max(2, int(round(d * box * box)))
        for j, eta in enumerate(noises):
            grid[i, j] = order_at(eta, n, box, steps, seed)
    return grid
