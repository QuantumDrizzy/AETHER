"""Site percolation — a *geometric* critical phenomenon (connectivity threshold).

The universal-criticality thread so far is dynamical (Ising, Vicsek, reservoirs).
Percolation adds a different flavour: pure geometry. Occupy each site of a square
lattice independently with probability p; below a critical p_c the occupied sites
form only small clusters, above it a single cluster suddenly spans the whole system.
The spanning probability jumps from 0 to 1 across p_c — a textbook second-order
transition with no dynamics at all.

For 2D site percolation on the square lattice the known threshold is
**p_c ≈ 0.5927**. This module measures the spanning probability vs p and recovers
p_c from the crossing, showing criticality is not just a property of dynamics but
of connectivity itself — relevant wherever a substrate's function switches on with
density (conductive networks, gels, neural connectivity).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

P_C_SQUARE_SITE = 0.5927       # known 2D square-lattice site threshold


def occupy(size: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random((size, size)) < p


def has_spanning_cluster(grid: np.ndarray) -> bool:
    """True if some occupied cluster touches both the top and bottom rows."""
    labels, n = label(grid)          # 4-connectivity by default
    if n == 0:
        return False
    top = set(np.unique(labels[0])) - {0}
    bottom = set(np.unique(labels[-1])) - {0}
    return len(top & bottom) > 0


def spanning_probability(size: int, p: float, trials: int = 40, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    hits = sum(has_spanning_cluster(occupy(size, p, rng)) for _ in range(trials))
    return hits / trials


def estimate_pc(size: int = 64, trials: int = 60, seed: int = 0) -> float:
    """Estimate p_c as the p where the spanning probability crosses 1/2."""
    ps = np.linspace(0.45, 0.75, 31)
    probs = np.array([spanning_probability(size, p, trials, seed) for p in ps])
    for i in range(len(probs) - 1):
        if probs[i] <= 0.5 <= probs[i + 1]:
            # linear interpolation of the crossing
            x0, x1, y0, y1 = ps[i], ps[i + 1], probs[i], probs[i + 1]
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
    return float(ps[int(np.argmin(np.abs(probs - 0.5)))])
