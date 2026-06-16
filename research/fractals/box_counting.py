"""Box-counting fractal dimension — measure self-similarity, exactly.

A fractal fills space at a non-integer dimension: cover it with boxes of side s and
the number of non-empty boxes scales as N(s) ∝ s^{−D_f}. The box-counting dimension
D_f is the slope of log N vs log(1/s). This is the measurement tool behind every
"fractal" claim in the self-organisation line (DLA, reaction-diffusion fronts,
percolation clusters); here it is validated on deterministic fractals whose
dimension is known in closed form:

  - Sierpinski triangle  →  D_f = log 3 / log 2 ≈ 1.585,
  - Sierpinski carpet    →  D_f = log 8 / log 3 ≈ 1.893,
  - a filled square      →  D_f = 2 (not fractal).

Recovering those exact numbers shows the estimator is sound; it can then be pointed
at any pattern.
"""

from __future__ import annotations

import math

import numpy as np


def box_count(grid: np.ndarray, s: int) -> int:
    """Number of s×s boxes containing at least one occupied cell."""
    h, w = grid.shape
    h2, w2 = (h // s) * s, (w // s) * s
    g = grid[:h2, :w2]
    blocks = g.reshape(h2 // s, s, w2 // s, s)
    return int(np.any(blocks, axis=(1, 3)).sum())


def box_counting_dimension(grid: np.ndarray, scales) -> float:
    """D_f = slope of log N(s) vs log(1/s) over the given box sizes."""
    s = np.asarray(scales, dtype=float)
    counts = np.array([box_count(grid, int(si)) for si in scales], dtype=float)
    return float(np.polyfit(np.log(1.0 / s), np.log(counts), 1)[0])


def sierpinski_triangle(power: int = 8) -> np.ndarray:
    """Sierpinski triangle on a 2^power grid: cell (x,y) set iff (x & y) == 0."""
    n = 2 ** power
    x = np.arange(n)
    return (x[None, :] & x[:, None]) == 0


def sierpinski_carpet(power: int = 5) -> np.ndarray:
    """Sierpinski carpet on a 3^power grid (remove the centre of every 3×3)."""
    n = 3 ** power
    yy, xx = np.mgrid[0:n, 0:n]
    keep = np.ones((n, n), dtype=bool)
    a, b = xx.copy(), yy.copy()
    for _ in range(power):
        keep &= ~((a % 3 == 1) & (b % 3 == 1))
        a //= 3
        b //= 3
    return keep


def filled_square(n: int = 256) -> np.ndarray:
    return np.ones((n, n), dtype=bool)


SIERPINSKI_TRIANGLE_DF = math.log(3) / math.log(2)   # ≈ 1.585
SIERPINSKI_CARPET_DF = math.log(8) / math.log(3)     # ≈ 1.893
