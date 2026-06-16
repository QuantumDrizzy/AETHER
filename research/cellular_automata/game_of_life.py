"""Conway's Game of Life — a discrete computational substrate.

The simplest "matter computes" model: a 2-state cellular automaton on a grid with
one local rule (a live cell survives with 2-3 live neighbours; a dead cell is born
with exactly 3). From that rule emerge still lifes, oscillators, and gliders that
travel — and the system is **Turing-complete** (you can build logic gates and a
universal computer out of glider collisions). It is the discrete cousin of DRIFT's
Ising substrate: structure + a local rule = computation.

Measured, verifiable facts:
  - a block is a still life (fixed point),
  - a blinker oscillates with period 2,
  - a glider returns to itself shifted by (1, 1) after period 4 (it moves),
  - from a random start the population self-organises down to a small, roughly
    stationary density (~3% of cells), far below the initial seeding.

Toroidal (wrap-around) boundary so gliders translate cleanly.
"""

from __future__ import annotations

import numpy as np


def step(grid: np.ndarray) -> np.ndarray:
    """One Game-of-Life generation (toroidal)."""
    g = grid.astype(np.int8)
    n = sum(np.roll(np.roll(g, di, 0), dj, 1)
            for di in (-1, 0, 1) for dj in (-1, 0, 1) if not (di == 0 and dj == 0))
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.int8)


def run(grid: np.ndarray, steps: int) -> np.ndarray:
    g = grid.astype(np.int8)
    for _ in range(steps):
        g = step(g)
    return g


def density(grid: np.ndarray) -> float:
    return float(grid.mean())


# ── canonical patterns (placed on an L×L grid) ────────────────────────────────
def _place(L: int, cells, origin=(1, 1)) -> np.ndarray:
    g = np.zeros((L, L), dtype=np.int8)
    r0, c0 = origin
    for r, c in cells:
        g[(r0 + r) % L, (c0 + c) % L] = 1
    return g


def block(L: int = 6) -> np.ndarray:
    return _place(L, [(0, 0), (0, 1), (1, 0), (1, 1)])


def blinker(L: int = 6) -> np.ndarray:
    return _place(L, [(0, 0), (0, 1), (0, 2)])


def glider(L: int = 20) -> np.ndarray:
    return _place(L, [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)])


def translated_equal(a: np.ndarray, b: np.ndarray, shift) -> bool:
    """True if grid b equals grid a shifted by `shift` (toroidal)."""
    return np.array_equal(np.roll(np.roll(a, shift[0], 0), shift[1], 1), b)
