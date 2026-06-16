"""Programmable matter — self-reconfiguration and self-repair.

A measurable abstraction of the Columbia "robot metabolism" video: a set of
modular units on a grid that rearrange themselves into a target shape using only
*local* moves (a unit hops to an adjacent empty cell) driven by a mismatch energy,

    E = | occupied  XOR  target |        (symmetric difference; 0 = perfect shape)

minimised by Metropolis annealing. The point isn't the shape — it's what happens
under **damage**: remove some units and let it re-anneal, and the survivors
rearrange so that *every remaining unit sits on the target* (graceful degradation),
covering as much of the shape as the reduced mass allows instead of stopping. That
is the honest, quantified version of "robots that reconfigure and repair rather
than fail" — local rules + energy descent, no global blueprint.

(The reconfiguration is an optimisation; at larger scale it is the kind of problem
the temp0r annealing spine solves — here kept self-contained with single-unit
moves.)
"""

from __future__ import annotations

import numpy as np


class ReconfigurableMatter:
    def __init__(self, size: int, target: set[tuple[int, int]],
                 n_units: int | None = None, seed: int = 0):
        self.L = size
        self.target = set(target)
        self.n_units = n_units if n_units is not None else len(self.target)
        self._rng = np.random.default_rng(seed)
        all_cells = [(r, c) for r in range(size) for c in range(size)]
        idx = self._rng.choice(len(all_cells), size=self.n_units, replace=False)
        self.occupied = {all_cells[i] for i in idx}

    # ── metrics ──────────────────────────────────────────────────────────────
    def energy(self) -> int:
        return len(self.occupied ^ self.target)

    def coverage(self) -> float:
        """Fraction of the target shape that is filled."""
        return len(self.occupied & self.target) / len(self.target)

    def on_target_fraction(self) -> float:
        """Fraction of units that sit on a target cell (placement efficiency)."""
        if not self.occupied:
            return 0.0
        return len(self.occupied & self.target) / len(self.occupied)

    # ── dynamics ─────────────────────────────────────────────────────────────
    def _neighbors(self, cell):
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.L and 0 <= nc < self.L:
                yield (nr, nc)

    def _propose(self):
        a = tuple(self._rng.choice(list(self.occupied)))
        free = [b for b in self._neighbors(a) if b not in self.occupied]
        if not free:
            return None
        b = free[self._rng.integers(len(free))]
        return a, b

    def sweep(self, temp: float) -> None:
        """One Metropolis sweep (n_units proposed local moves) at temperature `temp`."""
        for _ in range(self.n_units):
            mv = self._propose()
            if mv is None:
                continue
            a, b = mv
            # incremental dE for moving a (occupied) -> b (empty)
            d = (1 if a in self.target else -1) + (1 if b not in self.target else -1)
            if d < 0 or self._rng.random() < np.exp(-d / max(temp, 1e-9)):
                self.occupied.discard(a)
                self.occupied.add(b)

    def anneal(self, sweeps: int = 400, t_init: float = 3.0, t_final: float = 0.02) -> int:
        cooling = (t_final / t_init) ** (1.0 / max(sweeps, 1))
        temp = t_init
        for _ in range(sweeps):
            self.sweep(temp)
            temp *= cooling
        return self.energy()

    def damage(self, k: int) -> None:
        """Destroy k units (remove them); the rest must re-anneal to cope."""
        k = min(k, len(self.occupied))
        victims_idx = self._rng.choice(len(self.occupied), size=k, replace=False)
        occ = list(self.occupied)
        for i in victims_idx:
            self.occupied.discard(occ[i])
        self.n_units = len(self.occupied)


def square_target(size: int, side: int, origin: tuple[int, int] = (2, 2)) -> set:
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r in range(side) for c in range(side)}
