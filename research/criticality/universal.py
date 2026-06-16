"""Universal criticality — one transition shape across three different substrates.

Today's through-line, measured on the same axes. Three systems with nothing in
common at the microscopic level each show the same thing: an order parameter that
stays near 1 below a critical value of a control knob and collapses to ~0 above it.

  - Ising (the computronium substrate, DRIFT): magnetisation <|m|> vs temperature.
  - Vicsek (active matter): flocking order phi vs angular noise.
  - Reconfiguration (programmable matter): target coverage vs a fixed disorder
    temperature held during the moves.

Normalise each control knob by its own transition point (where order = 1/2) and
the three curves fall on top of one another: **criticality is substrate-
independent** — the edge of chaos where DRIFT's reservoir capacity peaks is the
same edge these systems live on. That is the unifying claim of the daemons /
computronium line, made into a figure instead of a slogan.
"""

from __future__ import annotations

import numpy as np

from research.active_matter.vicsek import VicsekModel
from research.programmable_matter.reconfigure import ReconfigurableMatter, square_target


# ── Ising (mean-field, exact enumeration) — the computronium substrate ────────
def _spins(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    return 2.0 * bits - 1.0


def ising_order(temperature: float, n: int = 10, J: float = 1.0) -> float:
    """Mean-field Ising magnetisation order parameter <|m|>/N at temperature T."""
    s = _spins(n)
    m = s.sum(axis=1)
    E = -(J / (2.0 * n)) * (m * m - n)
    w = np.exp(-(E - E.min()) / temperature)
    p = w / w.sum()
    return float(np.sum(p * np.abs(m)) / n)


# ── Vicsek (active matter) ────────────────────────────────────────────────────
def vicsek_order(noise: float, n: int = 250, steps: int = 140, seed: int = 0) -> float:
    return VicsekModel(n=n, noise=float(noise), seed=seed).run(steps=steps)


# ── Reconfiguration (programmable matter) at fixed disorder temperature ───────
def reconfig_order(temperature: float, sweeps: int = 400, seed: int = 0) -> float:
    tgt = square_target(12, side=4)
    m = ReconfigurableMatter(12, tgt, seed=seed)
    for _ in range(sweeps):
        m.sweep(temperature)
    return m.coverage()


def transition_point(controls: np.ndarray, orders: np.ndarray, level: float = 0.5) -> float:
    """Control value where the (descending) order parameter crosses `level`."""
    for i in range(len(orders) - 1):
        if orders[i] >= level >= orders[i + 1]:
            x0, x1, y0, y1 = controls[i], controls[i + 1], orders[i], orders[i + 1]
            return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
    return float(controls[int(np.argmin(np.abs(orders - level)))])
