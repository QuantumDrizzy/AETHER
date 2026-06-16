"""Maxwell's demon, made measurable: the Szilard engine.

A single particle in a box at temperature T. A "demon" measures which half the
particle is in, inserts a partition, and lets the gas push it back to full volume
— an isothermal expansion that extracts up to

    W = k_B T ln2   per bit   (∫ P dV from V/2 to V).

That looks like turning heat from one bath into work (a 2nd-law violation). The
resolution (Landauer/Bennett): the demon must *reset its memory* to repeat, and
erasing one bit costs ≥ k_B T ln2 — so the net work is ≤ 0. No free lunch.

The exact, measurable law (Sagawa–Ueda): an optimal feedback protocol converts the
**mutual information** the demon holds about the particle straight into work,

    <W_ext>  =  k_B T ln2 · I(measurement ; position)     [J]

For a symmetric measurement that errs with probability p, the mutual information
is I = 1 − H2(p) bits, so a perfect demon (p=0) extracts one full k_B T ln2 and a
useless one (p=0.5) extracts nothing. This is the exact bridge from information
(bits — the *same* mutual information used in HYLE) to energy (joules) to matter
(the gas): the demon is computronium with a goal — measure, convert, pay Landauer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K_B = 1.380649e-23
LN2 = math.log(2.0)


def binary_entropy(p: float) -> float:
    """H2(p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def mutual_information_bits(error_p: float) -> float:
    """Information the demon holds about the particle: I = 1 − H2(p) bits."""
    return 1.0 - binary_entropy(error_p)


def _empirical_mutual_information_bits(true_side: np.ndarray, belief: np.ndarray) -> float:
    """MI (bits) between two binary samples, estimated from their joint counts."""
    n = len(true_side)
    joint = np.zeros((2, 2))
    for t, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        joint[t, b] = np.sum((true_side == t) & (belief == b)) / n
    pt = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    mi = 0.0
    for t in range(2):
        for b in range(2):
            if joint[t, b] > 0 and pt[t] > 0 and pb[b] > 0:
                mi += joint[t, b] * math.log2(joint[t, b] / (pt[t] * pb[b]))
    return mi


def work_per_bit(temperature_k: float = 300.0) -> float:
    """Ideal Szilard work per perfect bit of information: k_B T ln2 (J)."""
    return K_B * temperature_k * LN2


def landauer_erasure_cost(temperature_k: float = 300.0) -> float:
    """Cost to reset one bit of the demon's memory each cycle: k_B T ln2 (J)."""
    return K_B * temperature_k * LN2


def work_bound(temperature_k: float, error_p: float) -> float:
    """Extractable work (optimal feedback): k_B T ln2 (1 − H2(p))."""
    return work_per_bit(temperature_k) * mutual_information_bits(error_p)


@dataclass
class EngineResult:
    mean_work_per_cycle: float       # measured (= k_B T ln2 * empirical MI), J
    info_bound: float                # theoretical k_B T ln2 (1 − H2(p)), J
    ideal_work_per_bit: float        # k_B T ln2, J
    erasure_cost: float              # Landauer reset (per cycle), J
    mutual_information_bits: float   # empirical MI the demon held

    @property
    def net_work_after_erasure(self) -> float:
        return self.mean_work_per_cycle - self.erasure_cost

    def summary(self) -> str:
        return (
            f"demon held I = {self.mutual_information_bits:.3f} bits -> extracted "
            f"{self.mean_work_per_cycle:.3e} J/cycle (theory {self.info_bound:.3e}); "
            f"after Landauer erasure net = {self.net_work_after_erasure:.3e} J "
            f"(<= 0 -> 2nd law safe)."
        )


def simulate_engine(n_cycles: int = 20000, temperature_k: float = 300.0,
                    error_p: float = 0.0, seed: int = 0) -> EngineResult:
    """Run the measurement many times, estimate the mutual information the demon
    actually holds, and convert it to work at the Sagawa–Ueda rate k_B T ln2 · I.

    The particle is equally likely on either side; the demon's reading errs with
    probability `error_p`. We measure the *empirical* mutual information between
    truth and belief (the same quantity HYLE computes), so the extracted work is
    grounded in sampled information, not assumed. Optimal feedback => the work
    equals the bound; it can never exceed it.
    """
    rng = np.random.default_rng(seed)
    true_side = rng.integers(0, 2, size=n_cycles)
    flips = rng.random(n_cycles) < error_p
    belief = np.where(flips, 1 - true_side, true_side)

    mi_emp = _empirical_mutual_information_bits(true_side, belief)
    w_ext = work_per_bit(temperature_k) * mi_emp
    return EngineResult(
        mean_work_per_cycle=w_ext,
        info_bound=work_bound(temperature_k, error_p),
        ideal_work_per_bit=work_per_bit(temperature_k),
        erasure_cost=landauer_erasure_cost(temperature_k),
        mutual_information_bits=mi_emp,
    )
