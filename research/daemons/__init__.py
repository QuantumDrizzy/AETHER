"""Daemons — Maxwell's demon and information-driven matter.

Matter that uses information to act: the thermodynamic root (Szilard engine) of
the programmable/active-matter spectrum. Ties information (bits, mutual
information) to energy (Landauer) to matter (the gas) — computronium with a goal.
"""

from research.daemons.szilard import (
    EngineResult,
    binary_entropy,
    landauer_erasure_cost,
    mutual_information_bits,
    simulate_engine,
    work_bound,
    work_per_bit,
)

__all__ = [
    "binary_entropy",
    "mutual_information_bits",
    "work_per_bit",
    "landauer_erasure_cost",
    "work_bound",
    "simulate_engine",
    "EngineResult",
]
