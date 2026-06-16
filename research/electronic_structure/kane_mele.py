"""The Kane–Mele model — a Z2 topological insulator (quantum spin Hall).

Where Haldane breaks time-reversal symmetry to get a Chern insulator, Kane–Mele
*restores* it: intrinsic spin–orbit coupling acts like Haldane's complex hopping
but with **opposite sign for the two spins**. So spin-up sees a Haldane model with
+λ and spin-down sees its time-reversed partner with −λ. Each spin block carries a
Chern number ±1, they cancel (total Chern 0, time-reversal safe), but the spin
Chern (C↑ − C↓)/2 is non-zero: a **Z2 topological insulator** with helical edge
states — the quantum spin Hall effect (Kane & Mele 2005).

Reuses the Haldane block (no Rashba term here, so the two spins decouple and the
Chern of each block is well defined). Z2 = |C↑ − C↓| / 2 mod 2:
  - Z2 = 1 (topological) when the sublattice mass is small: |M| < 3√3 λ_SO,
  - Z2 = 0 (trivial) otherwise — same boundary as Haldane, now for each spin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.electronic_structure.haldane import Haldane


@dataclass(frozen=True)
class KaneMele:
    a_cc: float = 1.42
    t: float = 1.0
    lambda_so: float = 0.1     # intrinsic spin-orbit (plays Haldane's t2 role)
    M: float = 0.0             # sublattice mass (Rashba omitted: spins decouple)

    def _spin_block(self, spin: int) -> Haldane:
        # spin = +1 (up) -> phi = +pi/2 ; spin = -1 (down) -> phi = -pi/2
        return Haldane(a_cc=self.a_cc, t=self.t, t2=self.lambda_so,
                       phi=spin * np.pi / 2, M=self.M)

    def spin_chern_numbers(self, n_k: int = 36) -> tuple[int, int]:
        """(C_up, C_down) — the Chern number of each decoupled spin block."""
        return (self._spin_block(+1).chern_number(n_k),
                self._spin_block(-1).chern_number(n_k))

    def spin_chern(self, n_k: int = 36) -> int:
        c_up, c_dn = self.spin_chern_numbers(n_k)
        return (c_up - c_dn) // 2

    def z2_invariant(self, n_k: int = 36) -> int:
        """Z2 = (C_up − C_down)/2 mod 2: 1 = topological (QSH), 0 = trivial."""
        return abs(self.spin_chern(n_k)) % 2

    def band_gap(self, n_k: int = 60) -> float:
        return self._spin_block(+1).band_gap(n_k)

    def is_topological(self) -> bool:
        """Analytic: |M| < 3√3 λ_SO (each spin block is a Chern insulator)."""
        return self._spin_block(+1).is_topological()
