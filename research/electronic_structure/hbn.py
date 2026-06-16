"""Gapped honeycomb (hexagonal boron nitride) — a band gap from broken symmetry.

Graphene's two sublattices (A, B) are identical, so the bands touch at the Dirac
points (zero gap). In hexagonal boron nitride the A site is boron and B is
nitrogen: the on-site energies differ by ±Δ, and that single broken symmetry opens
a gap. The Bloch Hamiltonian is graphene's, plus a sublattice mass term:

    H(k) = [[ +Δ,      t·f(k) ],
            [ t·f*(k),  −Δ    ]]        E_±(k) = ± sqrt(Δ² + |t·f(k)|²)

At the Dirac point f = 0, so E_± = ±Δ and the gap is exactly 2Δ — a real insulator
made from the same lattice as a semimetal, purely by making the two atoms
different. The classic "structure/composition sets the property" lesson, on the
electronic side (the metamaterials module shows the mechanical side).

Reuses GrapheneTB for geometry + structure factor; only the mass term is new.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.electronic_structure.graphene import GrapheneTB


@dataclass(frozen=True)
class GappedHoneycomb:
    a_cc: float = 1.45      # B–N distance (Å)
    t: float = 2.3          # hopping (eV)
    delta: float = 2.3      # sublattice mass (eV); gap = 2·delta

    @property
    def _g(self) -> GrapheneTB:
        return GrapheneTB(a_cc=self.a_cc, t=self.t)

    def structure_factor(self, k):
        return self._g.structure_factor(k)

    @property
    def K_point(self) -> np.ndarray:
        return self._g.K_point

    def bands(self, k):
        """Closed-form bands E_±(k) = ± sqrt(Δ² + (t|f|)²) in eV (single k or batch)."""
        tf = self.t * np.abs(self.structure_factor(k))
        e = np.sqrt(self.delta ** 2 + tf ** 2)
        return -e, e

    def hamiltonian(self, k) -> np.ndarray:
        f = self.structure_factor(np.asarray(k, dtype=float).reshape(2))
        return np.array([[self.delta, self.t * f],
                         [self.t * np.conj(f), -self.delta]], dtype=complex)

    def eigenergies(self, k) -> np.ndarray:
        return np.linalg.eigvalsh(self.hamiltonian(k))

    @property
    def band_gap_eV(self) -> float:
        """Analytic direct gap at the Dirac point: 2|Δ|."""
        return 2.0 * abs(self.delta)

    def measured_gap_eV(self, n_k: int = 120) -> float:
        """Minimum E_+ − E_− over a Brillouin-zone grid (should equal 2|Δ|)."""
        b1, b2 = self._g.reciprocal_vectors
        u = np.linspace(0.0, 1.0, n_k, endpoint=False)
        uu, vv = np.meshgrid(u, u, indexing="ij")
        k = uu.ravel()[:, None] * b1[None, :] + vv.ravel()[:, None] * b2[None, :]
        e_minus, e_plus = self.bands(k)
        return float(np.min(e_plus - e_minus))


def _demo() -> None:
    h = GappedHoneycomb()
    print("Gapped honeycomb / hBN (a = {:.2f} A, t = {:.1f} eV, delta = {:.1f} eV)"
          .format(h.a_cc, h.t, h.delta))
    print(f"  analytic gap (2*delta) : {h.band_gap_eV:.3f} eV")
    print(f"  measured gap over BZ   : {h.measured_gap_eV():.3f} eV")
    print(f"  delta=0 -> graphene gap: {GappedHoneycomb(delta=0.0).measured_gap_eV():.2e} eV (~0)")


if __name__ == "__main__":
    _demo()
