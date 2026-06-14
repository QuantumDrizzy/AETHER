"""Su-Schrieffer-Heeger (SSH) chain - the canonical 1D topological model.

A 1D chain with two atoms per cell and alternating hoppings t1 (intra-cell) and
t2 (inter-cell). It is the textbook topological insulator and the cleanest
demonstration of the bulk-boundary correspondence:

- Bulk gap = 2|t1 - t2|, closing at t1 = t2.
- A finite open chain in the TOPOLOGICAL phase (t2 > t1) hosts two near-zero-energy
  edge states; in the TRIVIAL phase (t1 > t2) the gap is clean (no mid-gap states).
- Chiral (sublattice) symmetry -> the spectrum is symmetric about E = 0.

Built on the general `TightBinding` solver, exercising it on both a periodic model
(H(k)) and a finite/open system (single H) - the path the general solver opens
toward real materials (next: 2-D graphene nanoribbons). See docs/ADR-0001.

Run standalone:  python -m research.electronic_structure.ssh
"""

from __future__ import annotations

import numpy as np

from research.electronic_structure.tight_binding import Hopping, TightBinding


def ssh_periodic(t1: float = 1.0, t2: float = 1.0, cell: float = 1.0) -> TightBinding:
    """Bulk SSH unit cell: 2 orbitals, intra-cell hop t1, inter-cell hop t2."""
    hoppings = [
        Hopping(0, 1, (0.0, 0.0), t1),     # A-B inside the cell
        Hopping(1, 0, (cell, 0.0), t2),    # B(cell 0) - A(cell +1)
    ]
    return TightBinding(n_orbitals=2, hoppings=hoppings)


def ssh_finite(n_cells: int, t1: float = 1.0, t2: float = 1.0) -> TightBinding:
    """Finite OPEN SSH chain of n_cells (2*n_cells atoms), no periodicity."""
    n = 2 * n_cells
    hoppings = []
    for c in range(n_cells):
        a, b = 2 * c, 2 * c + 1
        hoppings.append(Hopping(a, b, (0.0, 0.0), t1))          # intra-cell bond
        if b + 1 < n:
            hoppings.append(Hopping(b, b + 1, (0.0, 0.0), t2))  # inter-cell bond
    return TightBinding(n_orbitals=n, hoppings=hoppings)


def bulk_gap(t1: float, t2: float, n_k: int = 4001) -> float:
    """Minimum direct gap of the bulk SSH bands over the 1-D BZ (= 2|t1-t2|)."""
    tb = ssh_periodic(t1, t2)
    k = np.zeros((n_k, 2))
    k[:, 0] = np.linspace(-np.pi, np.pi, n_k)
    e = tb.bands(k)  # (n_k, 2), ascending
    return float(np.min(e[:, 1] - e[:, 0]))


def finite_spectrum(n_cells: int, t1: float, t2: float) -> np.ndarray:
    """Eigenenergies of the finite open chain (k irrelevant: all bonds intra-system)."""
    return ssh_finite(n_cells, t1, t2).bands(np.array([[0.0, 0.0]]))[0]


def count_edge_states(n_cells: int, t1: float, t2: float, tol: float | None = None) -> int:
    """Number of near-zero-energy states (edge states) in the finite chain."""
    if tol is None:
        tol = 0.1 * min(t1, t2)
    return int(np.sum(np.abs(finite_spectrum(n_cells, t1, t2)) < tol))


def _demo() -> None:
    print("SSH chain (canonical 1-D topological model)")
    for t1, t2 in [(1.0, 0.6), (1.0, 1.0), (0.6, 1.0)]:
        print(f"  t1={t1}, t2={t2}: bulk gap = {bulk_gap(t1, t2):.3f}  (2|t1-t2| = {2*abs(t1-t2):.3f})")
    print("  finite open chain (25 cells):")
    print(f"    trivial  (t1=1.0,t2=0.5): {count_edge_states(25, 1.0, 0.5)} edge states")
    print(f"    topolog. (t1=0.5,t2=1.0): {count_edge_states(25, 0.5, 1.0)} edge states")


if __name__ == "__main__":
    _demo()
