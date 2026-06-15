"""AETHER × Spectra — band edges of large tight-binding systems via Lanczos.

This is the **first real consumer of the Spectra spine** (see Spectra's
docs/ADR-0001): AETHER owns the physics (it builds H(k) from a `TightBinding`
model); Spectra owns the linear algebra (extremal eigenvalues of a Hermitian
operator). For large N — long nanoribbons, supercells, multi-orbital materials —
the band *edges* are what matter, and Lanczos gets them without the full O(N³)
dense diagonalization. Validated against the dense reference (see tests).
"""

from __future__ import annotations

import numpy as np

from spectra import LinearOperator, lanczos

from research.electronic_structure.tight_binding import TightBinding


def tb_hamiltonian(tb: TightBinding, k: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """The N×N Bloch Hamiltonian H(k) at a single k-point (complex, Hermitian)."""
    return tb.hamiltonian_batch(np.asarray(k, dtype=float))[0]


def band_edges(
    tb: TightBinding,
    k: tuple[float, float] = (0.0, 0.0),
    n_edges: int = 4,
    iters: int | None = None,
) -> np.ndarray:
    """Extremal band energies of H(k) via Spectra's matrix-free Lanczos.

    The cheap path for large systems where full diagonalization is wasteful;
    returns the converged Ritz eigenvalues (ascending), whose extremes are the
    valence-bottom / conduction-top edges.
    """
    h = tb_hamiltonian(tb, k)
    op = LinearOperator.from_dense(h, hermitian=True)
    return lanczos(op, k=n_edges, iters=iters)
