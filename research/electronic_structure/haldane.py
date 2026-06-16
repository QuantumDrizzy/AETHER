"""The Haldane model — a topological insulator on the honeycomb lattice.

The progression graphene -> hBN -> Haldane: graphene is a gapless semimetal, hBN
opens a trivial gap with a sublattice mass M, and Haldane adds *complex* next-
nearest-neighbour hopping t2·e^{iφ} that breaks time-reversal symmetry and opens a
**topological** gap — one carrying a non-zero Chern number and chiral edge states
(the quantum anomalous Hall effect; Haldane 1988, Nobel 2016).

Bloch Hamiltonian, H(k) = ε(k) I + d(k)·σ, with NN vectors δ and NNN vectors a:
    d_x + i d_y = -t · Σ_δ e^{i k·δ}          (same structure factor as graphene)
    d_z(k)      =  M - 2 t2 sinφ · Σ_i sin(k·a_i)
    ε(k)        = -2 t2 cosφ · Σ_i cos(k·a_i)   (shifts bands, irrelevant to topology)

The lower band's Chern number is computed gauge-invariantly by the
Fukui–Hatsugai–Suzuki plaquette method. The model is topological (C = ±1) when
|M| < 3√3 · t2 · |sinφ| and trivial (C = 0) otherwise — a phase boundary this
module reproduces numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Haldane:
    a_cc: float = 1.42
    t: float = 1.0          # nearest-neighbour hopping
    t2: float = 0.1         # next-nearest-neighbour hopping magnitude
    phi: float = np.pi / 2  # complex NNN phase (breaks time reversal)
    M: float = 0.0          # sublattice mass (trivial gap)

    @property
    def deltas(self) -> np.ndarray:
        a = self.a_cc
        return a * np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])

    @property
    def nnn(self) -> np.ndarray:
        """Next-nearest-neighbour (A->A) vectors, set the chirality of the phase."""
        a = self.a_cc
        return a * np.array([[0.0, np.sqrt(3)], [-1.5, -np.sqrt(3) / 2], [1.5, -np.sqrt(3) / 2]])

    @property
    def reciprocal_vectors(self):
        a = self.a_cc
        a1 = a * np.array([1.5, -np.sqrt(3) / 2])
        a2 = a * np.array([1.5, np.sqrt(3) / 2])
        area = a1[0] * a2[1] - a1[1] * a2[0]
        b1 = 2 * np.pi / area * np.array([a2[1], -a2[0]])
        b2 = 2 * np.pi / area * np.array([-a1[1], a1[0]])
        return b1, b2

    def hamiltonian(self, k) -> np.ndarray:
        k = np.asarray(k, dtype=float).reshape(2)
        f = np.exp(1j * (self.deltas @ k)).sum()
        dx, dy = -self.t * f.real, -self.t * f.imag
        s = np.sin(self.nnn @ k)
        c = np.cos(self.nnn @ k)
        dz = self.M - 2 * self.t2 * np.sin(self.phi) * s.sum()
        eps = -2 * self.t2 * np.cos(self.phi) * c.sum()
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        return eps * np.eye(2) + dx * sx + dy * sy + dz * sz

    def lower_band_vector(self, k) -> np.ndarray:
        w, v = np.linalg.eigh(self.hamiltonian(k))
        return v[:, 0]                       # eigenvector of the lower eigenvalue

    def chern_number(self, n_k: int = 36) -> int:
        """Chern number of the lower band via Fukui–Hatsugai–Suzuki plaquettes."""
        b1, b2 = self.reciprocal_vectors
        u = np.linspace(0, 1, n_k, endpoint=False)
        # precompute lower-band eigenvectors on the grid
        vecs = np.empty((n_k, n_k, 2), dtype=complex)
        for i, ui in enumerate(u):
            for j, vj in enumerate(u):
                vecs[i, j] = self.lower_band_vector(ui * b1 + vj * b2)

        def link(a, b):
            z = np.vdot(a, b)
            return z / abs(z)

        total = 0.0
        for i in range(n_k):
            for j in range(n_k):
                ip, jp = (i + 1) % n_k, (j + 1) % n_k
                u1 = link(vecs[i, j], vecs[ip, j])
                u2 = link(vecs[ip, j], vecs[ip, jp])
                u3 = link(vecs[ip, jp], vecs[i, jp])
                u4 = link(vecs[i, jp], vecs[i, j])
                total += np.angle(u1 * u2 * u3 * u4)
        return int(round(total / (2 * np.pi)))

    def band_gap(self, n_k: int = 60) -> float:
        """Minimum direct gap over the BZ. Chern number is only defined when > 0."""
        b1, b2 = self.reciprocal_vectors
        u = np.linspace(0, 1, n_k, endpoint=False)
        gap = np.inf
        for ui in u:
            for vj in u:
                w = np.linalg.eigvalsh(self.hamiltonian(ui * b1 + vj * b2))
                gap = min(gap, w[1] - w[0])
        return float(gap)

    def is_topological(self) -> bool:
        """Analytic phase boundary: |M| < 3√3 · t2 · |sinφ| (and a gap is open)."""
        return abs(self.M) < 3 * np.sqrt(3) * self.t2 * abs(np.sin(self.phi))
