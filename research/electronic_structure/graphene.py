"""Nearest-neighbor tight-binding model of graphene — the scientific spine.

This is the canonical electronic-structure model in 2-D materials: a honeycomb
lattice with two sublattices (A, B), one p_z orbital per carbon atom, and a
single nearest-neighbor hopping `t`. Despite its simplicity it reproduces the
physics that made graphene famous and, crucially, every prediction here has a
**closed-form answer** to validate against (see tests/python/test_graphene_tb.py):

- Dirac cones: the gap closes linearly at the Brillouin-zone corners (K, K').
- Bandwidth = 6|t|  (E = ±3t at the Γ point).
- Electron–hole symmetry: E_+(k) = −E_−(k).
- Fermi velocity v_F = 3·t·a_cc / (2·ħ) ≈ 10^6 m/s (~ c/300).

Bloch Hamiltonian (2×2):
    H(k) = [[0,        t·f(k)],
            [t·f*(k),  0     ]]
with structure factor f(k) = Σ_δ exp(i k·δ) over the three nearest-neighbor
vectors δ. Eigenvalues: E_±(k) = ± t·|f(k)|.

Pure NumPy / CPU for now (fast to iterate and validate). A GPU-batched k-space
sweep (torch.linalg.eigh) is a later optimization, to be validated against this
reference. See docs/ADR-0001.

Run standalone:  python -m research.electronic_structure.graphene
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Physical constants
EV_TO_J = 1.602176634e-19      # J per eV
HBAR_JS = 1.054571817e-34      # reduced Planck constant, J·s
ANGSTROM_M = 1.0e-10           # m per Å
C_LIGHT = 299_792_458.0        # m/s


@dataclass(frozen=True)
class GrapheneTB:
    """Tight-binding graphene.

    Parameters
    ----------
    a_cc : float
        Carbon–carbon distance in Å (default 1.42 Å).
    t : float
        Nearest-neighbor hopping magnitude in eV (default 2.7 eV).
    """

    a_cc: float = 1.42
    t: float = 2.7

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def deltas(self) -> np.ndarray:
        """The three nearest-neighbor vectors (Å), from an A atom to its B atoms."""
        a = self.a_cc
        return a * np.array(
            [
                [1.0, 0.0],
                [-0.5, np.sqrt(3.0) / 2.0],
                [-0.5, -np.sqrt(3.0) / 2.0],
            ]
        )

    @property
    def K_point(self) -> np.ndarray:
        """A Dirac point K in reciprocal space (1/Å), where the gap closes."""
        return (2.0 * np.pi / (3.0 * self.a_cc)) * np.array([1.0, 1.0 / np.sqrt(3.0)])

    @property
    def K_prime_point(self) -> np.ndarray:
        """The inequivalent Dirac point K' (1/Å)."""
        return (2.0 * np.pi / (3.0 * self.a_cc)) * np.array([1.0, -1.0 / np.sqrt(3.0)])

    # ------------------------------------------------------------------
    # Electronic structure
    # ------------------------------------------------------------------

    def structure_factor(self, k):
        """f(k) = Σ_δ exp(i k·δ). Accepts a single k (shape (2,)) or a batch
        (shape (N, 2)); returns a complex scalar or array accordingly."""
        k = np.asarray(k, dtype=float)
        single = k.ndim == 1
        k2 = k.reshape(1, 2) if single else k
        f = np.exp(1j * (k2 @ self.deltas.T)).sum(axis=-1)
        return complex(f[0]) if single else f

    def bands(self, k):
        """Closed-form bands. Returns (E_minus, E_plus) in eV.

        Works on a single k or a batch (vectorized)."""
        f_abs = np.abs(self.structure_factor(k))
        return -self.t * f_abs, self.t * f_abs

    def hamiltonian(self, k) -> np.ndarray:
        """The 2×2 Bloch Hamiltonian at a single k point (eV)."""
        f = self.structure_factor(np.asarray(k, dtype=float).reshape(2))
        return np.array([[0.0, self.t * f], [self.t * np.conj(f), 0.0]], dtype=complex)

    def eigenergies(self, k) -> np.ndarray:
        """Eigenvalues of H(k) at a single k, ascending: [E_minus, E_plus] (eV).

        Independent of `bands()` (diagonalization vs closed form) — the two
        agreeing is itself a validation."""
        return np.linalg.eigvalsh(self.hamiltonian(k))

    # ------------------------------------------------------------------
    # Derived physics
    # ------------------------------------------------------------------

    @property
    def bandwidth_eV(self) -> float:
        """Total π-band width = 6|t| (from −3t at Γ to +3t at Γ)."""
        return 6.0 * abs(self.t)

    @property
    def fermi_velocity_m_s(self) -> float:
        """Analytic Fermi velocity v_F = 3·t·a_cc / (2·ħ) in m/s."""
        t_j = abs(self.t) * EV_TO_J
        a_m = self.a_cc * ANGSTROM_M
        return 3.0 * t_j * a_m / (2.0 * HBAR_JS)

    def band_path(self, corners: np.ndarray, n_per_segment: int = 200):
        """Sample a k-path through high-symmetry corners for plotting.

        Returns (k_points (M,2), distance (M,)) where distance is the cumulative
        path length, suitable for a band-structure plot."""
        corners = np.asarray(corners, dtype=float)
        ks, dist = [], []
        d0 = 0.0
        for a, b in zip(corners[:-1], corners[1:]):
            seg = np.linspace(a, b, n_per_segment, endpoint=False)
            seglen = np.linalg.norm(b - a)
            ks.append(seg)
            dist.append(d0 + np.linspace(0.0, seglen, n_per_segment, endpoint=False))
            d0 += seglen
        ks.append(corners[-1:].copy())
        dist.append(np.array([d0]))
        return np.vstack(ks), np.concatenate(dist)

    @property
    def reciprocal_vectors(self):
        """Reciprocal lattice vectors (b1, b2) in 1/Å, with a_i·b_j = 2π·δ_ij."""
        a = self.a_cc
        a1 = a * np.array([1.5, -np.sqrt(3.0) / 2.0])
        a2 = a * np.array([1.5, np.sqrt(3.0) / 2.0])
        area = a1[0] * a2[1] - a1[1] * a2[0]
        b1 = 2.0 * np.pi / area * np.array([a2[1], -a2[0]])
        b2 = 2.0 * np.pi / area * np.array([-a1[1], a1[0]])
        return b1, b2

    def density_of_states(self, n_k: int = 300, n_bins: int = 400):
        """Density of states over the Brillouin zone.

        Samples the reciprocal unit cell on an n_k × n_k grid, evaluates both
        bands, and histograms them. Returns (energy_centers, dos) in eV / (states
        per eV per unit cell), normalized so ∫ dos dE = 2 (two bands per cell).

        For graphene this reproduces the textbook shape: DOS vanishes linearly at
        the Dirac point (E=0) and has van Hove (logarithmic) singularities at ±t.
        """
        b1, b2 = self.reciprocal_vectors
        u = np.linspace(0.0, 1.0, n_k, endpoint=False)
        uu, vv = np.meshgrid(u, u, indexing="ij")
        k = uu.ravel()[:, None] * b1[None, :] + vv.ravel()[:, None] * b2[None, :]
        e_minus, e_plus = self.bands(k)
        energies = np.concatenate([e_minus, e_plus])

        emax = 3.0 * abs(self.t)
        hist, edges = np.histogram(energies, bins=n_bins, range=(-emax, emax))
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_w = edges[1] - edges[0]
        dos = hist / (hist.sum() * bin_w) * 2.0  # ∫ dos dE = 2 bands
        return centers, dos


def _demo() -> None:
    # ASCII-only output (Windows consoles default to cp1252 and choke on Greek/symbols).
    g = GrapheneTB()
    print("Graphene tight-binding (a_cc = {:.2f} Angstrom, t = {:.1f} eV)".format(g.a_cc, g.t))
    e_minus, e_plus = g.bands(np.array([0.0, 0.0]))
    print(f"  Gamma point energies  : {e_minus:+.3f} / {e_plus:+.3f} eV  (+/-3t = +/-{3*g.t:.3f})")
    print(f"  bandwidth             : {g.bandwidth_eV:.3f} eV  (= 6t)")
    eK = g.bands(g.K_point)
    print(f"  K point (Dirac) gap   : {abs(eK[1] - eK[0]):.2e} eV  (should be ~0)")
    vf = g.fermi_velocity_m_s
    print(f"  Fermi velocity        : {vf:.3e} m/s  (~ c/{C_LIGHT / vf:.0f})")


if __name__ == "__main__":
    _demo()
