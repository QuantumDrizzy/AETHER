"""Phonon dispersion of 1D atomic chains — band structure, but for vibrations.

The lattice-dynamics twin of the electronic tight-binding work in
`electronic_structure/`: instead of electrons hopping on a lattice we have atoms
on springs, and diagonalising the **dynamical matrix** D(k) gives omega^2(k)
instead of an energy band. Same machinery (Bloch's theorem, a small k-dependent
matrix per cell), different physics.

Two canonical cases, both built as real dynamical matrices and validated against
their closed forms:

* **Monatomic chain** (1 atom/cell, spring K, mass m): one acoustic branch
  omega(k) = 2 sqrt(K/m) |sin(k a / 2)|. Linear (sound) near k=0, flat at the
  zone boundary (standing wave, zero group velocity).
* **Diatomic chain** (masses m1 != m2): the unit cell doubles, so the band folds
  into an **acoustic** branch (-> 0 at k=0) and an **optical** branch, separated
  by a **phonon band gap** at the zone boundary — exactly the vibrational analogue
  of the hBN electronic gap from a two-atom basis (`electronic_structure/hbn.py`).
  At k=0 the optical branch sits at sqrt(2K (1/m1 + 1/m2)); at the zone boundary
  the gap runs from sqrt(2K/m_heavy) to sqrt(2K/m_light).

Run:  python -m research.phonons.phonon_chain
"""

from __future__ import annotations

import numpy as np


# --- Monatomic chain ---------------------------------------------------------

def monatomic_dynamical_matrix(k: float, K: float, m: float, a: float) -> np.ndarray:
    """1x1 dynamical matrix D(k) = (2K/m)(1 - cos k a). Eigenvalue = omega^2."""
    return np.array([[(2.0 * K / m) * (1.0 - np.cos(k * a))]])


def monatomic_dispersion(ks: np.ndarray, K: float = 1.0, m: float = 1.0,
                         a: float = 1.0) -> np.ndarray:
    """omega(k) by diagonalising D(k) at each k (returns shape (len(ks),))."""
    out = np.empty(len(ks))
    for i, k in enumerate(ks):
        w2 = np.linalg.eigvalsh(monatomic_dynamical_matrix(k, K, m, a))[0]
        out[i] = np.sqrt(max(w2, 0.0))
    return out


def monatomic_closed_form(ks: np.ndarray, K: float = 1.0, m: float = 1.0,
                          a: float = 1.0) -> np.ndarray:
    return 2.0 * np.sqrt(K / m) * np.abs(np.sin(ks * a / 2.0))


def sound_speed(K: float = 1.0, m: float = 1.0, a: float = 1.0) -> float:
    """Long-wavelength group velocity domega/dk at k->0 = a sqrt(K/m)."""
    return a * np.sqrt(K / m)


# --- Diatomic chain (two atoms per cell) -------------------------------------

def diatomic_dynamical_matrix(k: float, K: float, m1: float, m2: float,
                              a: float) -> np.ndarray:
    """Mass-weighted 2x2 dynamical matrix; eigenvalues are omega^2 (acoustic,
    optical). Off-diagonal coupling K(1 + e^{-ika})/sqrt(m1 m2)."""
    g = K * (1.0 + np.exp(-1j * k * a)) / np.sqrt(m1 * m2)
    return np.array([[2.0 * K / m1, -g],
                     [-np.conj(g), 2.0 * K / m2]], dtype=complex)


def diatomic_dispersion(ks: np.ndarray, K: float = 1.0, m1: float = 1.0,
                        m2: float = 2.0, a: float = 1.0):
    """Returns (acoustic, optical) omega arrays via eigendecomposition of D(k)."""
    ac = np.empty(len(ks))
    op = np.empty(len(ks))
    for i, k in enumerate(ks):
        w2 = np.linalg.eigvalsh(diatomic_dynamical_matrix(k, K, m1, m2, a))
        w2 = np.clip(w2, 0.0, None)
        ac[i], op[i] = np.sqrt(w2[0]), np.sqrt(w2[1])
    return ac, op


def optical_at_gamma(K: float = 1.0, m1: float = 1.0, m2: float = 2.0) -> float:
    """Closed form: optical branch frequency at k=0 = sqrt(2K(1/m1 + 1/m2))."""
    return np.sqrt(2.0 * K * (1.0 / m1 + 1.0 / m2))


def zone_boundary_gap(K: float = 1.0, m1: float = 1.0, m2: float = 2.0):
    """Phonon gap edges at k = pi/a: (acoustic top, optical bottom).
    Acoustic top = sqrt(2K/m_heavy), optical bottom = sqrt(2K/m_light)."""
    heavy, light = max(m1, m2), min(m1, m2)
    return np.sqrt(2.0 * K / heavy), np.sqrt(2.0 * K / light)


def _main() -> None:
    K, m, a = 1.0, 1.0, 1.0
    ks = np.linspace(-np.pi / a, np.pi / a, 401)

    num = monatomic_dispersion(ks, K, m, a)
    cf = monatomic_closed_form(ks, K, m, a)
    print("Monatomic chain:")
    print(f"  max |numeric - closed form| = {np.max(np.abs(num - cf)):.2e}")
    print(f"  omega_max = {num.max():.4f}  (theory 2 sqrt(K/m) = {2*np.sqrt(K/m):.4f})")
    print(f"  sound speed (k->0 slope) = {sound_speed(K, m, a):.4f}")

    m1, m2 = 1.0, 2.0
    ac, op = diatomic_dispersion(ks, K, m1, m2, a)
    print("Diatomic chain (m1=1, m2=2):")
    print(f"  acoustic(k=0) = {ac[len(ks)//2]:.4f}  (theory 0)")
    print(f"  optical(k=0)  = {op[len(ks)//2]:.4f}  (theory {optical_at_gamma(K, m1, m2):.4f})")
    ac_top, op_bot = zone_boundary_gap(K, m1, m2)
    print(f"  zone-boundary gap: [{ac_top:.4f}, {op_bot:.4f}]  width {op_bot-ac_top:.4f}")


if __name__ == "__main__":
    _main()
