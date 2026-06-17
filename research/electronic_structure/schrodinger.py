"""1D Schrodinger solver — the real-space complement to k-space tight-binding.

The tight-binding modules (`graphene_tb`, `ssh`, `haldane`, ...) diagonalise a
Bloch Hamiltonian in k-space. This is the other half of single-particle quantum
mechanics: solve H psi = E psi directly in real space by finite differences,

    H = -1/2 d^2/dx^2 + V(x)     (hbar = m = 1)

with the kinetic term as the standard tridiagonal second-difference. Three
textbook potentials, each validated against its closed form:

* **Harmonic oscillator** V = 1/2 omega^2 x^2  ->  E_n = (n + 1/2) omega
* **Infinite square well** width L                ->  E_n = n^2 pi^2 / (2 L^2)
* **Rectangular barrier** -> quantum **tunnelling**: a transfer-matrix transmission
  T(E) checked against the analytic sinh/sin formula (T < 1 below the barrier,
  unit-transmission resonances above it).

Run:  python -m research.electronic_structure.schrodinger
"""

from __future__ import annotations

import numpy as np


# --- Bound states by finite differences --------------------------------------

def fd_hamiltonian(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Tridiagonal H = -1/2 d^2/dx^2 + V on a uniform grid (Dirichlet walls)."""
    n = len(x)
    dx = x[1] - x[0]
    k = 1.0 / dx ** 2
    H = np.diag(k + V) + np.diag(-0.5 * k * np.ones(n - 1), 1) \
        + np.diag(-0.5 * k * np.ones(n - 1), -1)
    return H


def solve(x: np.ndarray, V: np.ndarray, n_states: int = 5):
    """Lowest `n_states` eigenpairs (energies, normalised wavefunctions)."""
    E, psi = np.linalg.eigh(fd_hamiltonian(x, V))
    dx = x[1] - x[0]
    psi = psi / np.sqrt(dx)                    # normalise to integral |psi|^2 = 1
    return E[:n_states], psi[:, :n_states]


def harmonic_levels(n: int, omega: float = 1.0) -> np.ndarray:
    return omega * (np.arange(n) + 0.5)


def box_levels(n: int, L: float = 1.0) -> np.ndarray:
    nn = np.arange(1, n + 1)
    return nn ** 2 * np.pi ** 2 / (2.0 * L ** 2)


# --- Tunnelling: transmission through a rectangular barrier -------------------

def barrier_transmission_analytic(E: float, V0: float, a: float) -> float:
    """Closed-form T(E) for a rectangular barrier of height V0, width a (hbar=m=1)."""
    k = np.sqrt(2.0 * E)
    if E < V0:
        kappa = np.sqrt(2.0 * (V0 - E))
        return 1.0 / (1.0 + (V0 ** 2 * np.sinh(kappa * a) ** 2) / (4.0 * E * (V0 - E)))
    if E > V0:
        q = np.sqrt(2.0 * (E - V0))
        return 1.0 / (1.0 + (V0 ** 2 * np.sin(q * a) ** 2) / (4.0 * E * (E - V0)))
    return 1.0 / (1.0 + (V0 * a ** 2) / 2.0)   # E == V0 limit


def transmission_transfer_matrix(E: float, V: np.ndarray, dx: float) -> float:
    """T(E) for an arbitrary piecewise-constant V via plane-wave transfer matrices,
    with free regions (V=0) on both ends. Validated against the analytic barrier."""
    k0 = np.sqrt(2.0 * E + 0j)
    M = np.eye(2, dtype=complex)
    # local wavevector per slice; tiny imaginary part regularises k=0 (E == V slice)
    ks = np.sqrt(2.0 * (E - V) + 1e-12j)
    for i in range(len(V) - 1):
        k1, k2 = ks[i], ks[i + 1]
        # interface matrix (continuity of psi and psi')
        r = k1 / k2
        D = 0.5 * np.array([[1 + r, 1 - r],
                            [1 - r, 1 + r]], dtype=complex)
        # propagation across slice of width dx with wavevector k2
        ph = np.array([[np.exp(-1j * k2 * dx), 0],
                       [0, np.exp(1j * k2 * dx)]], dtype=complex)
        M = ph @ D @ M
    t = 1.0 / M[0, 0]
    return float(np.abs(t) ** 2 * (k0 / k0).real)   # equal k on both free ends


def _rect_barrier_grid(V0: float, a: float, pad: float = 6.0, n: int = 4000):
    x = np.linspace(-pad, a + pad, n)
    V = np.where((x >= 0.0) & (x <= a), V0, 0.0)
    return x, V


def _main() -> None:
    # harmonic oscillator
    x = np.linspace(-10, 10, 2000)
    E, _ = solve(x, 0.5 * x ** 2, n_states=5)
    print("Harmonic oscillator (omega=1):")
    print(f"  numeric  {np.round(E, 4)}")
    print(f"  analytic {np.round(harmonic_levels(5), 4)}")

    # infinite square well, L=1
    L = 1.0
    xb = np.linspace(0, L, 1001)[1:-1]
    Eb, _ = solve(xb, np.zeros_like(xb), n_states=4)
    print("Infinite well (L=1):")
    print(f"  numeric  {np.round(Eb, 3)}")
    print(f"  analytic {np.round(box_levels(4, L), 3)}")

    # tunnelling
    V0, a = 4.0, 1.0
    x2, V2 = _rect_barrier_grid(V0, a)
    dx = x2[1] - x2[0]
    print(f"Tunnelling through a barrier V0={V0}, a={a}:")
    for E in (1.0, 2.0, 4.0, 6.0):
        ta = barrier_transmission_analytic(E, V0, a)
        tn = transmission_transfer_matrix(E, V2, dx)
        print(f"  E={E:.1f}:  analytic T={ta:.4f}   numeric T={tn:.4f}")


if __name__ == "__main__":
    _main()
