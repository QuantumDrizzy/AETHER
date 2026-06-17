"""Phonon heat capacity — from the vibrational spectrum to a macroscopic C_v(T).

The thermodynamic payoff of `phonon_chain.py`: once you know the phonon
frequencies, statistical mechanics gives the lattice heat capacity. Each harmonic
mode of frequency omega is a quantum oscillator whose heat capacity is the
Einstein function

    c(omega, T) = k_B * x^2 e^x / (e^x - 1)^2 ,   x = hbar*omega / (k_B*T)

and the solid's C_v is the sum over all modes. Three routes, all consistent in
their limits (units hbar = k_B = 1 throughout):

* **From a spectrum** — sum c(omega, T) over the actual chain frequencies. This is
  the microscopic route that connects directly to `phonon_chain.py`.
* **Einstein model** — all modes at one frequency theta_E. Right at high T
  (Dulong-Petit 3 N k_B) but freezes out *exponentially* at low T — historically
  too fast, which is exactly why Debye was needed.
* **Debye model** — a linear (acoustic) spectrum up to a cutoff theta_D. Gives the
  famous **C_v ~ T^3** law at low T and Dulong-Petit at high T.

Run:  python -m research.phonons.heat_capacity
"""

from __future__ import annotations

import numpy as np


def _einstein_function(x: np.ndarray) -> np.ndarray:
    """x^2 e^x / (e^x - 1)^2, the heat capacity of one oscillator (-> 1 as x->0)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)                 # x -> 0 limit is 1 (classical k_B)
    nz = x > 1e-8
    xe = x[nz]
    ex = np.exp(xe)
    out[nz] = xe ** 2 * ex / (ex - 1.0) ** 2
    return out


def heat_capacity_from_spectrum(omegas: np.ndarray, T: float) -> float:
    """C_v(T) = sum over modes of the Einstein function (k_B = hbar = 1).
    High-T limit -> number of modes (each mode contributes k_B)."""
    if T <= 0.0:
        return 0.0
    omegas = np.asarray(omegas, dtype=float)
    omegas = omegas[omegas > 1e-12]       # drop zero modes (translations)
    return float(np.sum(_einstein_function(omegas / T)))


def einstein_heat_capacity(T, theta_E: float = 1.0, n_modes: int = 3) -> np.ndarray:
    """Einstein model: n_modes oscillators all at theta_E. -> n_modes at high T."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    c = np.zeros_like(T)
    pos = T > 0
    c[pos] = n_modes * _einstein_function(theta_E / T[pos])
    return c


def debye_heat_capacity(T, theta_D: float = 1.0, n_modes: int = 3,
                        n_grid: int = 2000) -> np.ndarray:
    """Debye model: C_v = 9 N (T/theta_D)^3 \\int_0^{theta_D/T} x^4 e^x/(e^x-1)^2 dx."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    out = np.zeros_like(T)
    for i, Ti in enumerate(T):
        if Ti <= 0.0:
            continue
        xmax = theta_D / Ti
        x = np.linspace(1e-6, xmax, n_grid)
        ex = np.exp(x)
        integrand = x ** 4 * ex / (ex - 1.0) ** 2
        integral = np.trapezoid(integrand, x)
        out[i] = 9.0 * n_modes / 3.0 * (Ti / theta_D) ** 3 * integral
    return out


def debye_lowT_T3(T, theta_D: float = 1.0, n_modes: int = 3) -> np.ndarray:
    """Analytic low-T Debye law: C_v -> (12 pi^4 / 5) N (T/theta_D)^3."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    return (12.0 * np.pi ** 4 / 5.0) * (n_modes / 3.0) * (T / theta_D) ** 3


def _main() -> None:
    from research.phonons.phonon_chain import monatomic_dispersion

    # microscopic route: heat capacity from the 1D chain spectrum
    ks = np.linspace(-np.pi, np.pi, 200, endpoint=False)
    omegas = monatomic_dispersion(ks, K=1.0, m=1.0, a=1.0)
    n_modes = np.sum(omegas > 1e-12)
    print("From the 1D chain phonon spectrum (k_B = hbar = 1):")
    for T in (0.1, 1.0, 100.0):
        cv = heat_capacity_from_spectrum(omegas, T)
        print(f"  T={T:6.1f}:  C_v = {cv:7.2f}   ({cv / n_modes:.3f} per mode)")
    print(f"  high-T limit -> {n_modes} modes (Dulong-Petit, 1D)")

    print("\nDebye vs Einstein (3 modes, theta = 1):")
    for T in (0.05, 0.2, 1.0, 5.0):
        d = debye_heat_capacity(T)[0]
        e = einstein_heat_capacity(T)[0]
        print(f"  T={T:4.2f}:  Debye {d:.4f}   Einstein {e:.4f}")
    Tlow = 0.02
    print(f"\nLow-T Debye: numeric {debye_heat_capacity(Tlow)[0]:.3e} vs "
          f"T^3 law {debye_lowT_T3(Tlow)[0]:.3e}")


if __name__ == "__main__":
    _main()
