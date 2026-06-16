"""2D Ising model by Monte Carlo — the gold-standard critical point.

The mean-field Ising in `universal.py` had a smeared, size-dependent transition;
the real 2D square-lattice Ising has an *exact* one. Metropolis sampling at
temperature T (J = 1, periodic boundary) gives the magnetisation and magnetic
susceptibility, and the susceptibility peaks at the Onsager critical temperature

    T_c = 2 / ln(1 + √2) ≈ 2.269.

Below T_c the lattice is ordered (|m| → 1); above it disordered (|m| → 0); at T_c
fluctuations diverge (χ peaks). This is the canonical second-order phase transition
that anchors the whole criticality thread — now measured, not assumed. Vectorised
checkerboard updates keep it fast.
"""

from __future__ import annotations

import numpy as np

T_C_ONSAGER = 2.0 / np.log(1.0 + np.sqrt(2.0))     # ≈ 2.269


def _checkerboard_sweep(s: np.ndarray, T: float, rng) -> None:
    L = s.shape[0]
    ii, jj = np.indices((L, L))
    for color in (0, 1):
        nb = (np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1))
        dE = 2.0 * s * nb                          # J = 1
        accept = (dE < 0) | (rng.random((L, L)) < np.exp(-dE / T))
        flip = accept & (((ii + jj) % 2) == color)
        s[flip] *= -1


def simulate(L: int = 24, T: float = 2.0, equil: int = 300, measure: int = 400,
             seed: int = 0):
    """Return (mean |m|, susceptibility χ) at temperature T."""
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1.0, 1.0]), size=(L, L))
    for _ in range(equil):
        _checkerboard_sweep(s, T, rng)
    N = L * L
    absM, M2 = [], []
    for _ in range(measure):
        _checkerboard_sweep(s, T, rng)
        M = s.sum()
        absM.append(abs(M))
        M2.append(M * M)
    absM = np.array(absM); M2 = np.array(M2)
    m = float(absM.mean() / N)
    chi = float((M2.mean() - absM.mean() ** 2) / (N * T))
    return m, chi


def sweep_temperature(Ts, L: int = 24, **kw):
    """Return (|m|, χ) arrays across the temperature sweep."""
    ms, chis = [], []
    for T in Ts:
        m, chi = simulate(L=L, T=float(T), **kw)
        ms.append(m); chis.append(chi)
    return np.asarray(ms), np.asarray(chis)


def critical_temperature(L: int = 24, Ts=None, **kw) -> float:
    """Estimate T_c as the temperature where the susceptibility peaks."""
    if Ts is None:
        Ts = np.linspace(1.6, 3.2, 17)
    _, chi = sweep_temperature(Ts, L=L, **kw)
    return float(Ts[int(np.argmax(chi))])
