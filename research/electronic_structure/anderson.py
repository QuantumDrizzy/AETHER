"""Anderson localization — disorder turns a metal into an insulator.

Take a clean 1D tight-binding chain (nearest-neighbour hopping t) and add random
on-site energies ε_i ~ Uniform[−W/2, W/2]. In the clean limit the eigenstates are
extended Bloch waves spread over the whole chain; switch on disorder and they
localise exponentially — the wavefunction collapses onto a few sites and transport
dies. In 1D this happens for *any* W > 0 (Anderson 1958): there is no metal in one
disordered dimension.

We quantify spread with the participation ratio

    PR(ψ) = 1 / Σ_i |ψ_i|^4 ,

the effective number of sites a state occupies: PR ≈ 2L/3 for an extended state,
PR = O(1) for a localised one. The mean PR over the spectrum collapses as W grows
— a clean, measurable localisation crossover, built on the same tight-binding
machinery as graphene/SSH/Haldane.
"""

from __future__ import annotations

import numpy as np


def _hamiltonian(L: int, W: float, t: float, rng) -> np.ndarray:
    h = np.zeros((L, L))
    diag = rng.uniform(-W / 2.0, W / 2.0, L)
    np.fill_diagonal(h, diag)
    idx = np.arange(L - 1)
    h[idx, idx + 1] = -t
    h[idx + 1, idx] = -t
    return h


def participation_ratios(L: int = 300, W: float = 1.0, t: float = 1.0, seed: int = 0) -> np.ndarray:
    """PR of every eigenstate of one disorder realisation."""
    rng = np.random.default_rng(seed)
    _, vecs = np.linalg.eigh(_hamiltonian(L, W, t, rng))
    ipr = np.sum(vecs ** 4, axis=0)          # columns are eigenvectors, already normalised
    return 1.0 / ipr


def mean_participation_ratio(L: int = 300, W: float = 1.0, t: float = 1.0,
                             trials: int = 8, seed: int = 0) -> float:
    """Spectrum- and disorder-averaged participation ratio."""
    vals = []
    for s in range(trials):
        vals.append(participation_ratios(L, W, t, seed + s).mean())
    return float(np.mean(vals))


def sweep_disorder(W_values, L: int = 300, trials: int = 8, seed: int = 0):
    return np.array([mean_participation_ratio(L, float(W), trials=trials, seed=seed) for W in W_values])
