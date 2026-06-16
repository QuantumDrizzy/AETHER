"""Hopfield network — associative memory and its storage capacity.

A fully-connected network of ±1 neurons with Hebbian weights
W_ij = (1/N) Σ_μ ξ^μ_i ξ^μ_j (zero diagonal) stores patterns ξ^μ as attractors:
present a noisy version and the dynamics s_i ← sign(Σ_j W_ij s_j) relax to the
nearest stored memory. It is the neural-memory face of DRIFT's Ising engine (recall
= rolling downhill to a ground state) and the substrate behind Hopfield's 2024
Nobel.

But memory is finite. As the load α = P/N rises, stored patterns stop being stable
and recall collapses — the Amit–Gutfreund–Sompolinsky capacity, α_c ≈ 0.138. This
module measures the collapse: the fraction of stored patterns that survive as fixed
points (and the retrieval overlap) as a function of α, recovering a crossover near
the textbook value.
"""

from __future__ import annotations

import numpy as np


class Hopfield:
    def __init__(self, n: int):
        self.n = n
        self.W = np.zeros((n, n))

    def store(self, patterns: np.ndarray) -> "Hopfield":
        """Hebbian storage of patterns, shape (P, N), entries ±1."""
        self.W = (patterns.T @ patterns) / self.n
        np.fill_diagonal(self.W, 0.0)
        return self

    def update(self, s: np.ndarray) -> np.ndarray:
        """One synchronous update step."""
        out = np.sign(self.W @ s)
        out[out == 0] = 1
        return out

    def recall(self, s: np.ndarray, steps: int = 5) -> np.ndarray:
        for _ in range(steps):
            s = self.update(s)
        return s


def random_patterns(P: int, N: int, rng) -> np.ndarray:
    return np.where(rng.random((P, N)) < 0.5, -1, 1).astype(float)


def capacity_scan(N: int = 200, alphas=None, trials: int = 4, seed: int = 0):
    """For each load α=P/N, the fraction of stored patterns that are exact fixed
    points and the mean retrieval overlap. Returns (alphas, frac_stable, overlap)."""
    if alphas is None:
        alphas = np.linspace(0.02, 0.30, 15)
    frac, ov = [], []
    for a in alphas:
        P = max(1, int(round(a * N)))
        fs, os_ = [], []
        for s in range(trials):
            rng = np.random.default_rng(seed + s)
            pat = random_patterns(P, N, rng)
            net = Hopfield(N).store(pat)
            updated = np.sign(pat @ net.W)          # one sync update of each pattern
            updated[updated == 0] = 1
            errors = np.sum(updated != pat, axis=1)   # bit errors per pattern
            fs.append(np.mean(errors == 0))
            os_.append(np.mean((updated * pat).sum(axis=1) / N))
        frac.append(np.mean(fs))
        ov.append(np.mean(os_))
    return np.asarray(alphas), np.asarray(frac), np.asarray(ov)
