"""Kuramoto model — spontaneous synchronization of coupled oscillators.

N phase oscillators, each with its own natural frequency ω_i, nudged toward their
neighbours' phases:

    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i)  =  ω_i + K·r·sin(ψ − θ_i),

where the mean field r·e^{iψ} = (1/N) Σ_j e^{iθ_j}. The order parameter r ∈ [0,1]
measures coherence: r ≈ 0 when the phases are scattered, r → 1 when the population
locks into one rhythm. Below a critical coupling K_c the oscillators drift
independently; above it a macroscopic fraction synchronizes spontaneously — the
same transition behind clapping audiences, flashing fireflies, and neural rhythms.

For Gaussian natural frequencies (σ = 1) the mean-field threshold is
K_c = 2/(π·g(0)) ≈ 1.60. This is the phase-oscillator cousin of Vicsek flocking
and another instance of the order/disorder transition that runs through the lab.
"""

from __future__ import annotations

import numpy as np


class Kuramoto:
    def __init__(self, n: int = 500, K: float = 1.0, omega: np.ndarray | None = None,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.K = K
        self.omega = rng.standard_normal(n) if omega is None else np.asarray(omega, float)
        self.theta = rng.uniform(-np.pi, np.pi, n)

    def order_parameter(self) -> float:
        return float(np.abs(np.exp(1j * self.theta).mean()))

    def step(self, dt: float = 0.05) -> None:
        z = np.exp(1j * self.theta).mean()
        r, psi = np.abs(z), np.angle(z)
        self.theta = self.theta + dt * (self.omega + self.K * r * np.sin(psi - self.theta))

    def run(self, steps: int = 2000, avg_last: int = 600, dt: float = 0.05) -> float:
        """Time-averaged order parameter in the steady state."""
        vals = []
        for t in range(steps):
            self.step(dt)
            if t >= steps - avg_last:
                vals.append(self.order_parameter())
        return float(np.mean(vals))


def sweep_coupling(Ks, n: int = 500, seed: int = 0, steps: int = 1800):
    """Steady-state order parameter r vs coupling K (shared natural frequencies)."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(n)            # drawn once: a clean control-parameter sweep
    return np.array([Kuramoto(n=n, K=float(K), omega=omega, seed=seed).run(steps=steps) for K in Ks])
