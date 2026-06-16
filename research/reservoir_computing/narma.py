"""NARMA-10 benchmark — reservoir task performance vs the edge of chaos.

Memory capacity (DRIFT) measures how much a reservoir remembers; this measures how
well it actually *computes* a hard task. NARMA-10 is the standard reservoir-
computing benchmark: a 10th-order nonlinear auto-regressive moving average driven
by random input,

    y(t+1) = 0.3 y(t) + 0.05 y(t) Σ_{i=0..9} y(t−i) + 1.5 u(t−9) u(t) + 0.1,
    u(t) ~ Uniform[0, 0.5].

It demands both memory (10 steps back) and nonlinearity. We train AETHER's echo-
state network to reproduce it and report the test NRMSE as a function of the
spectral radius ρ. The error is minimised at an intermediate ρ near the **edge of
chaos** — the same operating point where capacity peaks (DRIFT) and where the
universal-criticality experiment puts the order/disorder boundary. Too stable
(low ρ) forgets; too chaotic (high ρ) is unusable.
"""

from __future__ import annotations

import numpy as np

from research.reservoir_computing.esn import EchoStateNetwork


def narma10(u: np.ndarray) -> np.ndarray:
    """Generate the NARMA-10 target series from input series u."""
    n = len(u)
    y = np.zeros(n)
    for t in range(9, n - 1):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t]
                    + 0.1)
    return y


def _nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.ravel(), target.ravel()
    return float(np.sqrt(np.mean((pred - target) ** 2) / (np.var(target) + 1e-12)))


def narma_nrmse(spectral_radius: float, n_train: int = 1600, n_test: int = 800,
                reservoir_size: int = 200, seed: int = 0) -> float:
    """Train an ESN on NARMA-10 and return the test NRMSE for a given ρ."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, n_train + n_test + 50)
    y = narma10(u)
    X = u.reshape(-1, 1)
    Y = y.reshape(-1, 1)

    esn = EchoStateNetwork(reservoir_size=reservoir_size, spectral_radius=spectral_radius,
                           leak_rate=0.3, input_scaling=0.2, seed=seed)
    esn.fit(X[:n_train], Y[:n_train], warmup=100)
    pred = esn.predict(X[n_train:n_train + n_test])
    return _nrmse(pred[50:], Y[n_train:n_train + n_test][50:])


def sweep_spectral_radius(radii, **kw) -> np.ndarray:
    return np.array([narma_nrmse(float(r), **kw) for r in radii])
