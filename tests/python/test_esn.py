"""Validation of the Echo State Network (ADR-0001, P1 — audit existing physics).

- The reservoir weight matrix is rescaled to the requested spectral radius
  (the core ESN invariant; also the 'spectral radius' that spectra will own).
- The ESN actually learns a standard reservoir-computing benchmark: NARMA-10,
  a 10th-order nonlinear autoregressive task. A mean predictor gives NRMSE ≈ 1;
  a working ESN must come in well below that.

Run standalone:  python tests/python/test_esn.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import linalg as la

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.reservoir_computing.esn import EchoStateNetwork  # noqa: E402


def _narma10(t_len: int, seed: int = 0):
    """Standard NARMA-10 system: input u ~ U(0,0.5), 10th-order recurrence."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=t_len)
    y = np.zeros(t_len)
    for t in range(9, t_len - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - 9 : t + 1])
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
    return u, y


def test_esn_spectral_radius_is_rescaled():
    """W must be scaled so its spectral radius equals the requested value."""
    sr = 0.9
    esn = EchoStateNetwork(reservoir_size=200, spectral_radius=sr, seed=1)
    rho = float(np.max(np.abs(la.eigvals(esn.W))))
    assert np.isclose(rho, sr, rtol=1e-6), f"spectral radius {rho:.4f} != {sr}"


def test_esn_learns_narma10():
    """ESN learns NARMA-10 with NRMSE well below the trivial baseline of 1.0."""
    t_len = 3000
    u, y = _narma10(t_len, seed=0)
    X, Y = u.reshape(-1, 1), y.reshape(-1, 1)
    split = int(0.7 * t_len)

    esn = EchoStateNetwork(
        input_dim=1, reservoir_size=300, output_dim=1,
        spectral_radius=0.9, leak_rate=0.3, input_scaling=0.2, seed=42,
    )
    esn.fit(X[:split], Y[:split], warmup=200, ridge=1e-6)
    # Predict the full series so the reservoir is warm by the test split.
    y_pred = esn.predict(X)[split:]
    y_true = Y[split:]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nrmse = rmse / float(np.std(y_true))
    assert nrmse < 0.8, f"NARMA-10 NRMSE {nrmse:.3f} too high (mean baseline ~1.0)"


def _run_standalone() -> int:
    tests = [test_esn_spectral_radius_is_rescaled, test_esn_learns_narma10]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
