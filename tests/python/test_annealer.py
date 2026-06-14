"""Validation of the simulated-annealing QUBO solver (ADR-0001, P1).

On small QUBOs the global optimum is computable by brute force, so we can check
that simulated annealing actually finds it (not just "a low value"), and that the
energy it reports matches its returned sample.

Run standalone:  python tests/python/test_annealer.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.quantum_annealing.annealer import MaterialAnnealer  # noqa: E402


def _qubo_energy(qubo: dict, state) -> float:
    """QUBO energy E = Σ_(i,j) val · s_i · s_j (matches the solver's convention)."""
    return float(sum(v * state[i] * state[j] for (i, j), v in qubo.items()))


def _brute_force_min(qubo: dict, n: int) -> float:
    return min(_qubo_energy(qubo, bits) for bits in itertools.product((0, 1), repeat=n))


def _random_qubo(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q: dict[tuple[int, int], float] = {}
    for i in range(n):
        q[(i, i)] = float(rng.normal())  # linear terms on the diagonal
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                q[(i, j)] = float(rng.normal())  # couplings
    return q


def test_sa_finds_brute_force_optimum():
    """SA must recover the true global minimum of a small QUBO."""
    n = 10
    qubo = _random_qubo(n, seed=3)
    bf_min = _brute_force_min(qubo, n)
    res = MaterialAnnealer(method="SA", num_reads=80, seed=0).anneal(qubo)
    assert np.isclose(res.energy, bf_min, atol=1e-9), f"SA {res.energy:.6f} != optimum {bf_min:.6f}"


def test_sa_energy_matches_sample():
    """The reported energy must equal the QUBO energy of the returned sample."""
    n = 8
    qubo = _random_qubo(n, seed=5)
    res = MaterialAnnealer(method="SA", num_reads=40, seed=1).anneal(qubo)
    recomputed = _qubo_energy(qubo, res.sample)
    assert np.isclose(recomputed, res.energy, atol=1e-9), f"{recomputed} != {res.energy}"


def _run_standalone() -> int:
    tests = [test_sa_finds_brute_force_optimum, test_sa_energy_matches_sample]
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
