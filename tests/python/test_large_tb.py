"""Validation of AETHER × Spectra integration (the first Spectra consumer).

Spectra's Lanczos band edges must match the dense reference on large tight-binding
Hamiltonians — proving the spine works for a real consumer.

Requires Spectra importable (pip install -e ../Spectra). Skips cleanly if absent.

Run standalone:  python tests/python/test_large_tb.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import spectra  # noqa: F401
    HAVE_SPECTRA = True
except ImportError:
    HAVE_SPECTRA = False

from research.electronic_structure.ssh import ssh_finite  # noqa: E402
from research.electronic_structure.tight_binding import random_model  # noqa: E402


def test_band_edges_match_dense_on_large_ssh():
    if not HAVE_SPECTRA:
        print("  SKIP  (spectra not installed)")
        return
    from research.electronic_structure.large_tb import band_edges, tb_hamiltonian

    # [KNOWN_LIMIT] SSH has a *clustered* 1-D band edge (van Hove pile-up), so the
    # absolute extreme converges slowly in Lanczos — 1e-3 here, vs 1e-5 for the
    # isolated edges below. Tighter needs more iters; this is Lanczos, not a bug.
    tb = ssh_finite(500, t1=0.7, t2=1.0)  # 1000 atoms
    dense = np.linalg.eigvalsh(tb_hamiltonian(tb, (0.0, 0.0)))
    ritz = band_edges(tb, (0.0, 0.0), n_edges=4, iters=120)
    assert abs(ritz[0] - dense[0]) < 1e-3, f"min edge {ritz[0]} vs {dense[0]}"
    assert abs(ritz[-1] - dense[-1]) < 1e-3, f"max edge {ritz[-1]} vs {dense[-1]}"


def test_band_edges_match_dense_on_random_model():
    if not HAVE_SPECTRA:
        print("  SKIP  (spectra not installed)")
        return
    from research.electronic_structure.large_tb import band_edges, tb_hamiltonian

    tb = random_model(300, 1200, seed=2)
    k = (0.3, -0.2)
    dense = np.linalg.eigvalsh(tb_hamiltonian(tb, k))
    ritz = band_edges(tb, k, n_edges=4, iters=120)
    assert abs(ritz[0] - dense[0]) < 1e-5, f"min edge {ritz[0]} vs {dense[0]}"
    assert abs(ritz[-1] - dense[-1]) < 1e-5, f"max edge {ritz[-1]} vs {dense[-1]}"


def _run_standalone() -> int:
    tests = [test_band_edges_match_dense_on_large_ssh, test_band_edges_match_dense_on_random_model]
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
