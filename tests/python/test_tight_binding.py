"""Validation of the general N×N tight-binding solver (ADR-0001, spine).

- The general solver, fed graphene's hoppings, must reproduce the closed-form
  bands from graphene.py (the anchor of correctness).
- H(k) is Hermitian for arbitrary models.
- Eigenvalues are real and ascending.
- The GPU path agrees with the CPU path (skipped if no CUDA).

Run standalone:  python tests/python/test_tight_binding.py
Or with pytest:  python -m pytest tests/python/test_tight_binding.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.graphene import GrapheneTB  # noqa: E402
from research.electronic_structure.tight_binding import (  # noqa: E402
    graphene_model,
    random_model,
)


def test_graphene_general_matches_closed_form():
    """General solver on graphene == closed-form ±t|f(k)|."""
    closed = GrapheneTB()
    general = graphene_model()
    rng = np.random.default_rng(3)
    k = rng.uniform(-3.0, 3.0, size=(300, 2))
    e_minus, e_plus = closed.bands(k)
    ref = np.stack([e_minus, e_plus], axis=1)  # (300, 2), ascending
    got = general.bands(k)  # (300, 2), ascending
    assert np.allclose(got, ref, atol=1e-10)


def test_hamiltonian_is_hermitian():
    model = random_model(n_orbitals=8, n_hoppings=20, seed=2)
    rng = np.random.default_rng(4)
    h = model.hamiltonian_batch(rng.uniform(-3.0, 3.0, size=(10, 2)))
    assert np.allclose(h, np.conj(np.transpose(h, (0, 2, 1))), atol=1e-12)


def test_eigenvalues_real_and_sorted():
    model = random_model(n_orbitals=12, n_hoppings=30, seed=5)
    rng = np.random.default_rng(6)
    w = model.bands(rng.uniform(-3.0, 3.0, size=(50, 2)))
    assert w.shape == (50, 12)
    assert np.all(np.diff(w, axis=1) >= -1e-9)  # ascending


def test_gpu_matches_cpu():
    try:
        import torch

        if not torch.cuda.is_available():
            print("  SKIP test_gpu_matches_cpu: no CUDA")
            return
    except Exception:
        print("  SKIP test_gpu_matches_cpu: torch unavailable")
        return
    model = random_model(n_orbitals=32, n_hoppings=80, seed=7)
    rng = np.random.default_rng(8)
    k = rng.uniform(-3.0, 3.0, size=(500, 2))
    cpu = model.bands(k)
    gpu = model.bands_torch(k, device="cuda")
    assert np.allclose(cpu, gpu, atol=1e-4, rtol=1e-4)


def _run_standalone() -> int:
    tests = [
        test_graphene_general_matches_closed_form,
        test_hamiltonian_is_hermitian,
        test_eigenvalues_real_and_sorted,
        test_gpu_matches_cpu,
    ]
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
