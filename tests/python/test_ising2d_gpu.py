"""Validate the GPU Ising path: same physics as CPU, and faster at large L."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from research.criticality.ising2d import simulate as cpu_sim  # noqa: E402
from research.criticality.ising2d_gpu import benchmark, simulate_torch  # noqa: E402


def test_gpu_matches_cpu_physics():
    # at an equilibrating size, away from the deep-quench domain-trapping regime
    # (T well below T_c from a random start can metastably stripe), the GPU path
    # must reproduce the CPU magnetisation.
    for T in (2.0, 3.0):
        mg, _ = simulate_torch(48, T, equil=600, measure=300, device="cuda")
        mc, _ = cpu_sim(L=48, T=T, equil=600, measure=300)
        assert abs(mg - mc) < 0.1           # agree within Monte-Carlo noise


def test_gpu_wins_at_large_lattice():
    rows = benchmark(Ls=(512,), n_sweeps=40)
    L, cpu, gpu, speedup = rows[0]
    assert speedup > 5.0                    # GPU decisively faster at 512x512


def test_gpu_loses_at_tiny_lattice():
    rows = benchmark(Ls=(32,), n_sweeps=40)
    _, cpu, gpu, speedup = rows[0]
    assert speedup < 3.0                    # honest: kernel overhead dominates when small
