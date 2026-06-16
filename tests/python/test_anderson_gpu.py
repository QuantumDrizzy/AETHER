"""Validate the GPU Anderson path: same PR physics as CPU, faster eigensolve."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from research.electronic_structure.anderson import mean_participation_ratio as cpu_pr  # noqa: E402
from research.electronic_structure.anderson_gpu import (  # noqa: E402
    benchmark,
    mean_participation_ratio_torch,
)


def test_gpu_physics_matches_cpu():
    # self-averaging: GPU and CPU mean PR agree even with different disorder draws
    for W in (1.0, 4.0):
        g = mean_participation_ratio_torch(800, W, device="cuda")
        c = cpu_pr(800, W, trials=2)
        assert abs(g - c) / c < 0.1
    # and the localisation trend holds on GPU
    assert mean_participation_ratio_torch(800, 1.0, device="cuda") > \
        mean_participation_ratio_torch(800, 8.0, device="cuda")


def test_gpu_eigh_faster_at_large_L():
    L, cpu, gpu, speedup = benchmark(Ls=(2000,))[0]
    assert speedup > 3.0
