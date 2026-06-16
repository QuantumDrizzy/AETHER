"""Validate the GPU reaction-diffusion path: forms pattern, faster at large grids."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from research.reaction_diffusion.gray_scott_gpu import benchmark, simulate_torch  # noqa: E402


def test_gpu_forms_pattern():
    v = simulate_torch(128, steps=3000, device="cuda", seed=0)
    assert np.all(np.isfinite(v)) and v.min() >= 0 and v.max() <= 1
    assert v.std() > 0.06          # the maze seed self-organises (same as CPU)


def test_gpu_wins_at_large_grid():
    L, cpu, gpu, speedup = benchmark(sizes=(512,), n_steps=120)[0]
    assert speedup > 5.0           # GPU decisively faster at 512x512


def test_gpu_overhead_dominates_when_small():
    L, cpu, gpu, speedup = benchmark(sizes=(128,), n_steps=120)[0]
    assert speedup < 3.0           # honest: little benefit on a small grid
