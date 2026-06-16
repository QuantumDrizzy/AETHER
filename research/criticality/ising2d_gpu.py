"""2D Ising Monte Carlo on the GPU — where CUDA actually pays.

The checkerboard Metropolis update is embarrassingly parallel: every spin on one
sublattice flips independently given the other, so a whole L×L lattice updates in
one shot. On the CPU that is a NumPy sweep; on the GPU it is millions of spins per
kernel — exactly the regime where the hardware earns its keep (unlike a 500-node
Kuramoto, where launch overhead would lose). Same physics as `ising2d.py`,
validated to give the same magnetisation and Onsager T_c; this module adds the
PyTorch path and an honest CPU-vs-GPU benchmark with the crossover.

Run the benchmark:  python -m research.criticality.ising2d_gpu
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:                       # pragma: no cover
    _HAS_TORCH = False


def best_device() -> str:
    return "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"


def _sweep_torch(s, mask, T: float, gen) -> None:
    for color in (mask, ~mask):
        nb = (torch.roll(s, 1, 0) + torch.roll(s, -1, 0)
              + torch.roll(s, 1, 1) + torch.roll(s, -1, 1))
        dE = 2.0 * s * nb                  # J = 1
        rand = torch.rand(s.shape, device=s.device, generator=gen)
        accept = (dE < 0) | (rand < torch.exp(-dE / T))
        flip = accept & color
        s = torch.where(flip, -s, s)
    # in-place via copy (torch.where returns a new tensor)
    return s


def simulate_torch(L: int = 256, T: float = 2.27, equil: int = 200, measure: int = 200,
                   device: str | None = None, seed: int = 0):
    """(mean |m|, susceptibility χ) at temperature T, on `device` (default best)."""
    dev = device or best_device()
    gen = torch.Generator(device=dev).manual_seed(seed)
    s = torch.where(torch.rand((L, L), device=dev, generator=gen) < 0.5,
                    torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
    ii, jj = torch.meshgrid(torch.arange(L, device=dev), torch.arange(L, device=dev), indexing="ij")
    mask = ((ii + jj) % 2 == 0)
    for _ in range(equil):
        s = _sweep_torch(s, mask, T, gen)
    N = L * L
    absM, M2 = [], []
    for _ in range(measure):
        s = _sweep_torch(s, mask, T, gen)
        M = float(s.sum().item())
        absM.append(abs(M)); M2.append(M * M)
    absM = np.array(absM); M2 = np.array(M2)
    return float(absM.mean() / N), float((M2.mean() - absM.mean() ** 2) / (N * T))


def benchmark(Ls=(64, 128, 256, 512, 1024), n_sweeps: int = 60, T: float = 2.27):
    """Time CPU (NumPy) vs GPU (torch) checkerboard sweeps for each lattice size L."""
    import time

    from research.criticality.ising2d import _checkerboard_sweep
    rows = []
    for L in Ls:
        rng = np.random.default_rng(0)
        s_np = rng.choice(np.array([-1.0, 1.0]), size=(L, L))
        for _ in range(3):
            _checkerboard_sweep(s_np, T, rng)            # warm up
        t0 = time.perf_counter()
        for _ in range(n_sweeps):
            _checkerboard_sweep(s_np, T, rng)
        cpu = time.perf_counter() - t0

        gpu = float("nan")
        if best_device() == "cuda":
            dev = "cuda"
            gen = torch.Generator(device=dev).manual_seed(0)
            s = torch.where(torch.rand((L, L), device=dev, generator=gen) < 0.5,
                            torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
            ii, jj = torch.meshgrid(torch.arange(L, device=dev), torch.arange(L, device=dev), indexing="ij")
            mask = ((ii + jj) % 2 == 0)
            for _ in range(3):
                s = _sweep_torch(s, mask, T, gen)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_sweeps):
                s = _sweep_torch(s, mask, T, gen)
            torch.cuda.synchronize()
            gpu = time.perf_counter() - t0
        rows.append((L, cpu, gpu, cpu / gpu if gpu == gpu else float("nan")))
    return rows


def _main() -> None:
    print(f"device: {best_device()}")
    print(f"  {'L':>5}  {'CPU (s)':>9}  {'GPU (s)':>9}  {'speedup':>8}")
    for L, cpu, gpu, sp in benchmark():
        print(f"  {L:>5}  {cpu:>9.3f}  {gpu:>9.3f}  {sp:>7.1f}x")


if __name__ == "__main__":
    _main()
