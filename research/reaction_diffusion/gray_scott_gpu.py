"""Gray-Scott reaction-diffusion on the GPU — large grids where CUDA pays.

Reaction-diffusion is a stencil over a grid: every cell updates from its neighbours
each step, independently — the same embarrassingly-parallel shape as the Ising
checkerboard. On a 1024×1024 field over thousands of steps that is a million cells
per kernel, exactly where the GPU earns its keep. Same dynamics as `gray_scott.py`
(maze preset), validated to grow the same structure; this adds the PyTorch path and
an honest CPU-vs-GPU benchmark.

Run the benchmark:  python -m research.reaction_diffusion.gray_scott_gpu
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:                       # pragma: no cover
    _HAS_TORCH = False

F_MAZE, K_MAZE, DU, DV = 0.029, 0.057, 0.16, 0.08


def best_device() -> str:
    return "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"


def _lap(a):
    return (-1.0 * a
            + 0.20 * (torch.roll(a, 1, 0) + torch.roll(a, -1, 0)
                      + torch.roll(a, 1, 1) + torch.roll(a, -1, 1))
            + 0.05 * (torch.roll(torch.roll(a, 1, 0), 1, 1) + torch.roll(torch.roll(a, 1, 0), -1, 1)
                      + torch.roll(torch.roll(a, -1, 0), 1, 1) + torch.roll(torch.roll(a, -1, 0), -1, 1)))


def simulate_torch(size: int = 256, steps: int = 4000, device: str | None = None, seed: int = 0):
    """Run the maze preset on `device`; return the final v field as a NumPy array."""
    dev = device or best_device()
    gen = torch.Generator(device=dev).manual_seed(seed)
    u = torch.ones((size, size), device=dev)
    v = torch.zeros((size, size), device=dev)
    r = size // 10
    c = size // 2
    u[c - r:c + r, c - r:c + r] = 0.5
    v[c - r:c + r, c - r:c + r] = 0.25
    v = v + 0.02 * torch.rand((size, size), device=dev, generator=gen)
    for _ in range(steps):
        uvv = u * v * v
        u = torch.clamp(u + DU * _lap(u) - uvv + F_MAZE * (1 - u), 0, 1)
        v = torch.clamp(v + DV * _lap(v) + uvv - (F_MAZE + K_MAZE) * v, 0, 1)
    return v.cpu().numpy()


def benchmark(sizes=(128, 256, 512, 1024), n_steps: int = 300):
    """Time CPU (NumPy) vs GPU (torch) reaction-diffusion steps per grid size."""
    import time

    from research.reaction_diffusion.gray_scott import GrayScott
    rows = []
    for L in sizes:
        gs = GrayScott()
        u, v = gs.seeded_init(L, seed=0)
        for _ in range(3):
            u, v = gs.step(u, v)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            u, v = gs.step(u, v)
        cpu = time.perf_counter() - t0

        gpu = float("nan")
        if best_device() == "cuda":
            dev = "cuda"
            a = torch.ones((L, L), device=dev)
            b = torch.rand((L, L), device=dev)
            for _ in range(3):
                a = torch.clamp(a + DU * _lap(a), 0, 1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_steps):
                uvv = a * b * b
                a = torch.clamp(a + DU * _lap(a) - uvv + F_MAZE * (1 - a), 0, 1)
                b = torch.clamp(b + DV * _lap(b) + uvv - (F_MAZE + K_MAZE) * b, 0, 1)
            torch.cuda.synchronize()
            gpu = time.perf_counter() - t0
        rows.append((L, cpu, gpu, cpu / gpu if gpu == gpu else float("nan")))
    return rows


def _main() -> None:
    print(f"device: {best_device()}")
    print(f"  {'grid':>6}  {'CPU (s)':>9}  {'GPU (s)':>9}  {'speedup':>8}")
    for L, cpu, gpu, sp in benchmark():
        print(f"  {L:>4}^2  {cpu:>9.3f}  {gpu:>9.3f}  {sp:>7.1f}x")


if __name__ == "__main__":
    _main()
