"""Anderson localization on the GPU — dense diagonalization where CUDA pays.

The third HPC pattern in the lab where the GPU earns its keep: not a lattice update
(Ising) or a stencil (Gray-Scott) but **dense linear algebra**. Computing the
participation ratio needs the eigenvectors of the L×L disordered Hamiltonian, and
`torch.linalg.eigh` on the GPU beats LAPACK-on-CPU once L is large (thousands).
Same physics as `anderson.py` (PR collapses with disorder); this adds the PyTorch
eigensolver path and an honest CPU-vs-GPU benchmark.

Run the benchmark:  python -m research.electronic_structure.anderson_gpu
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


def _build_torch(L: int, W: float, t: float, device: str, gen):
    diag = (torch.rand(L, device=device, generator=gen) - 0.5) * W
    H = torch.diag(diag)
    off = -t * torch.ones(L - 1, device=device)
    H += torch.diag(off, 1) + torch.diag(off, -1)
    return H


def mean_participation_ratio_torch(L: int = 1000, W: float = 2.0, t: float = 1.0,
                                   device: str | None = None, seed: int = 0) -> float:
    dev = device or best_device()
    gen = torch.Generator(device=dev).manual_seed(seed)
    H = _build_torch(L, W, t, dev, gen)
    _, vecs = torch.linalg.eigh(H)
    ipr = (vecs ** 4).sum(dim=0)
    return float((1.0 / ipr).mean().item())


def benchmark(Ls=(500, 1000, 2000, 3000), W: float = 2.0):
    """Time the eigensolve (eigh) on CPU (NumPy) vs GPU (torch) per matrix size."""
    import time
    rows = []
    for L in Ls:
        rng = np.random.default_rng(0)
        diag = (rng.random(L) - 0.5) * W
        H = np.diag(diag) + np.diag(-np.ones(L - 1), 1) + np.diag(-np.ones(L - 1), -1)
        t0 = time.perf_counter()
        np.linalg.eigh(H)
        cpu = time.perf_counter() - t0

        gpu = float("nan")
        if best_device() == "cuda":
            Ht = torch.tensor(H, device="cuda")
            torch.linalg.eigh(Ht)            # warm up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.linalg.eigh(Ht)
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
