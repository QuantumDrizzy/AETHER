"""GPU Anderson — figure: dense eigensolve speedup (CPU vs GPU) vs matrix size."""

from __future__ import annotations

import os

import numpy as np

from research.electronic_structure.anderson_gpu import benchmark, best_device


def figure_speedup(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    rows = benchmark(Ls=(500, 1000, 2000, 3000, 4000))
    Ls = np.array([r[0] for r in rows])
    speed = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ls, speed, "o-", color="#ff7f0e", lw=2, ms=7)
    ax.axhline(1.0, color="#d62728", ls="--", lw=1.2)
    for L, sp in zip(Ls, speed):
        ax.annotate(f"{sp:.0f}x", (L, sp), textcoords="offset points", xytext=(0, 7), fontsize=8)
    ax.set_xlabel("matrix size  L  (L×L Hamiltonian)")
    ax.set_ylabel("GPU eigh speedup over CPU")
    ax.set_title(f"Anderson localization on {best_device().upper()}:\n"
                 "dense diagonalization (torch.linalg.eigh) — the 3rd HPC pattern CUDA wins")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "anderson_gpu_speedup.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_speedup())
