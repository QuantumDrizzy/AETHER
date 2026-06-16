"""GPU Ising — figure: CPU-vs-GPU speedup vs lattice size (where CUDA pays)."""

from __future__ import annotations

import os

import numpy as np

from research.criticality.ising2d_gpu import benchmark, best_device


def figure_speedup(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    rows = benchmark(Ls=(32, 64, 128, 256, 512, 1024), n_sweeps=60)
    Ls = np.array([r[0] for r in rows])
    speed = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(Ls, speed, "o-", color="#2ca02c", lw=2, ms=7)
    ax.axhline(1.0, color="#d62728", ls="--", lw=1.5)
    ax.annotate("break-even (GPU loses below)", (Ls[0], 1.1), fontsize=8, color="#d62728")
    for L, sp in zip(Ls, speed):
        label = f"{sp:.1f}x" if sp < 10 else f"{sp:.0f}x"
        ax.annotate(label, (L, sp), textcoords="offset points", xytext=(0, 7), fontsize=8)
    ax.set_xlabel("lattice size  L  (L×L spins)")
    ax.set_ylabel("GPU speedup over CPU")
    ax.set_title(f"2D Ising Monte Carlo on {best_device().upper()}:\n"
                 "CUDA loses on small lattices, wins decisively on large ones")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ising2d_gpu_speedup.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_speedup())
