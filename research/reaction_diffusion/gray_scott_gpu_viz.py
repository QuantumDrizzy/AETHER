"""GPU Gray-Scott — figure: CPU-vs-GPU speedup vs grid size."""

from __future__ import annotations

import os

import numpy as np

from research.reaction_diffusion.gray_scott_gpu import benchmark, best_device


def figure_speedup(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    rows = benchmark(sizes=(128, 256, 512, 1024), n_steps=150)
    Ls = np.array([r[0] for r in rows])
    speed = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(Ls, speed, "o-", color="#9467bd", lw=2, ms=7)
    ax.axhline(1.0, color="#d62728", ls="--", lw=1.5)
    ax.annotate("break-even", (Ls[0], 1.1), fontsize=8, color="#d62728")
    for L, sp in zip(Ls, speed):
        ax.annotate(f"{sp:.0f}x" if sp >= 10 else f"{sp:.1f}x",
                    (L, sp), textcoords="offset points", xytext=(0, 7), fontsize=8)
    ax.set_xlabel("grid size  L  (L×L cells)")
    ax.set_ylabel("GPU speedup over CPU")
    ax.set_title(f"Gray-Scott reaction-diffusion on {best_device().upper()}:\n"
                 "a stencil over a large grid is where CUDA pays")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "gray_scott_gpu_speedup.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_speedup())
