"""Small-world — figure: the Watts-Strogatz C(p) / L(p) signature."""

from __future__ import annotations

import os

import numpy as np

from research.networks.small_world import sweep_rewiring


def figure_small_world(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    ps = np.logspace(-3, 0, 16)
    C, L = sweep_rewiring(ps, n=240, k=6)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(ps, C, "o-", color="#2ca02c", lw=2, label="C(p) / C(0)  — clustering")
    ax.semilogx(ps, L, "s-", color="#1f77b4", lw=2, label="L(p) / L(0)  — path length")
    ax.axvspan(0.005, 0.1, color="#d62728", alpha=0.08)
    ax.annotate("small-world window:\nshort paths, still clustered", (0.006, 0.45),
                fontsize=8, color="#a00")
    ax.set_xlabel("rewiring probability  p")
    ax.set_ylabel("normalised")
    ax.set_ylim(0, 1.05)
    ax.set_title("Watts-Strogatz small-world:\npath length collapses while clustering stays high")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "small_world.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_small_world())
