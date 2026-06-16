"""Hopfield — figure: the storage-capacity collapse vs load alpha."""

from __future__ import annotations

import os

import numpy as np

from research.neuro.hopfield import capacity_scan


def figure_capacity(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    a, frac, ov = capacity_scan(N=400, alphas=np.linspace(0.02, 0.30, 18), trials=5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a, frac, "o-", color="#1f77b4", lw=2, label="fraction of patterns stable (exact)")
    ax.plot(a, ov, "s-", color="#2ca02c", lw=2, label="mean retrieval overlap")
    ax.axvline(0.138, color="#d62728", ls="--", lw=1.5)
    ax.annotate("α_c ≈ 0.138\n(Amit–Gutfreund–Sompolinsky)", (0.145, 0.55), fontsize=8, color="#d62728")
    ax.set_xlabel("load  α = P / N")
    ax.set_ylabel("fraction / overlap")
    ax.set_ylim(0, 1.05)
    ax.set_title("Hopfield associative memory has a capacity:\n"
                 "stored patterns collapse as load passes α_c ≈ 0.138")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "hopfield_capacity.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_capacity())
