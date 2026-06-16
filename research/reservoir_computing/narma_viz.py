"""NARMA-10 — figure: task error vs spectral radius (optimum below the edge)."""

from __future__ import annotations

import os

import numpy as np

from research.reservoir_computing.narma import sweep_spectral_radius


def figure_narma(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    radii = np.linspace(0.1, 1.4, 18)
    nr = sweep_spectral_radius(radii, reservoir_size=200, seed=0)
    best = radii[int(np.argmin(nr))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(radii, nr, "o-", color="#1f77b4", lw=2)
    ax.axvline(1.0, color="#d62728", ls="--", lw=1.5)
    ax.annotate("edge of chaos\n(echo-state property lost)", (1.02, nr.max() * 0.3),
                fontsize=8, color="#d62728")
    ax.axvline(best, color="#2ca02c", ls=":", lw=1.5)
    ax.annotate(f"best ρ ≈ {best:.2f}", (best - 0.35, nr.min() * 1.2), fontsize=9, color="#2ca02c")
    ax.set_xlabel("spectral radius  ρ")
    ax.set_ylabel("NARMA-10 test NRMSE  (log)")
    ax.set_title("Reservoir task performance: best below the edge, ruined above it\n"
                 "(too stable forgets; chaos is unusable)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "narma_nrmse.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_narma())
