"""Kuramoto — figure: the synchronization transition r vs coupling K."""

from __future__ import annotations

import os

import numpy as np

from research.active_matter.kuramoto import sweep_coupling


def figure_sync(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    Ks = np.linspace(0.0, 4.0, 21)
    r = sweep_coupling(Ks, n=600, steps=1800)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ks, r, "o-", color="#1f77b4", lw=2)
    ax.axvline(1.60, color="#d62728", ls="--", lw=1.5)
    ax.annotate("K_c ≈ 1.60\n(Gaussian ω)", (1.65, 0.2), fontsize=8, color="#d62728")
    ax.set_xlabel("coupling  K")
    ax.set_ylabel("order parameter  r")
    ax.set_ylim(0, 1.02)
    ax.set_title("Kuramoto: oscillators synchronize spontaneously above K_c\n"
                 "(incoherent drift below; one shared rhythm above)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "kuramoto_sync.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_sync())
