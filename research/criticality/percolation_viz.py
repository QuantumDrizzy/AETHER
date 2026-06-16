"""Percolation — figure: the spanning transition sharpening toward p_c with size."""

from __future__ import annotations

import os

import numpy as np

from research.criticality.percolation import P_C_SQUARE_SITE, spanning_probability


def figure_percolation(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    ps = np.linspace(0.45, 0.75, 31)
    fig, ax = plt.subplots(figsize=(8, 5))
    for L, color in [(24, "#9ecae1"), (48, "#4292c6"), (96, "#08519c")]:
        probs = [spanning_probability(L, p, trials=50, seed=0) for p in ps]
        ax.plot(ps, probs, "o-", ms=3, color=color, lw=1.8, label=f"L = {L}")
    ax.axvline(P_C_SQUARE_SITE, color="#d62728", ls="--", lw=1.5)
    ax.annotate(f"p_c ≈ {P_C_SQUARE_SITE}", (P_C_SQUARE_SITE + 0.005, 0.1), color="#d62728", fontsize=9)
    ax.set_xlabel("site occupation probability  p")
    ax.set_ylabel("spanning probability")
    ax.set_title("Percolation: a geometric phase transition\n"
                 "the spanning probability sharpens into a step at p_c as L grows")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "percolation_transition.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_percolation())
