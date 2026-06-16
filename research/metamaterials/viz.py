"""Mechanical metamaterials — figure: Poisson's ratio vs rib angle."""

from __future__ import annotations

import os

import numpy as np

from research.metamaterials.auxetic import poisson_ratio_12


def figure_auxetic(outdir: str = "figures", h_over_l: float = 2.0) -> str:
    import matplotlib.pyplot as plt

    neg = np.linspace(-42, -4, 80)
    pos = np.linspace(4, 42, 80)
    nu_neg = [poisson_ratio_12(a, h_over_l) for a in neg]
    nu_pos = [poisson_ratio_12(a, h_over_l) for a in pos]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(neg, nu_neg, color="#d62728", lw=2.5, label="re-entrant (auxetic, ν < 0)")
    ax.plot(pos, nu_pos, color="#1f77b4", lw=2.5, label="normal honeycomb (ν > 0)")
    ax.axhline(0, color="#444", lw=1)
    ax.axhspan(ax.get_ylim()[0], 0, color="#d62728", alpha=0.06)
    ax.set_xlabel("rib angle  θ  (degrees)")
    ax.set_ylabel("effective Poisson's ratio  ν*₁₂")
    ax.set_title("A metamaterial property from structure alone:\n"
                 "flip the ribs inward (θ < 0) and the honeycomb becomes auxetic")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "auxetic_poisson.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_auxetic())
