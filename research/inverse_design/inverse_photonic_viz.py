"""Figure: inverse photonic design — achieved gap vs target (the inverse is exact)."""

from __future__ import annotations

import os

import numpy as np

from research.inverse_design.inverse_photonic import forward_gap, numeric_inverse


def figure_inverse(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    targets = np.linspace(0.10, 0.60, 11)
    achieved, designed_n2 = [], []
    for t in targets:
        (_n1, n2, _f0), gap = numeric_inverse(1.0, float(t))
        achieved.append(gap[1])
        designed_n2.append(n2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot([targets[0], targets[-1]], [targets[0], targets[-1]], "k--", lw=1, label="ideal (y=x)")
    ax1.plot(targets, achieved, "o", color="#1f77b4", ms=6, label="achieved")
    ax1.set_xlabel("target relative bandwidth  Δf/f")
    ax1.set_ylabel("achieved Δf/f")
    ax1.set_title("Inverse design hits the target gap\n(design centre 1.0, vary bandwidth)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(targets, designed_n2, "s-", color="#d62728", lw=2, ms=5)
    ax2.set_xlabel("target relative bandwidth  Δf/f")
    ax2.set_ylabel("designed index  n2  (n1=1)")
    ax2.set_title("Structure the inverse prescribes\n(wider gap needs more index contrast)")
    ax2.grid(alpha=0.3)

    fig.suptitle("AETHER — inverse design of a 1D photonic crystal (target gap → structure)")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "inverse_photonic.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_inverse())
