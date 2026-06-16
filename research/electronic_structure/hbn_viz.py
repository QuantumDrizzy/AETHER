"""hBN — figure: gapless graphene cone vs the gap opened by sublattice asymmetry."""

from __future__ import annotations

import os

import numpy as np

from research.electronic_structure.graphene import GrapheneTB
from research.electronic_structure.hbn import GappedHoneycomb


def figure_gap_opening(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    g = GrapheneTB(a_cc=1.45, t=2.3)
    h = GappedHoneycomb(a_cc=1.45, t=2.3, delta=2.3)
    K = g.K_point
    # straight k-line through the Dirac point K (from 0.4K to 1.6K)
    ts = np.linspace(0.4, 1.6, 400)
    kline = ts[:, None] * K[None, :]
    x = ts - 1.0  # 0 at the Dirac point

    gm, gp = g.bands(kline)
    hm, hp = h.bands(kline)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, gp, color="#1f77b4", lw=2)
    ax.plot(x, gm, color="#1f77b4", lw=2, label="graphene (gapless Dirac cone)")
    ax.plot(x, hp, color="#d62728", lw=2)
    ax.plot(x, hm, color="#d62728", lw=2, label="hBN (gap = 2Δ = 4.6 eV)")
    ax.axvline(0, color="#888", ls=":", lw=1)
    ax.annotate("Dirac point K", (0.02, -5.5), fontsize=9, color="#555")
    ax.set_xlabel("k along Γ→K  (0 = Dirac point)")
    ax.set_ylabel("energy (eV)")
    ax.set_title("A band gap from broken sublattice symmetry:\n"
                 "same honeycomb, but making the two atoms different opens a gap")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "hbn_band_gap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_gap_opening())
