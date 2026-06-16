"""Photonic crystal — figure: the band gaps (|½ Tr M| > 1) of a Bragg stack."""

from __future__ import annotations

import os

import numpy as np

from research.em_simulation.photonic_crystal import PhotonicCrystal


def figure_band_gaps(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    pc = PhotonicCrystal(n1=1.0, n2=2.0, f0=1.0)
    fs = np.linspace(1e-3, 3.0, 1500)
    bc = np.array([pc.bloch_cos(f) for f in fs])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fs, bc, color="#1f77b4", lw=1.6, label="½ Tr(M) = cos(KΛ)")
    ax.axhspan(1, bc.max() + 0.5, color="#d62728", alpha=0.07)
    ax.axhspan(bc.min() - 0.5, -1, color="#d62728", alpha=0.07)
    ax.axhline(1, color="#d62728", ls="--", lw=1)
    ax.axhline(-1, color="#d62728", ls="--", lw=1)
    for lo, hi in pc.band_gaps(3.0):
        ax.axvspan(lo, hi, color="#d62728", alpha=0.12)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("frequency  f / f₀")
    ax.set_ylabel("½ Tr(M)")
    ax.set_title("Photonic band gaps from periodic structure:\n"
                 "|½ Tr M| > 1 (shaded) → light cannot propagate (Bragg reflection)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "photonic_band_gap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_band_gaps())
