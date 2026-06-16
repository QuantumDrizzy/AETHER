"""Vicsek — figure: the flocking region in the (noise, density) plane."""

from __future__ import annotations

import os

import numpy as np

from research.active_matter.phase_diagram import phase_diagram


def figure_phase_diagram(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    noises = np.linspace(0.2, 6.0, 12)
    densities = np.linspace(0.1, 4.5, 9)
    grid = phase_diagram(noises, densities, box=7.0, steps=130)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(noises, densities, grid, cmap="viridis", vmin=0, vmax=1, shading="auto")
    ax.set_xlabel("angular noise  η")
    ax.set_ylabel("density  N / box²")
    ax.set_title("Vicsek flocking region:\n"
                 "order needs low noise AND enough density (bright = flocking)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("order parameter  φ")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "vicsek_phase_diagram.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_phase_diagram())
