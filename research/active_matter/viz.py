"""Active matter — figure: the flocking order parameter vs noise."""

from __future__ import annotations

import os

import numpy as np

from research.active_matter.vicsek import sweep_noise


def figure_flocking_transition(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    etas = np.linspace(0.1, 6.0, 16)
    phis = sweep_noise(etas, steps=180, n=300, seed=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(etas, phis, "o-", color="#2ca02c", lw=2, ms=5)
    ax.set_xlabel("angular noise  eta")
    ax.set_ylabel("order parameter  phi")
    ax.set_ylim(0, 1.02)
    ax.set_title("Active matter (Vicsek): a swarm flocks from local rules,\n"
                 "then disorders as noise rises -- a non-equilibrium transition")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "vicsek_flocking.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_flocking_transition())
