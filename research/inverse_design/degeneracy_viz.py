"""Figure: inverse-design degeneracy — each target is hit by a *family* of structures.

Three panels, one per inverse problem: the free parameter runs along x, the locked design
variable along y, and every plotted point forward-verifies to the same target property.

    python -m research.inverse_design.degeneracy_viz
"""

from __future__ import annotations

import os

import numpy as np

from research.inverse_design.degeneracy import (
    auxetic_family_verified,
    photonic_family_verified,
    ssh_family_verified,
)


def figure_degeneracy(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # photonic: n1 free, n2 locked by contrast (same gap fraction 0.30)
    ph = photonic_family_verified(1.0, 0.30, np.linspace(1.0, 2.2, 13))
    n1 = [m[0] for m in ph]; n2 = [m[1] for m in ph]
    axes[0].plot(n1, n2, "o-", color="#1f77b4")
    axes[0].set_title("Photonic: gap fraction 0.30\nn1 free, contrast locked")
    axes[0].set_xlabel("n1 (free)"); axes[0].set_ylabel("n2 (locked)")

    # SSH: energy scale t1 free, t2 = t1 + gap/2 (same gap 0.8, same winding)
    ss = ssh_family_verified(0.8, True, np.linspace(0.5, 4.0, 13))
    t1 = [m[0] for m in ss]; t2 = [m[1] for m in ss]
    axes[1].plot(t1, t2, "o-", color="#2ca02c")
    axes[1].set_title("SSH: gap 0.8, topological\nenergy scale t1 free")
    axes[1].set_xlabel("t1 (free)"); axes[1].set_ylabel("t2 = t1 + gap/2")

    # auxetic: aspect h/l free, each its own re-entrant angle (same ν = −0.5)
    ax = auxetic_family_verified(-0.5, np.linspace(1.2, 4.0, 15))
    hl = [m[1] for m in ax]; th = [m[0] for m in ax]
    axes[2].plot(hl, th, "o-", color="#d62728")
    axes[2].set_title("Auxetic: ν = −0.5\naspect h/l free")
    axes[2].set_xlabel("h/l (free)"); axes[2].set_ylabel("rib angle θ (deg)")

    fig.suptitle("The inverse is many-to-one: every point on each curve is a distinct "
                 "structure with the *same* target property", fontsize=11)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "inverse_degeneracy.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_degeneracy())
