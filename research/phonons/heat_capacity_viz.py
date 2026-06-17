"""Heat capacity figure: Debye vs Einstein, Dulong-Petit plateau + low-T T^3."""

from __future__ import annotations

import os

import numpy as np

from research.phonons.heat_capacity import (
    debye_heat_capacity,
    debye_lowT_T3,
    einstein_heat_capacity,
)


def figure_heat_capacity(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    T = np.linspace(0.01, 3.0, 300)
    deb = debye_heat_capacity(T, theta_D=1.0, n_modes=3)
    ein = einstein_heat_capacity(T, theta_E=1.0, n_modes=3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(T, deb, color="#1f77b4", lw=2, label="Debye")
    ax1.plot(T, ein, color="#ff7f0e", lw=2, label="Einstein")
    ax1.axhline(3.0, color="#888", ls="--", lw=1.2)
    ax1.annotate("Dulong-Petit  3N k_B", (1.8, 3.05), fontsize=8, color="#555")
    ax1.set_xlabel("temperature  T / θ"); ax1.set_ylabel(r"$C_v$  (units $k_B$)")
    ax1.set_title("Phonon heat capacity: saturates at Dulong-Petit")
    ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    # low-T log-log: Debye follows the T^3 law, Einstein collapses faster
    Tl = np.linspace(0.02, 0.3, 120)
    ax2.loglog(Tl, debye_heat_capacity(Tl), color="#1f77b4", lw=2, label="Debye (numeric)")
    ax2.loglog(Tl, debye_lowT_T3(Tl), "k:", lw=1.5, label=r"$T^3$ law")
    ax2.loglog(Tl, einstein_heat_capacity(Tl), color="#ff7f0e", lw=2, label="Einstein")
    ax2.set_xlabel("temperature  T / θ"); ax2.set_ylabel(r"$C_v$  (units $k_B$)")
    ax2.set_title(r"Low-T: Debye $\sim T^3$, Einstein freezes exponentially")
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3, which="both")

    fig.suptitle("From phonons to thermodynamics — lattice heat capacity")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "phonon_heat_capacity.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_heat_capacity())
