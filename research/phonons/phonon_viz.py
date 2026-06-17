"""Phonon dispersion figure: monatomic acoustic branch + diatomic gap."""

from __future__ import annotations

import os

import numpy as np

from research.phonons.phonon_chain import (
    diatomic_dispersion,
    monatomic_dispersion,
    zone_boundary_gap,
)


def figure_dispersion(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    a = 1.0
    ks = np.linspace(-np.pi / a, np.pi / a, 401)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    mono = monatomic_dispersion(ks, 1.0, 1.0, a)
    ax1.plot(ks, mono, color="#1f77b4", lw=2)
    ax1.set_title("Monatomic chain — single acoustic branch")
    ax1.set_xlabel("wavevector  k"); ax1.set_ylabel(r"$\omega(k)$")
    ax1.axvline(np.pi / a, color="grey", ls=":", lw=1)
    ax1.axvline(-np.pi / a, color="grey", ls=":", lw=1)
    ax1.grid(alpha=0.3)

    m1, m2 = 1.0, 2.0
    ac, op = diatomic_dispersion(ks, 1.0, m1, m2, a)
    ax2.plot(ks, ac, color="#2ca02c", lw=2, label="acoustic")
    ax2.plot(ks, op, color="#d62728", lw=2, label="optical")
    ac_top, op_bot = zone_boundary_gap(1.0, m1, m2)
    ax2.axhspan(ac_top, op_bot, color="orange", alpha=0.2)
    ax2.annotate("phonon\nband gap", (0, (ac_top + op_bot) / 2),
                 ha="center", va="center", fontsize=9)
    ax2.set_title(f"Diatomic chain (m1={m1:g}, m2={m2:g}) — acoustic + optical, gap")
    ax2.set_xlabel("wavevector  k"); ax2.set_ylabel(r"$\omega(k)$")
    ax2.legend(loc="lower center")
    ax2.grid(alpha=0.3)

    fig.suptitle("Phonon dispersion of 1D chains — band structure for lattice vibrations")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "phonon_dispersion.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_dispersion())
