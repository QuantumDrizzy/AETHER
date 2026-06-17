"""Schrodinger figure: QHO eigenstates + barrier tunnelling transmission."""

from __future__ import annotations

import os

import numpy as np

from research.electronic_structure.schrodinger import (
    _rect_barrier_grid,
    barrier_transmission_analytic,
    solve,
    transmission_transfer_matrix,
)


def figure_schrodinger(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # harmonic oscillator: potential + first few states stacked at their energies
    x = np.linspace(-6, 6, 1600)
    V = 0.5 * x ** 2
    E, psi = solve(x, V, n_states=5)
    ax1.plot(x, V, color="#444", lw=1.5)
    for n in range(5):
        scale = 1.2
        ax1.axhline(E[n], color="grey", ls=":", lw=0.6)
        ax1.plot(x, E[n] + scale * psi[:, n], lw=1.5)
    ax1.set_ylim(0, 6)
    ax1.set_xlabel("x"); ax1.set_ylabel("energy / wavefunction")
    ax1.set_title(r"Harmonic oscillator: $E_n=(n+\frac{1}{2})\omega$")

    # tunnelling transmission vs energy
    V0, a = 4.0, 1.0
    xg, Vg = _rect_barrier_grid(V0, a, n=4000)
    dx = xg[1] - xg[0]
    Es = np.linspace(0.2, 12, 240)
    Ta = np.array([barrier_transmission_analytic(e, V0, a) for e in Es])
    Tn = np.array([transmission_transfer_matrix(e, Vg, dx) for e in Es])
    ax2.plot(Es, Ta, color="#1f77b4", lw=2, label="analytic")
    ax2.plot(Es, Tn, "--", color="#d62728", lw=1.3, label="transfer matrix")
    ax2.axvline(V0, color="#888", ls=":", lw=1.2)
    ax2.annotate("barrier top $V_0$", (V0 + 0.1, 0.1), fontsize=8, color="#555")
    ax2.set_xlabel("energy E"); ax2.set_ylabel("transmission T(E)")
    ax2.set_title(f"Tunnelling through a barrier (V0={V0:g}, a={a:g})")
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

    fig.suptitle("Real-space Schrodinger — the complement to k-space tight-binding")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "schrodinger.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_schrodinger())
