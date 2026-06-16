"""Haldane model — figure: the topological phase diagram (Chern in the φ–M plane)."""

from __future__ import annotations

import os

import numpy as np

from research.electronic_structure.haldane import Haldane


def figure_phase_diagram(outdir: str = "figures", t2: float = 0.1,
                         n_phi: int = 21, n_m: int = 21, n_k: int = 16) -> str:
    import matplotlib.pyplot as plt

    phis = np.linspace(-np.pi, np.pi, n_phi)
    ms = np.linspace(-6.0, 6.0, n_m)            # in units of t2
    chern = np.zeros((n_m, n_phi), dtype=int)
    for j, ph in enumerate(phis):
        for i, mm in enumerate(ms):
            chern[i, j] = Haldane(t2=t2, phi=ph, M=mm * t2).chern_number(n_k=n_k)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(phis, ms, chern, cmap="coolwarm", vmin=-1, vmax=1, shading="auto")
    # analytic phase boundaries M = ± 3√3 t2 sinφ  (in units of t2)
    pf = np.linspace(-np.pi, np.pi, 200)
    ax.plot(pf, 3 * np.sqrt(3) * np.sin(pf), "k--", lw=1)
    ax.plot(pf, -3 * np.sqrt(3) * np.sin(pf), "k--", lw=1)
    ax.set_xlabel("complex NNN phase  φ")
    ax.set_ylabel("sublattice mass  M / t₂")
    ax.set_title("Haldane model phase diagram:\n"
                 "complex hopping opens topological lobes (Chern = ±1)")
    cb = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cb.set_label("Chern number")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "haldane_phase_diagram.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_phase_diagram())
