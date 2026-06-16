"""Kane-Mele — figure: spin Chern numbers cancel, but Z2 stays nonzero (QSH)."""

from __future__ import annotations

import os

import numpy as np

from research.electronic_structure.kane_mele import KaneMele


def figure_z2_transition(outdir: str = "figures", lambda_so: float = 0.1) -> str:
    import matplotlib.pyplot as plt

    ms = np.linspace(0.0, 1.0, 13)
    c_up, c_dn, z2 = [], [], []
    for m in ms:
        km = KaneMele(lambda_so=lambda_so, M=float(m))
        cu, cd = km.spin_chern_numbers(28)
        c_up.append(cu)
        c_dn.append(cd)
        z2.append(km.z2_invariant(28))

    boundary = 3 * np.sqrt(3) * lambda_so
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ms, c_up, "o-", color="#1f77b4", lw=2, label="C↑ (spin up)")
    ax.plot(ms, c_dn, "s-", color="#d62728", lw=2, label="C↓ (spin down)")
    ax.plot(ms, z2, "^-", color="#2ca02c", lw=2.5, label="Z₂ invariant")
    ax.axvline(boundary, color="#888", ls="--", lw=1.5)
    ax.annotate(f"|M| = 3√3 λ\n= {boundary:.2f}", (boundary + 0.02, -0.7), fontsize=8, color="#555")
    ax.set_xlabel("sublattice mass  M")
    ax.set_ylabel("invariant")
    ax.set_yticks([-1, 0, 1])
    ax.set_title("Kane-Mele: spin Cherns cancel (total Chern 0, time-reversal safe),\n"
                 "but Z₂ = 1 below the boundary — a quantum spin Hall insulator")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "kane_mele_z2.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_z2_transition())
