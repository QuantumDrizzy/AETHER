"""Anderson — figure: mean participation ratio collapsing with disorder."""

from __future__ import annotations

import os

import numpy as np

from research.electronic_structure.anderson import participation_ratios, sweep_disorder


def figure_localization(outdir: str = "figures", L: int = 300) -> str:
    import matplotlib.pyplot as plt

    Ws = np.linspace(0.3, 9.0, 18)
    pr = sweep_disorder(Ws, L=L, trials=6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(Ws, pr, "o-", color="#1f77b4", lw=2)
    ax1.axhline(2 * L / 3, color="#888", ls=":", lw=1)
    ax1.annotate("extended ~ 2L/3", (0.35, 2 * L / 3 * 1.05), fontsize=8, color="#555")
    ax1.set_xlabel("disorder strength  W")
    ax1.set_ylabel("mean participation ratio")
    ax1.set_title("States localise as disorder grows")
    ax1.grid(alpha=0.3, which="both")

    # a strongly-localised eigenstate profile
    rng_seed = 1
    import numpy as _np
    _ = _np
    # show |psi|^2 of one mid-spectrum state at strong disorder
    from research.electronic_structure.anderson import _hamiltonian
    H = _hamiltonian(L, 6.0, 1.0, np.random.default_rng(rng_seed))
    _, vecs = np.linalg.eigh(H)
    psi = vecs[:, L // 2] ** 2
    ax2.plot(psi, color="#d62728", lw=1.2)
    ax2.set_xlabel("site")
    ax2.set_ylabel("|ψ|²")
    ax2.set_title("a localised eigenstate (W = 6): lives on a few sites")
    ax2.grid(alpha=0.3)

    fig.suptitle("Anderson localization in 1D: any disorder kills the metal", fontsize=12)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "anderson_localization.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_localization())
