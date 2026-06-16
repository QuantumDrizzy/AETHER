"""2D Ising — figure: magnetisation and susceptibility across T_c."""

from __future__ import annotations

import os

import numpy as np

from research.criticality.ising2d import T_C_ONSAGER, sweep_temperature


def figure_ising2d(outdir: str = "figures", L: int = 32) -> str:
    import matplotlib.pyplot as plt

    Ts = np.linspace(1.6, 3.2, 22)
    m, chi = sweep_temperature(Ts, L=L, equil=400, measure=500)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(Ts, m, "o-", color="#1f77b4", lw=2, label="magnetisation |m|")
    ax1.set_xlabel("temperature  T")
    ax1.set_ylabel("|m|", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(Ts, chi, "s--", color="#d62728", lw=2, label="susceptibility χ")
    ax2.set_ylabel("χ", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.axvline(T_C_ONSAGER, color="#888", ls=":", lw=1.5)
    ax1.annotate(f"Onsager T_c = {T_C_ONSAGER:.3f}", (T_C_ONSAGER + 0.03, 0.1), fontsize=9, color="#555")
    ax1.set_title(f"2D Ising (L={L}): order collapses and χ diverges at T_c")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "ising2d_transition.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_ising2d())
