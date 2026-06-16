"""Daemons — figure: extracted work vs the demon's measurement error."""

from __future__ import annotations

import os

import numpy as np

from research.daemons.szilard import simulate_engine, work_bound, work_per_bit


def figure_work_vs_error(outdir: str = "figures", temperature_k: float = 300.0) -> str:
    import matplotlib.pyplot as plt

    ps = np.linspace(0.0, 0.5, 26)
    measured = np.array([simulate_engine(80000, temperature_k, p, 0).mean_work_per_cycle for p in ps])
    theory = np.array([work_bound(temperature_k, p) for p in ps])
    quantum = work_per_bit(temperature_k)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ps, theory / quantum, color="#1f77b4", lw=2,
            label="Sagawa-Ueda bound: 1 - H2(p)")
    ax.scatter(ps, measured / quantum, s=18, color="#2ca02c",
               label="measured  (k_B T ln2 x empirical MI)", zorder=3)
    ax.axhline(1.0, color="#7f7f7f", ls=":", lw=1)
    ax.annotate("perfect demon = k_B T ln2", (0.01, 1.01), fontsize=9, color="#555")
    ax.set_xlabel("demon measurement error  p")
    ax.set_ylabel("work extracted  /  k_B T ln2")
    ax.set_title("Maxwell's demon: work extracted = information held\n"
                 "(no information at p = 0.5 -> no work; 2nd law safe)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "maxwell_demon_work.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_work_vs_error())
