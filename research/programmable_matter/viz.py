"""Programmable matter — figure: the self-repair timeline under damage."""

from __future__ import annotations

import os

from research.programmable_matter.reconfigure import ReconfigurableMatter, square_target


def _cool(t_init, t_final, sweeps):
    import numpy as np
    return t_init * (t_final / t_init) ** (np.arange(sweeps) / max(sweeps - 1, 1))


def figure_self_repair(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    tgt = square_target(12, side=4)            # 16 units
    m = ReconfigurableMatter(12, tgt, seed=0)

    # one continuous schedule: cool to form the shape, damage, then cool again to repair
    phase1 = _cool(3.0, 0.01, 500)
    phase2 = _cool(1.0, 0.01, 500)             # gentle reheat, not a full melt
    damage_at = len(phase1)

    coverage, on_target = [], []
    for i, temp in enumerate(np.concatenate([phase1, phase2])):
        if i == damage_at:
            m.damage(5)                        # destroy 5 of 16 units
        m.sweep(temp)
        coverage.append(m.coverage())
        on_target.append(m.on_target_fraction())

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(coverage)))
    ax.plot(xs, coverage, color="#1f77b4", lw=2, label="target coverage")
    ax.plot(xs, on_target, color="#2ca02c", lw=2, label="units on target (efficiency)")
    ax.axvline(damage_at, color="#d62728", ls="--", lw=1.5)
    ax.annotate("damage: -5 units", (damage_at + 4, 0.15), color="#d62728", fontsize=9)
    ax.set_xlabel("sweep")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Programmable matter: reconfigure, get damaged, repair\n"
                 "coverage drops with lost mass, but survivors all return to the target")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "programmable_matter_repair.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_self_repair())
