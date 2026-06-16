"""Universal criticality — figure: three substrates, one transition shape.

Each order parameter is plotted against its control knob normalised by that
system's own transition point (order = 1/2). If the curves overlap, criticality is
substrate-independent.
"""

from __future__ import annotations

import os

import numpy as np

from research.criticality.universal import (
    ising_order,
    reconfig_order,
    transition_point,
    vicsek_order,
)


def _normalised(controls, orders):
    controls = np.asarray(controls, dtype=float)
    orders = np.asarray(orders, dtype=float)
    xc = transition_point(controls, orders)
    return controls / xc, orders


def figure_universal_criticality(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    T = np.linspace(0.2, 2.0, 14)
    ising = np.array([ising_order(t) for t in T])

    eta = np.linspace(0.2, 6.0, 14)
    vics = np.array([vicsek_order(e) for e in eta])

    Tr = np.linspace(0.1, 6.0, 14)
    recon = np.array([reconfig_order(t) for t in Tr])

    fig, ax = plt.subplots(figsize=(8, 5))
    for (ctrl, order), color, label in [
        (_normalised(T, ising), "#d62728", "Ising magnetisation (computronium)"),
        (_normalised(eta, vics), "#2ca02c", "Vicsek flocking (active matter)"),
        (_normalised(Tr, recon), "#1f77b4", "reconfiguration coverage (programmable)"),
    ]:
        ax.plot(ctrl, order, "o-", color=color, lw=2, ms=4, label=label)

    ax.axvline(1.0, color="#888", ls="--", lw=1.5)
    ax.annotate("critical point", (1.03, 0.92), fontsize=9, color="#555")
    ax.set_xlim(0, 2.6)        # focus on the transition; beyond it is the disordered floor
    ax.set_xlabel("control knob / its own transition point")
    ax.set_ylabel("order parameter (normalised)")
    ax.set_title("Criticality is substrate-independent:\n"
                 "three unrelated systems, one order->disorder transition")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "universal_criticality.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_universal_criticality())
