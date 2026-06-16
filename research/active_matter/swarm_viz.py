"""Active swarm — figure: self-repair with vs without spare agents."""

from __future__ import annotations

import os

from research.active_matter.swarm_assembly import TargetSeekingSwarm, ring_target


def _trace(n_agents, seed, damage_at=25, total=55, k=3):
    swarm = TargetSeekingSwarm(ring_target(12), n_agents=n_agents, seed=seed)
    cov = []
    for i in range(total):
        if i == damage_at:
            swarm.damage(k)
        swarm.run(steps=8)
        cov.append(swarm.coverage())
    return cov, damage_at


def figure_swarm_repair(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    with_spares, dmg = _trace(n_agents=16, seed=1)   # 4 spares
    no_spares, _ = _trace(n_agents=12, seed=0)       # exact mass

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(with_spares))
    ax.plot(xs, with_spares, color="#2ca02c", lw=2, label="with spare agents (16) -> full repair")
    ax.plot(xs, no_spares, color="#1f77b4", lw=2, label="exact mass (12) -> graceful degradation")
    ax.axvline(dmg, color="#d62728", ls="--", lw=1.5)
    ax.annotate("damage: -3 agents", (dmg + 0.5, 0.2), color="#d62728", fontsize=9)
    ax.axhline(9 / 12, color="#888", ls=":", lw=1)
    ax.annotate("ceiling 9/12", (1, 9 / 12 + 0.01), color="#888", fontsize=8)
    ax.set_xlabel("phase")
    ax.set_ylabel("target coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Target-seeking active swarm: damage and self-repair\n"
                 "redundancy heals fully; without spares it degrades gracefully")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "swarm_self_repair.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_swarm_repair())
