"""Gray-Scott — figure: a maze pattern self-organising from a single seed."""

from __future__ import annotations

import os

from research.reaction_diffusion.gray_scott import GrayScott


def figure_pattern(outdir: str = "figures", size: int = 140, steps: int = 10000) -> str:
    import matplotlib.pyplot as plt

    gs = GrayScott()                       # maze preset
    u, v = gs.seeded_init(size, seed=0)
    snapshots = {}
    done = 0
    for target in (0, 2000, steps):
        u, v = gs.run(u, v, target - done)
        snapshots[target] = v.copy()
        done = target

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, t in zip(axes, (0, 2000, steps)):
        ax.imshow(snapshots[t], cmap="magma", vmin=0, vmax=0.4)
        ax.set_title(f"t = {t}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Reaction-diffusion (Gray-Scott): a seed self-organises into a maze\n"
                 "morphogenesis from two local rules — pattern in a chemical substrate", fontsize=11)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "gray_scott_pattern.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_pattern())
