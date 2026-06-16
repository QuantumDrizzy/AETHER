"""Game of Life — figure: a glider walking across the grid (t = 0..4)."""

from __future__ import annotations

import os

from research.cellular_automata.game_of_life import glider, step


def figure_glider(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    g = glider(12)
    frames = [g.copy()]
    for _ in range(4):
        g = step(g)
        frames.append(g.copy())

    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6))
    for t, (ax, fr) in enumerate(zip(axes, frames)):
        ax.imshow(fr, cmap="binary", vmin=0, vmax=1)
        ax.set_title(f"t = {t}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Game of Life: a glider computes by moving — it returns to itself, "
                 "shifted by (1,1), every 4 steps", fontsize=11)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "game_of_life_glider.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_glider())
