"""Wolfram CA figure: the four complexity classes from a single seed."""

from __future__ import annotations

import os

from research.cellular_automata.wolfram import evolve


def figure_classes(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    width, steps = 401, 200
    panels = [
        (160, "Class I — homogeneous"),
        (108, "Class II — periodic"),
        (90, "Rule 90 — Sierpinski (D=1.585)"),
        (30, "Class III — chaos (RNG)"),
        (110, "Class IV — universal (Cook 2004)"),
        (184, "Rule 184 — traffic flow"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (rule, title) in zip(axes.ravel(), panels):
        st = evolve(rule, width, steps)
        ax.imshow(st, cmap="binary", interpolation="nearest", aspect="auto")
        ax.set_title(f"rule {rule}\n{title}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Elementary cellular automata — 8-bit rules, four complexity classes",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "wolfram_classes.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_classes())
