"""Fractals — figure: the Sierpinski triangle and its box-counting fit."""

from __future__ import annotations

import os

import numpy as np

from research.fractals.box_counting import (
    SIERPINSKI_TRIANGLE_DF,
    box_count,
    sierpinski_triangle,
)


def figure_box_counting(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    g = sierpinski_triangle(8)
    scales = [1, 2, 4, 8, 16, 32, 64]
    counts = [box_count(g, s) for s in scales]
    inv_s = 1.0 / np.array(scales, dtype=float)
    slope = np.polyfit(np.log(inv_s), np.log(counts), 1)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.imshow(g, cmap="binary")
    ax1.set_title("Sierpinski triangle")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2.loglog(inv_s, counts, "o", color="#1f77b4", ms=7)
    fit = np.exp(np.polyval(np.polyfit(np.log(inv_s), np.log(counts), 1), np.log(inv_s)))
    ax2.loglog(inv_s, fit, "-", color="#d62728", lw=1.5)
    ax2.set_xlabel("1 / box size")
    ax2.set_ylabel("non-empty boxes  N(s)")
    ax2.set_title(f"box-counting: D_f = {slope:.3f}\n(exact log3/log2 = {SIERPINSKI_TRIANGLE_DF:.3f})")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "fractal_box_counting.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_box_counting())
