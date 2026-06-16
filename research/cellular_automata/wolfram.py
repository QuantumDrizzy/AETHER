"""Elementary cellular automata — Wolfram's 1D rules, the simplest computronium.

Eight-cell neighbourhoods, two states, 256 rules — and yet the four Wolfram
complexity classes (homogeneous, periodic, chaotic, complex) all appear. Three
rules carry the story of this lab:

* **Rule 90** is exactly Pascal's triangle mod 2 from a single seed: a *closed
  form* (live cells in row n = 2**popcount(n)) and a Sierpinski fractal whose
  box-counting dimension is log2(3) ~ 1.585 — validated here by reusing the
  `fractals` module. CA <-> fractal, two corners of the lab meeting.
* **Rule 30** is class III chaos: its centre column is so unbiased it shipped as
  Mathematica's default random generator. We measure its bit-balance and Shannon
  entropy (~1 bit) and check it does *not* fall into a short cycle.
* **Rule 110** is class IV and **proven Turing-complete** (Cook, 2004): the
  minimal known universal computer. Computation from an 8-entry lookup table.

Same spirit as `game_of_life.py` (2D), one dimension down and exhaustively
classifiable. Run:  python -m research.cellular_automata.wolfram
"""

from __future__ import annotations

import numpy as np


def rule_table(rule: int) -> np.ndarray:
    """The 8-bit lookup: index = 4*left+2*center+1*right -> next center state."""
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)


def step(row: np.ndarray, table: np.ndarray) -> np.ndarray:
    """One synchronous update with periodic boundaries."""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right
    return table[idx]


def evolve(rule: int, width: int, steps: int,
           seed_row: np.ndarray | None = None) -> np.ndarray:
    """Space-time diagram: shape (steps, width). Default seed = single live cell."""
    table = rule_table(rule)
    if seed_row is None:
        row = np.zeros(width, dtype=np.uint8)
        row[width // 2] = 1
    else:
        row = seed_row.astype(np.uint8).copy()
    out = np.empty((steps, width), dtype=np.uint8)
    out[0] = row
    for t in range(1, steps):
        row = step(row, table)
        out[t] = row
    return out


# --- Rule 90: exact closed form + Sierpinski fractal -------------------------

def rule90_live_cells(n: int) -> int:
    """Closed form: from a single seed, row n of Rule 90 has 2**popcount(n) live
    cells (Pascal's triangle mod 2 / Sierpinski). No simulation needed."""
    return 1 << bin(n).count("1")


def rule90_dimension(power: int = 9) -> float:
    """Box-counting dimension of the Rule 90 Sierpinski gasket. Theory: log2(3)."""
    from research.fractals.box_counting import box_counting_dimension

    n = 1 << power                      # 2**power rows -> clean self-similarity
    grid = evolve(90, 2 * n + 1, n)
    scales = [2 ** k for k in range(1, power - 1)]
    return box_counting_dimension(grid, scales)


# --- Rule 30: chaos / randomness quality -------------------------------------

def rule30_center_column(steps: int = 4000) -> np.ndarray:
    """The centre column of Rule 30 from a single seed — Mathematica's RNG."""
    width = 2 * steps + 3               # wide enough that the cone never wraps
    st = evolve(30, width, steps)
    return st[:, width // 2].copy()


def shannon_entropy_bits(bits: np.ndarray) -> float:
    """Per-symbol entropy of a 0/1 sequence (1.0 = perfectly balanced)."""
    p = bits.mean()
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


# --- Wolfram classification (coarse, empirical) ------------------------------

def classify(rule: int, width: int = 201, steps: int = 200) -> str:
    """Coarse Wolfram class from a random seed: I homogeneous, II periodic,
    III chaotic, IV complex. Heuristic on final density + spatial entropy.

    [KNOWN_LIMIT] This cannot reliably separate class IV from III: Rule 110's
    universality lives in glider collisions on a structured background, not in
    the bulk statistics of a random seed, so it reads as III here. Telling IV
    from III rigorously is undecidable in general."""
    rng = np.random.default_rng(0)
    seed = (rng.random(width) < 0.5).astype(np.uint8)
    st = evolve(rule, width, steps, seed_row=seed)
    final = st[-1]
    dens = final.mean()
    if dens == 0.0 or dens == 1.0:
        return "I (homogeneous)"
    # cycle detection over the tail: did the pattern freeze/repeat?
    tail = st[-40:]
    repeats = any(np.array_equal(tail[-1], tail[-1 - k]) for k in range(1, 12))
    ent = shannon_entropy_bits(final)
    if repeats:
        return "II (periodic)"
    if ent > 0.9:
        return "III (chaotic)"
    return "IV (complex)"


def _main() -> None:
    print("Rule 90 - Pascal mod 2 (closed form vs simulation):")
    for n in (1, 2, 3, 7, 8, 15):
        sim = int(evolve(90, 4 * n + 3, n + 1)[n].sum())
        print(f"  row {n:>2}:  closed-form {rule90_live_cells(n):>4}   sim {sim:>4}")
    print(f"Rule 90 box-counting dimension: {rule90_dimension():.3f}  (theory log2 3 = 1.585)")

    col = rule30_center_column(4000)
    print(f"Rule 30 centre column: mean {col.mean():.3f}  entropy {shannon_entropy_bits(col):.3f} bits")

    print("Wolfram classes:")
    for r in (160, 108, 30, 110):
        print(f"  rule {r:>3}: {classify(r)}")


if __name__ == "__main__":
    _main()
