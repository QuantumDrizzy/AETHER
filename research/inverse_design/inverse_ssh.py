"""Inverse topological design — target gap + topological phase -> SSH hoppings.

Inverts the SSH model (`electronic_structure/ssh.py`). The forward facts:

    bulk gap = 2|t1 − t2| ,   winding W = 1 (topological, t2 > t1) or 0 (trivial, t1 > t2)

so you can ask for a *topological invariant*, not just a number: "give me a chain with
gap G that IS (or is NOT) topological", and recover the structure in closed form —

    |t1 − t2| = G/2 ,   t2 = t1 + G/2 (topological)  or  t1 − G/2 (trivial).

This is the inverse of a **topological property**: the design is validated by
round-trip against the bulk gap, the bulk winding number, AND the edge states of a
finite open chain (bulk-boundary correspondence) — a structure isn't accepted until it
shows the requested phase three independent ways.
"""
from __future__ import annotations

from research.electronic_structure.ssh import (
    bulk_gap,
    count_edge_states,
    winding_number,
)


def design(target_gap: float, topological: bool, t1: float = 1.0) -> tuple[float, float]:
    """Closed-form (t1, t2) for a target bulk gap and topological phase."""
    if target_gap <= 0:
        raise ValueError("target gap must be positive (a gapless chain is the transition)")
    half = target_gap / 2.0
    t2 = t1 + half if topological else t1 - half
    if t2 <= 0:
        raise ValueError(
            f"trivial phase needs t2>0: target gap {target_gap} too large for t1={t1} "
            f"(need gap < {2 * t1}); raise t1 or pick the topological phase")
    return (t1, t2)


def forward_check(t1: float, t2: float, n_cells: int = 25) -> dict:
    """What the designed chain actually has: gap, bulk winding, finite-chain edge states."""
    return {
        "gap": bulk_gap(t1, t2),
        "winding": winding_number(t1, t2),
        "edge_states": count_edge_states(n_cells, t1, t2),
        "topological": winding_number(t1, t2) == 1,
    }


def _main() -> None:
    print("=== AETHER — inverse topological design: target gap + phase -> SSH (t1,t2) ===\n")
    for gap, topo in [(0.8, True), (0.8, False), (1.2, True)]:
        t1, t2 = design(gap, topo)
        chk = forward_check(t1, t2)
        want = "topological" if topo else "trivial"
        print(f"  target gap={gap}, {want:<11} -> (t1={t1:.2f}, t2={t2:.2f}) | "
              f"achieved gap={chk['gap']:.3f}, W={chk['winding']}, "
              f"edge states={chk['edge_states']} -> {'topological' if chk['topological'] else 'trivial'}")
    print("\n  round-trip (recover a known chain from its gap+phase):")
    for t1_true, t2_true in [(1.0, 1.6), (1.0, 0.4)]:
        gap = bulk_gap(t1_true, t2_true)
        topo = winding_number(t1_true, t2_true) == 1
        t1, t2 = design(gap, topo, t1=t1_true)
        print(f"   true (t1={t1_true}, t2={t2_true}) -> gap={gap:.3f}, {'topo' if topo else 'triv'}"
              f"  ->  recovered (t1={t1:.2f}, t2={t2:.2f})")


if __name__ == "__main__":
    _main()
