"""Inverse design of a gapped honeycomb (hBN) — target band structure -> parameters.

Inverts the electronic forward model (`electronic_structure/hbn.py`): the gapped
honeycomb has band gap 2Δ and conduction-band maximum E_top = √(Δ² + (t·f_max)²),
where f_max = max|f(k)| over the Brillouin zone (a geometric constant ≈ 3 for the
honeycomb). Given a target (gap, E_top), recover the structure parameters:

    Δ = gap / 2 ,   t = √(E_top² − Δ²) / f_max          (closed form)

The electronic sibling of the auxetic/photonic inverse designs. Validated by
round-trip against the *actual* model (measured gap over a BZ grid + measured band
top), not just the formula — the calibrate-on-known discipline.
"""
from __future__ import annotations

import numpy as np

from research.electronic_structure.hbn import GappedHoneycomb


def f_max() -> float:
    """max |structure factor| over the Brillouin zone (geometric constant ~3)."""
    g = GappedHoneycomb()._g
    b1, b2 = g.reciprocal_vectors
    u = np.linspace(0.0, 1.0, 240, endpoint=False)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    k = uu.ravel()[:, None] * b1[None, :] + vv.ravel()[:, None] * b2[None, :]
    return float(np.max(np.abs(g.structure_factor(k))))


def forward_gap_top(delta: float, t: float, a_cc: float = 1.45) -> tuple[float, float]:
    """(gap, conduction-band maximum E_top) measured on the actual BZ model."""
    h = GappedHoneycomb(a_cc=a_cc, t=t, delta=delta)
    gap = h.measured_gap_eV()
    g = h._g
    b1, b2 = g.reciprocal_vectors
    u = np.linspace(0.0, 1.0, 240, endpoint=False)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    k = uu.ravel()[:, None] * b1[None, :] + vv.ravel()[:, None] * b2[None, :]
    _e_minus, e_plus = h.bands(k)
    return gap, float(np.max(e_plus))


def analytic_inverse(target_gap: float, target_top: float, a_cc: float = 1.45):
    """Closed-form (delta, t) for a target (gap, conduction-band top)."""
    delta = target_gap / 2.0
    fm = f_max()
    inside = target_top ** 2 - delta ** 2
    if inside < 0:
        raise ValueError("target top is below the gap edge (E_top must exceed Δ)")
    t = np.sqrt(inside) / fm
    return float(delta), float(t)


def _main() -> None:
    print("=== AETHER — inverse design: target band structure -> hBN parameters ===\n")
    print(f"  geometric constant f_max = {f_max():.4f}  (~3 for the honeycomb)\n")
    print("  round-trip (recover known parameters from the band structure):")
    for d_true, t_true in [(2.3, 2.3), (1.0, 2.0), (3.0, 1.5)]:
        gap, top = forward_gap_top(d_true, t_true)
        d, t = analytic_inverse(gap, top)
        print(f"   true (Δ={d_true}, t={t_true}) -> gap={gap:.3f}, E_top={top:.3f} eV"
              f"  ->  inverse Δ={d:.3f}, t={t:.3f}")
    print("\n  design from a spec (target gap = 2.0 eV, E_top = 5.0 eV):")
    d, t = analytic_inverse(2.0, 5.0)
    g2, top2 = forward_gap_top(d, t)
    print(f"   -> Δ={d:.3f}, t={t:.3f}; achieved gap={g2:.3f}, E_top={top2:.3f} eV")


if __name__ == "__main__":
    _main()
