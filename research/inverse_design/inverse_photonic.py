"""Inverse design of a 1D photonic crystal — target band gap -> structure.

The forward model (`em_simulation/photonic_crystal.py`) maps a structure (indices
n1, n2; design frequency f0 of a quarter-wave stack) to its photonic band gap. This
inverts it: given a **target first gap** (centre frequency f_c and relative bandwidth
Δf/f_c), find the structure that produces it.

Two routes, the honest pair:

* **Analytic** — for a quarter-wave stack the first gap centres exactly at f0 and its
  relative bandwidth obeys  Δf/f0 = (4/π)·arcsin((n2−n1)/(n2+n1))  (standard result).
  Inverting: f0 = f_c, and n2 = n1·(1+r)/(1−r) with r = sin(π/4 · Δf/f_c). Closed form.
* **Numeric** — minimise the gap mismatch over (n2, f0) through the forward model
  itself (no formula assumed), from the analytic guess. This is the general inverse-
  design pattern when no closed form exists.

Validated by **round-trip**: take a known structure, read its gap, invert, and check
the recovered structure (and its actual gap) match — the same calibrate-on-known
discipline used elsewhere in the lab. The forward model already matches the analytic
bandwidth to ~1e-3, so the analytic inverse is exact up to the gap-edge sampling.
"""
from __future__ import annotations

import numpy as np

from research.em_simulation.photonic_crystal import PhotonicCrystal


def forward_gap(n1: float, n2: float, f0: float) -> tuple[float, float] | None:
    """(centre, relative bandwidth) of the first gap, or None if there is no gap."""
    g = PhotonicCrystal(n1, n2, f0).first_gap(f_max=2.0 * f0)
    if g is None:
        return None
    lo, hi = g
    centre = 0.5 * (lo + hi)
    return centre, (hi - lo) / centre


def analytic_inverse(f_center: float, rel_bandwidth: float, n1: float = 1.0):
    """Closed-form structure (n1, n2, f0) for a target quarter-wave-stack gap."""
    r = np.sin((np.pi / 4.0) * rel_bandwidth)
    n2 = n1 * (1.0 + r) / (1.0 - r)
    return (n1, n2, f_center)


def numeric_inverse(f_center: float, rel_bandwidth: float, n1: float = 1.0):
    """Optimise (n2, f0) through the forward model to hit the target gap.

    Returns ((n1, n2, f0), achieved_gap). Starts from the analytic guess."""
    from scipy.optimize import minimize

    _, n2_0, f0_0 = analytic_inverse(f_center, rel_bandwidth, n1)

    def loss(x):
        n2, f0 = x
        if n2 <= n1 or f0 <= 0:
            return 1e6
        fg = forward_gap(n1, n2, f0)
        if fg is None:
            return 1e6
        c, w = fg
        return (c - f_center) ** 2 + (w - rel_bandwidth) ** 2

    res = minimize(loss, [n2_0, f0_0], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-10})
    n2, f0 = res.x
    return (n1, float(n2), float(f0)), forward_gap(n1, float(n2), float(f0))


def _main() -> None:
    print("=== AETHER — inverse design: target photonic band gap -> structure ===\n")
    print("  round-trip check (recover a known structure from its gap):")
    for true_n2, true_f0 in [(2.0, 1.0), (3.0, 1.5), (1.6, 0.8)]:
        c, w = forward_gap(1.0, true_n2, true_f0)
        (_, an2, af0) = analytic_inverse(c, w)
        (_, nn2, nf0), achieved = numeric_inverse(c, w)
        print(f"   true (n2={true_n2}, f0={true_f0}) -> gap(c={c:.3f}, Δ/f={w:.3f})")
        print(f"     analytic inverse: n2={an2:.3f}, f0={af0:.3f}")
        print(f"     numeric  inverse: n2={nn2:.3f}, f0={nf0:.3f}  -> gap{tuple(round(x,3) for x in achieved)}")

    print("\n  design from a spec (target centre 1.0, bandwidth 0.30):")
    (_, n2, f0), got = numeric_inverse(1.0, 0.30)
    print(f"     -> n2={n2:.3f}, f0={f0:.3f}; achieved gap centre={got[0]:.3f}, Δ/f={got[1]:.3f}")


if __name__ == "__main__":
    _main()
