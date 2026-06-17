"""Inverse design of an auxetic honeycomb — target Poisson's ratio -> geometry.

Inverts the Gibson-Ashby forward model (`metamaterials/auxetic.py`):

    ν*_12(θ, h/l) = cos²θ / [ (h/l + sinθ)·sinθ ]

Given a TARGET Poisson's ratio (incl. negative = auxetic) and a chosen aspect h/l,
find the rib angle θ. Rearranging is a **quadratic in s = sinθ**:

    (ν+1)·s² + ν(h/l)·s − 1 = 0   →   s = [−ν(h/l) ± √(ν²(h/l)² + 4(ν+1))] / (2(ν+1))

so the inverse is closed-form (pick the physical root, |s| ≤ 1, sign-consistent with
the target: auxetic targets need θ < 0, re-entrant ribs). A numeric bisection on the
forward model cross-checks it. Validated by round-trip: design for a target ν, run
the forward model, recover the ν.
"""
from __future__ import annotations

import math

from research.metamaterials.auxetic import poisson_ratio_12


def analytic_inverse(target_nu: float, h_over_l: float = 2.0) -> float:
    """Rib angle θ (deg) giving the target Poisson's ratio at aspect h/l (closed form)."""
    a, b, c = target_nu + 1.0, target_nu * h_over_l, -1.0
    if abs(a) < 1e-12:                       # ν = −1 special case: linear
        roots = [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            raise ValueError("no real geometry for this (ν, h/l)")
        roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
    # physical sine values, then pick the one whose forward ν matches the target
    cands = [math.degrees(math.asin(s)) for s in roots if abs(s) <= 1.0]
    cands = [th for th in cands if abs(th) > 1e-6]

    def _err(th: float) -> float:
        try:
            return abs(poisson_ratio_12(th, h_over_l) - target_nu)
        except ValueError:                   # degenerate geometry (denom -> 0)
            return float("inf")

    cands = [th for th in cands if math.isfinite(_err(th))]
    if not cands:
        raise ValueError("no physical rib angle for this (ν, h/l)")
    return min(cands, key=_err)


def numeric_inverse(target_nu: float, h_over_l: float = 2.0) -> float:
    """θ (deg) by bisection on the forward model; auxetic target -> θ<0 bracket."""
    lo, hi = (-89.0, -0.5) if target_nu < 0 else (0.5, 89.0)
    f = lambda th: poisson_ratio_12(th, h_over_l) - target_nu
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:                        # not bracketed in the natural range
        return analytic_inverse(target_nu, h_over_l)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _main() -> None:
    print("=== AETHER — inverse design: target Poisson's ratio -> honeycomb angle ===\n")
    print("  round-trip (recover a known geometry from its ν):")
    for th_true, hl in [(30.0, 1.0), (-25.0, 2.0), (-40.0, 2.0)]:
        nu = poisson_ratio_12(th_true, hl)
        th_a = analytic_inverse(nu, hl)
        th_n = numeric_inverse(nu, hl)
        tag = "auxetic" if nu < 0 else "normal"
        print(f"   θ={th_true:+.1f}° (h/l={hl}) -> ν={nu:+.3f} [{tag}] | "
              f"analytic θ={th_a:+.2f}°  numeric θ={th_n:+.2f}°")
    print("\n  design for a target ν = -0.50 (auxetic), h/l = 2:")
    th = analytic_inverse(-0.5, 2.0)
    print(f"   -> θ = {th:+.2f}°, achieved ν = {poisson_ratio_12(th, 2.0):+.3f}")


if __name__ == "__main__":
    _main()
