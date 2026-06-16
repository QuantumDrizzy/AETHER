"""Mechanical metamaterials — the auxetic (negative Poisson's ratio) honeycomb.

A metamaterial's defining trait: an effective property set by the *structure*, not
the base material. The cleanest mechanical example is the **re-entrant honeycomb**,
whose in-plane Poisson's ratio can be **negative** (auxetic) — stretch it and it
gets *wider* instead of thinner, something no common bulk solid does.

Gibson–Ashby cellular-solid mechanics give the effective in-plane Poisson's ratio
for a honeycomb of inclined ribs (length l, angle θ from horizontal) and vertical
ribs (length h):

    ν*_12 = cos²θ / [ (h/l + sinθ) · sinθ ]

A regular hexagon (θ = +30°, h = l) gives ν = +1; flipping the ribs inward (θ < 0,
"re-entrant") flips the sign to ν < 0 — auxetic. The base material never changes;
only the geometry does. That is the whole point of a metamaterial.

Reference: Gibson & Ashby, *Cellular Solids* (2nd ed.), ch. 4.
"""

from __future__ import annotations

import math


def poisson_ratio_12(theta_deg: float, h_over_l: float) -> float:
    """Effective in-plane Poisson's ratio ν*_12 of a honeycomb (loading dir 1)."""
    th = math.radians(theta_deg)
    s, c = math.sin(th), math.cos(th)
    denom = (h_over_l + s) * s
    if abs(denom) < 1e-12:
        raise ValueError("degenerate geometry (θ → 0): transverse coupling vanishes")
    return c * c / denom


def poisson_ratio_21(theta_deg: float, h_over_l: float) -> float:
    """The reciprocal direction; Gibson–Ashby give ν12 · ν21 = 1."""
    th = math.radians(theta_deg)
    s, c = math.sin(th), math.cos(th)
    return (h_over_l + s) * s / (c * c)


def is_auxetic(theta_deg: float, h_over_l: float) -> bool:
    """True if the structure has a negative Poisson's ratio (re-entrant ribs)."""
    return poisson_ratio_12(theta_deg, h_over_l) < 0.0


def sweep_angle(angles_deg, h_over_l: float = 2.0):
    """Effective Poisson's ratio vs rib angle (sign flips into the auxetic regime)."""
    return [poisson_ratio_12(a, h_over_l) for a in angles_deg]
