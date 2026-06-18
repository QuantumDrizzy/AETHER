"""Inverse-design degeneracy — return the *family* of structures, not a false-unique one.

ADR-0001's honesty clause: an inverse problem is usually many-to-one — several structures
produce the same target property — so a designer that hands back a single answer is lying by
omission. This module makes the degeneracy explicit for each inverse problem already built:
it generates the solution manifold and *forward-verifies every member hits the target*,
turning "the inverse is multi-modal" from a caveat into a checkable object.

The three degeneracies, each a genuine free parameter of the forward physics:

  * **Photonic** — the first-gap *relative* bandwidth depends only on the index contrast
    r = (n2−n1)/(n2+n1) (quarter-wave law), not the absolute indices. So a target bandwidth
    fixes the ratio and leaves n1 free: a one-parameter family of material pairs, same gap
    fraction. (Centre is set independently by the period f0.)
  * **SSH** — gap and winding depend on |t1−t2| and sign(t2−t1) only; the overall energy
    scale t1 is free. The family is the ray {(t1, t1 ± gap/2)}.
  * **Auxetic** — ν*_12(θ, h/l) has two geometric knobs; fixing ν leaves the aspect h/l free,
    each h/l giving its own rib angle θ. A one-parameter family of honeycombs, same ν.

Every generator is paired with a forward check through the *validated* forward model — a
family member is only reported if the physics agrees it hits the target.
"""

from __future__ import annotations

from research.inverse_design.inverse_photonic import analytic_inverse as _photonic_inv
from research.inverse_design.inverse_photonic import forward_gap
from research.inverse_design.inverse_auxetic import analytic_inverse as _auxetic_inv
from research.inverse_design.inverse_ssh import design as _ssh_design
from research.inverse_design.inverse_ssh import forward_check as _ssh_check
from research.metamaterials.auxetic import poisson_ratio_12


# ── photonic: target (centre, rel. bandwidth) -> family over n1 ───────────────────────
def photonic_family(f_center, rel_bandwidth, n1_values):
    """[(n1, n2, f0)] — same gap fraction, different absolute indices (contrast locked)."""
    return [_photonic_inv(f_center, rel_bandwidth, n1) for n1 in n1_values]


def photonic_family_verified(f_center, rel_bandwidth, n1_values, tol=2e-2):
    """Keep only members whose forward gap actually matches the target (centre, bandwidth)."""
    out = []
    for n1, n2, f0 in photonic_family(f_center, rel_bandwidth, n1_values):
        fg = forward_gap(n1, n2, f0)
        if fg is None:
            continue
        c, w = fg
        if abs(c - f_center) <= tol * f_center and abs(w - rel_bandwidth) <= tol:
            out.append((n1, n2, f0))
    return out


# ── SSH: target gap + phase -> family over the energy scale t1 ────────────────────────
def ssh_family(target_gap, topological, t1_values):
    """[(t1, t2)] on the degenerate ray; trivial phase needs t1 > gap/2 for t2 > 0."""
    out = []
    for t1 in t1_values:
        try:
            out.append(_ssh_design(target_gap, topological, t1))
        except ValueError:                       # t1 too small for the trivial branch
            continue
    return out


def ssh_family_verified(target_gap, topological, t1_values, tol=1e-9):
    """Members whose forward gap + winding match the target gap and requested phase."""
    out = []
    for t1, t2 in ssh_family(target_gap, topological, t1_values):
        chk = _ssh_check(t1, t2)
        if abs(chk["gap"] - target_gap) <= tol and chk["topological"] == topological:
            out.append((t1, t2))
    return out


# ── auxetic: target ν -> family over the aspect ratio h/l ─────────────────────────────
def auxetic_family(target_nu, h_over_l_values):
    """[(theta_deg, h_over_l)] — same Poisson ratio, different aspect (each its own θ)."""
    out = []
    for hl in h_over_l_values:
        try:
            out.append((_auxetic_inv(target_nu, hl), hl))
        except ValueError:                       # no real geometry at this aspect
            continue
    return out


def auxetic_family_verified(target_nu, h_over_l_values, tol=1e-3):
    """Members whose forward Poisson ratio matches the target."""
    out = []
    for theta, hl in auxetic_family(target_nu, h_over_l_values):
        try:
            if abs(poisson_ratio_12(theta, hl) - target_nu) <= tol:
                out.append((theta, hl))
        except ValueError:
            continue
    return out


def _main() -> None:
    print("=== AETHER — inverse-design degeneracy: the family, not a false-unique answer ===\n")

    print("  photonic — target gap fraction 0.30 at centre 1.0; n1 free, contrast locked:")
    for n1, n2, f0 in photonic_family_verified(1.0, 0.30, [1.0, 1.3, 1.6, 2.0]):
        c, w = forward_gap(n1, n2, f0)
        print(f"    (n1={n1:.2f}, n2={n2:.3f}) -> centre={c:.3f}, bandwidth={w:.3f}")

    print("\n  SSH — target gap 0.8, topological; energy scale t1 free:")
    for t1, t2 in ssh_family_verified(0.8, True, [0.5, 1.0, 2.0, 4.0]):
        chk = _ssh_check(t1, t2)
        print(f"    (t1={t1:.2f}, t2={t2:.2f}) -> gap={chk['gap']:.3f}, W={chk['winding']}")

    print("\n  auxetic — target ν=−0.5; aspect h/l free, each its own re-entrant angle:")
    for theta, hl in auxetic_family_verified(-0.5, [1.0, 1.5, 2.0, 3.0]):
        print(f"    (θ={theta:+.2f}°, h/l={hl:.1f}) -> ν={poisson_ratio_12(theta, hl):+.3f}")

    print("\n  Each list is a one-parameter manifold of distinct structures hitting one target.")


if __name__ == "__main__":
    _main()
