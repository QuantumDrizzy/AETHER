"""Validation of the graphene tight-binding model against closed-form physics.

Every check here has an exact analytic answer (ADR-0001, scientific spine):

- bandwidth = 6t, with E = ±3t at Γ
- Dirac points: the gap closes to zero at K and K'
- electron–hole symmetry: E_+(k) = −E_−(k)
- matrix diagonalization agrees with the closed form E = ±t|f(k)|
- Fermi velocity v_F ≈ 10^6 m/s (~ c/300), matching 3·t·a_cc / (2ħ)

Run standalone:  python tests/python/test_graphene_tb.py
Or with pytest:  python -m pytest tests/python/test_graphene_tb.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.electronic_structure.graphene import (  # noqa: E402
    ANGSTROM_M,
    EV_TO_J,
    HBAR_JS,
    GrapheneTB,
)


def test_bandwidth_and_gamma_energies():
    """At Γ the two bands sit at ±3t, so the full bandwidth is 6t."""
    g = GrapheneTB()
    e_minus, e_plus = g.bands(np.array([0.0, 0.0]))
    assert np.isclose(e_plus, 3.0 * g.t, rtol=1e-10)
    assert np.isclose(e_minus, -3.0 * g.t, rtol=1e-10)
    assert np.isclose(e_plus - e_minus, g.bandwidth_eV, rtol=1e-10)


def test_dirac_points_gap_closes():
    """The band gap must vanish at both inequivalent Dirac points K and K'."""
    g = GrapheneTB()
    for kpt in (g.K_point, g.K_prime_point):
        e_minus, e_plus = g.bands(kpt)
        assert abs(e_plus - e_minus) < 1e-9, "gap does not close at a Dirac point"


def test_electron_hole_symmetry():
    """E_+(k) = −E_−(k) for arbitrary k (bipartite-lattice symmetry)."""
    g = GrapheneTB()
    rng = np.random.default_rng(42)
    k = rng.uniform(-3.0, 3.0, size=(500, 2))
    e_minus, e_plus = g.bands(k)
    assert np.allclose(e_plus, -e_minus, atol=1e-12)


def test_diagonalization_matches_closed_form():
    """Eigenvalues of H(k) must equal the closed form ±t|f(k)| — two independent
    computations agreeing is a strong correctness check."""
    g = GrapheneTB()
    rng = np.random.default_rng(7)
    for k in rng.uniform(-3.0, 3.0, size=(200, 2)):
        e_minus, e_plus = g.bands(k)
        diag = g.eigenergies(k)  # ascending
        assert np.allclose(diag, [e_minus, e_plus], atol=1e-10)


def test_fermi_velocity_is_about_1e6_m_s():
    """Numerically fit the linear slope near K and compare to the analytic v_F,
    and check it lands at the famous ~10^6 m/s (~ c/300)."""
    g = GrapheneTB()
    K = g.K_point
    direction = np.array([1.0, 0.0])  # cone is isotropic to leading order
    q = np.linspace(1e-4, 5e-3, 30)  # small offsets from K, in 1/Å
    k = K[None, :] + q[:, None] * direction[None, :]
    _, e_plus = g.bands(k)

    slope_eV_A = np.polyfit(q, e_plus, 1)[0]  # eV·Å
    v_f_numeric = slope_eV_A * EV_TO_J * ANGSTROM_M / HBAR_JS  # m/s

    # Agreement with the analytic v_F = 3 t a_cc / (2 ħ).
    assert np.isclose(v_f_numeric, g.fermi_velocity_m_s, rtol=0.02)
    # And the iconic order of magnitude.
    assert 7.0e5 < v_f_numeric < 1.1e6, f"v_F = {v_f_numeric:.3e} m/s off the known ~1e6"


def test_dos_normalization_and_symmetry():
    """DOS integrates to 2 (two bands per cell) and is electron–hole symmetric."""
    g = GrapheneTB()
    centers, dos = g.density_of_states(n_k=400, n_bins=400)
    bin_w = centers[1] - centers[0]
    assert np.isclose((dos * bin_w).sum(), 2.0, rtol=1e-9)
    assert np.mean(np.abs(dos - dos[::-1])) < 0.03 * dos.max(), "DOS not e–h symmetric"


def test_dos_dirac_point_and_van_hove():
    """Textbook graphene DOS: vanishes at the Dirac point, peaks (van Hove) at ±t."""
    g = GrapheneTB()
    centers, dos = g.density_of_states(n_k=600, n_bins=400)
    near_zero = np.abs(centers) < 0.15
    bulk = (np.abs(centers) > 0.5) & (np.abs(centers) < 2.0)
    assert dos[near_zero].mean() < 0.2 * dos[bulk].mean(), "DOS does not vanish at Dirac point"
    e_at_peak = abs(centers[int(np.argmax(dos))])
    assert abs(e_at_peak - abs(g.t)) < 0.3, f"van Hove peak at {e_at_peak:.2f} eV, expected ~{g.t}"


def _run_standalone() -> int:
    tests = [
        test_bandwidth_and_gamma_energies,
        test_dirac_points_gap_closes,
        test_electron_hole_symmetry,
        test_diagonalization_matches_closed_form,
        test_fermi_velocity_is_about_1e6_m_s,
        test_dos_normalization_and_symmetry,
        test_dos_dirac_point_and_van_hove,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
