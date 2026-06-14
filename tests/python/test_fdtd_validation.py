"""Validation of the 1-D FDTD electromagnetic solver against physics references.

These are the first real validation tests in AETHER (ADR-0001, P1). They check
the existing `FDTDSimulator1D` against quantities with a known answer, so its
output can actually be trusted:

1. Vacuum propagation speed ≈ c   — validates the Yee update + Courant timestep.
2. Numerical stability             — no NaN/Inf blow-up over a long run.
3. PML absorption                  — energy decays after the pulse, i.e. the
                                      absorbing boundary works (no spurious
                                      reflection trapping energy).

Run standalone:  python tests/python/test_fdtd_validation.py
Or with pytest:  python -m pytest tests/python/test_fdtd_validation.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the repo root importable whether run via pytest or standalone.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.em_simulation.resonance import FDTDSimulator1D

C0 = 299_792_458.0  # speed of light, m/s


def _well_resolved_sim() -> FDTDSimulator1D:
    """A vacuum domain where the source wavelength is well resolved.

    length 0.02 m, 400 cells -> dx = 5e-5 m. At f = 3e11 Hz the vacuum
    wavelength is c/f = 1e-3 m = 20 cells/wavelength (good FDTD resolution),
    and the domain spans ~20 wavelengths so a packet has room to propagate.
    """
    return FDTDSimulator1D(length_m=0.02, resolution=400, dt_factor=0.99)


def test_vacuum_propagation_speed_is_c():
    """The wave packet's peak must travel at ~c in vacuum (within numerical
    dispersion + discrete-grid measurement error)."""
    sim = _well_resolved_sim()
    sim.add_source(position=0.25, frequency=3e11, source_type="gaussian_pulse")
    result = sim.run(num_steps=900)

    E = result.E_field_history
    dt, dx = sim.dt, sim.dx
    src = sim._source_idx
    lo, hi = src + 15, sim.resolution - sim.pml_layers - 5  # right-going region, pre-PML

    # Energy of the right-going packet inside the window, per step.
    win = E[:, lo:hi] ** 2
    win_energy = win.sum(axis=1)
    strong = win_energy > 0.2 * win_energy.max()  # only steps where the packet is in transit

    # Track its energy centroid (group velocity ~ c, smoother than argmax).
    idx = np.arange(win.shape[1])
    steps, positions = [], []
    for step in np.where(strong)[0]:
        w = win[step]
        centroid = (idx * w).sum() / w.sum()
        steps.append(step * dt)
        positions.append((lo + centroid) * dx)

    assert len(steps) > 50, f"not enough clean samples to fit a velocity ({len(steps)})"
    velocity = np.polyfit(np.array(steps), np.array(positions), 1)[0]
    ratio = velocity / C0
    # [KNOWN_LIMIT] discrete-grid argmax + numerical dispersion -> a few % error.
    assert 0.95 < ratio < 1.05, f"vacuum speed {ratio:.3f}c is not ~c"


def test_simulation_is_stable():
    """No NaN/Inf and bounded fields over a long run (Courant stability)."""
    sim = _well_resolved_sim()
    sim.add_source(position=0.5, frequency=3e11, source_type="ricker")
    result = sim.run(num_steps=2000)
    E = result.E_field_history
    assert np.all(np.isfinite(E)), "fields contain NaN/Inf -> unstable"
    assert np.abs(E).max() < 1e3, "fields blew up -> unstable"


def test_pml_absorbs_energy():
    """After the source stops, the PML must absorb the energy: final domain
    energy should be a small fraction of the peak energy."""
    sim = _well_resolved_sim()
    sim.add_source(position=0.5, frequency=3e11, source_type="gaussian_pulse")
    result = sim.run(num_steps=4000)

    eps = sim.eps0 * sim.epsilon_r
    mu = sim.mu0 * sim.mu_r
    E = result.E_field_history
    H = result.H_field_history
    # EM energy density integrated over the domain, per time step.
    energy = 0.5 * (eps * E**2 + mu * H**2).sum(axis=1) * sim.dx

    peak = energy.max()
    final = energy[-1]
    assert peak > 0.0, "no energy was injected"
    assert final < 0.05 * peak, f"PML left {final / peak:.1%} of peak energy (poor absorption)"


def _run_standalone() -> int:
    tests = [
        test_vacuum_propagation_speed_is_c,
        test_simulation_is_stable,
        test_pml_absorbs_energy,
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
    import sys

    sys.exit(1 if _run_standalone() else 0)
