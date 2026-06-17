"""Brownian motion and the Langevin equation — the thermal substrate of the lab.

The kT that appears everywhere else (Landauer's bound, the Maxwell demon, the
edge of chaos) comes from thermal noise. This module simulates that noise
directly and recovers the three results that tie noise, dissipation and
temperature together — each checked against its closed form:

* **Einstein relation** — a free overdamped particle diffuses with
  <x^2(t)> = 2 D t, and the diffusion constant is D = kT / gamma. Measuring the
  mean-squared-displacement slope recovers D.
* **Equipartition** — an underdamped particle thermalises to <½ m v^2> = ½ kT
  per degree of freedom, independent of gamma.
* **Fluctuation-dissipation** — the velocity autocorrelation decays as
  (kT/m) e^{-(gamma/m) t}, and its time integral equals the same D. The
  fluctuations (noise) and the dissipation (friction) are two faces of one
  temperature.

Run:  python -m research.stochastic.brownian
"""

from __future__ import annotations

import numpy as np


# --- Overdamped free diffusion: the Einstein relation ------------------------

def simulate_overdamped(n_particles: int = 4000, n_steps: int = 2000,
                        dt: float = 0.01, D: float = 1.0, seed: int = 0) -> np.ndarray:
    """Free overdamped Brownian motion (V=0): x_{t+1} = x_t + sqrt(2 D dt) eta.
    Returns positions of shape (n_steps + 1, n_particles)."""
    rng = np.random.default_rng(seed)
    steps = np.sqrt(2.0 * D * dt) * rng.standard_normal((n_steps, n_particles))
    x = np.zeros((n_steps + 1, n_particles))
    x[1:] = np.cumsum(steps, axis=0)
    return x


def mean_squared_displacement(x: np.ndarray) -> np.ndarray:
    """<x^2(t)> averaged over particles (x starts at 0)."""
    return np.mean(x ** 2, axis=1)


def diffusion_from_msd(x: np.ndarray, dt: float) -> float:
    """Fit D from the MSD slope: <x^2> = 2 D t in 1D."""
    msd = mean_squared_displacement(x)
    t = np.arange(len(msd)) * dt
    slope = np.polyfit(t, msd, 1)[0]
    return slope / 2.0


def einstein_D(kT: float, gamma: float) -> float:
    """The Einstein relation: D = kT / gamma."""
    return kT / gamma


# --- Underdamped Langevin: equipartition + fluctuation-dissipation -----------

def simulate_underdamped(n_particles: int = 4000, n_steps: int = 4000,
                         dt: float = 0.005, gamma: float = 1.0, kT: float = 1.0,
                         m: float = 1.0, seed: int = 1):
    """Underdamped Langevin: m dv = -gamma v dt + sqrt(2 gamma kT) dW.
    Returns (positions, velocities), each (n_steps + 1, n_particles)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_steps + 1, n_particles))
    v = np.zeros((n_steps + 1, n_particles))
    # start velocities from the Maxwell-Boltzmann equilibrium
    v[0] = np.sqrt(kT / m) * rng.standard_normal(n_particles)
    noise = np.sqrt(2.0 * gamma * kT * dt) / m * rng.standard_normal((n_steps, n_particles))
    a = gamma / m
    for t in range(n_steps):
        v[t + 1] = v[t] - a * v[t] * dt + noise[t]
        x[t + 1] = x[t] + v[t] * dt
    return x, v


def kinetic_temperature(v: np.ndarray, m: float = 1.0, burn: int = 500) -> float:
    """Measured kT from equipartition: <m v^2> over the thermalised tail."""
    return float(m * np.mean(v[burn:] ** 2))


def velocity_autocorrelation(v: np.ndarray, max_lag: int, burn: int = 500) -> np.ndarray:
    """VACF C(tau) = <v(t) v(t+tau)>, averaged over particles and start times."""
    vt = v[burn:]
    n = vt.shape[0]
    c = np.empty(max_lag)
    for lag in range(max_lag):
        c[lag] = np.mean(vt[:n - lag] * vt[lag:])
    return c


def _main() -> None:
    # Einstein relation
    dt = 0.01
    D_true = 1.5
    x = simulate_overdamped(D=D_true, dt=dt)
    D_meas = diffusion_from_msd(x, dt)
    print("Einstein relation (overdamped free diffusion):")
    print(f"  D input {D_true:.3f}  ->  D from MSD slope {D_meas:.3f}")

    # equipartition + fluctuation-dissipation
    gamma, kT, m, dts = 2.0, 1.0, 1.0, 0.005
    xx, v = simulate_underdamped(gamma=gamma, kT=kT, m=m, dt=dts)
    print("Underdamped Langevin:")
    print(f"  equipartition  <m v^2> = {kinetic_temperature(v, m):.3f}  (kT = {kT:.3f})")
    vacf = velocity_autocorrelation(v, max_lag=400)
    print(f"  VACF(0) = {vacf[0]:.3f}  (kT/m = {kT/m:.3f})")
    D_fdt = np.trapezoid(vacf, dx=dts)
    print(f"  D from VACF integral = {D_fdt:.3f}  (kT/gamma = {einstein_D(kT, gamma):.3f})")


if __name__ == "__main__":
    _main()
