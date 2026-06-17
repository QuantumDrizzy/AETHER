"""Brownian figure: trajectories + MSD (Einstein) and the velocity autocorrelation."""

from __future__ import annotations

import os

import numpy as np

from research.stochastic.brownian import (
    diffusion_from_msd,
    mean_squared_displacement,
    simulate_overdamped,
    simulate_underdamped,
    velocity_autocorrelation,
)


def figure_brownian(outdir: str = "figures") -> str:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: a few trajectories + MSD overlay
    dt, D = 0.01, 1.0
    x = simulate_overdamped(n_particles=2000, n_steps=2000, dt=dt, D=D, seed=0)
    t = np.arange(x.shape[0]) * dt
    for i in range(12):
        ax1.plot(t, x[:, i], lw=0.6, alpha=0.5)
    ax1.set_xlabel("time"); ax1.set_ylabel("position x")
    ax1.set_title("Overdamped Brownian trajectories")

    axm = ax1.twinx()
    msd = mean_squared_displacement(x)
    axm.plot(t, msd, color="k", lw=2)
    axm.plot(t, 2 * D * t, "r--", lw=1.5)
    axm.set_ylabel(r"$\langle x^2\rangle$  (black);  $2Dt$ (red)")
    D_meas = diffusion_from_msd(x, dt)
    ax1.text(0.03, 0.95, f"Einstein: D={D_meas:.2f} (input {D})",
             transform=ax1.transAxes, fontsize=8, va="top")

    # right: velocity autocorrelation with the exp(-gamma t/m) law
    gamma, kT, m = 2.0, 1.0, 1.0
    _, v = simulate_underdamped(gamma=gamma, kT=kT, m=m, dt=0.005, n_steps=6000, seed=1)
    lags = 400
    vacf = velocity_autocorrelation(v, max_lag=lags)
    tau = np.arange(lags) * 0.005
    ax2.plot(tau, vacf, color="#1f77b4", lw=2, label="measured VACF")
    ax2.plot(tau, (kT / m) * np.exp(-gamma / m * tau), "r--", lw=1.5,
             label=r"$(kT/m)\,e^{-\gamma t/m}$")
    ax2.axhline(0, color="grey", lw=0.6)
    ax2.set_xlabel(r"lag $\tau$"); ax2.set_ylabel(r"$\langle v(0)v(\tau)\rangle$")
    ax2.set_title("Fluctuation-dissipation: velocity autocorrelation")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Brownian motion — Einstein relation & fluctuation-dissipation")
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "brownian.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_brownian())
