"""
Metamaterial EM Response Simulator
====================================
Drude-Lorentz modelling of split-ring resonator (SRR) and wire-array
metamaterials.  Computes effective parameters, transmission, and
negative-index bands.

Run standalone:
    python -m research.em_simulation.metamaterial
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MetamaterialSimulator:
    """Simulate electromagnetic response of metamaterial structures.

    Supports analytical Drude-Lorentz models for:
    * Split-Ring Resonator (SRR) - magnetic response
    * Wire array - electric response
    * Combined SRR + wire - potential negative refractive index

    Parameters
    ----------
    unit_cell_size : float
        Lattice constant of the metamaterial unit cell (metres).
    geometry : str
        ``'SRR'``, ``'wire'``, or ``'SRR+wire'``.
    """

    # Default model parameters (SI units)
    DEFAULT_PARAMS: dict[str, dict[str, float]] = {
        "SRR": {
            "f_m0": 10e9,        # magnetic resonance frequency (Hz)
            "gamma_m": 0.1e9,    # magnetic damping (Hz)
            "F_m": 0.56,         # filling fraction (coupling)
            "f_p": 15e9,         # plasma frequency for wire component (Hz)
            "gamma_e": 0.05e9,   # electric damping (Hz)
        },
        "wire": {
            "f_p": 12e9,
            "gamma_e": 0.1e9,
            "f_m0": 0.0,
            "gamma_m": 0.0,
            "F_m": 0.0,
        },
        "SRR+wire": {
            "f_m0": 10e9,
            "gamma_m": 0.1e9,
            "F_m": 0.56,
            "f_p": 12e9,
            "gamma_e": 0.05e9,
        },
    }

    def __init__(
        self,
        unit_cell_size: float = 3e-3,
        geometry: str = "SRR",
    ) -> None:
        self.a = unit_cell_size
        self.geometry = geometry
        self.params = dict(self.DEFAULT_PARAMS.get(geometry, self.DEFAULT_PARAMS["SRR"]))
        self.c0 = 299_792_458.0

    # ------------------------------------------------------------------
    # Drude-Lorentz model
    # ------------------------------------------------------------------

    def drude_lorentz_model(
        self,
        freq: np.ndarray,
        params: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute effective ε_eff and µ_eff using Drude-Lorentz models.

        Parameters
        ----------
        freq : np.ndarray
            Frequencies (Hz).
        params : dict, optional
            Model parameters.  Defaults to ``self.params``.

        Returns
        -------
        eps_eff : np.ndarray (complex)
            Effective relative permittivity.
        mu_eff : np.ndarray (complex)
            Effective relative permeability.
        """
        p = params or self.params
        omega = 2.0 * np.pi * freq

        # --- Electric: Drude model  ε = 1 - ω_p² / (ω² + jωγ_e) -----
        omega_p = 2.0 * np.pi * p.get("f_p", 0.0)
        gamma_e = 2.0 * np.pi * p.get("gamma_e", 0.0)
        if omega_p > 0:
            eps_eff = 1.0 - omega_p ** 2 / (omega ** 2 + 1j * omega * gamma_e + 1e-30)
        else:
            eps_eff = np.ones_like(omega, dtype=np.complex128)

        # --- Magnetic: Lorentz model  µ = 1 - Fω² / (ω² - ω₀² + jωγ_m)
        omega_m0 = 2.0 * np.pi * p.get("f_m0", 0.0)
        gamma_m = 2.0 * np.pi * p.get("gamma_m", 0.0)
        F_m = p.get("F_m", 0.0)
        if omega_m0 > 0:
            mu_eff = 1.0 - F_m * omega ** 2 / (omega ** 2 - omega_m0 ** 2 + 1j * omega * gamma_m + 1e-30)
        else:
            mu_eff = np.ones_like(omega, dtype=np.complex128)

        return eps_eff, mu_eff

    # ------------------------------------------------------------------
    # Effective parameters
    # ------------------------------------------------------------------

    def compute_effective_parameters(
        self, frequencies: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute effective ε, µ, n, and Z over a frequency range.

        Returns
        -------
        dict with keys ``eps_eff``, ``mu_eff``, ``n_eff``, ``Z_eff``,
        ``frequencies``.
        """
        eps, mu = self.drude_lorentz_model(frequencies)

        n_eff = np.sqrt(eps * mu + 0j)
        # Choose the branch with positive imaginary part (passive medium)
        n_eff = np.where(np.imag(n_eff) < 0, -n_eff, n_eff)

        Z_eff = np.sqrt(mu / (eps + 1e-30) + 0j)

        return {
            "frequencies": frequencies,
            "eps_eff": eps,
            "mu_eff": mu,
            "n_eff": n_eff,
            "Z_eff": Z_eff,
        }

    # ------------------------------------------------------------------
    # Transmission
    # ------------------------------------------------------------------

    def transmission_coefficient(
        self,
        freq: np.ndarray,
        num_layers: int = 1,
    ) -> np.ndarray:
        """Compute transmission through N identical metamaterial layers.

        Uses the transfer-matrix method for a slab of thickness
        ``num_layers * unit_cell_size`` embedded in vacuum.

        Parameters
        ----------
        freq : np.ndarray
            Frequencies (Hz).
        num_layers : int
            Number of unit-cell layers.

        Returns
        -------
        np.ndarray
            |T|² (power transmission coefficient).
        """
        eps, mu = self.drude_lorentz_model(freq)
        n = np.sqrt(eps * mu + 0j)
        n = np.where(np.imag(n) < 0, -n, n)
        Z = np.sqrt(mu / (eps + 1e-30) + 0j)

        d = num_layers * self.a
        k = 2.0 * np.pi * freq * n / self.c0
        phi = k * d

        # Fresnel at vacuum-metamaterial interfaces
        r = (Z - 1.0) / (Z + 1.0 + 1e-30)
        t = 2.0 * Z / (Z + 1.0 + 1e-30)

        # Fabry-Pérot transmission
        T_complex = t ** 2 * np.exp(1j * phi) / (1.0 - r ** 2 * np.exp(2j * phi) + 1e-30)
        return np.abs(T_complex) ** 2

    # ------------------------------------------------------------------
    # Negative-index band
    # ------------------------------------------------------------------

    def find_negative_index_band(
        self,
        freq_range: tuple[float, float] | None = None,
        num_points: int = 10000,
    ) -> tuple[float, float] | None:
        """Find frequency band where the real part of n_eff < 0.

        Returns
        -------
        tuple[float, float] or None
            (f_low, f_high) in Hz, or None if no NIM band found.
        """
        if freq_range is None:
            f_m0 = self.params.get("f_m0", 10e9)
            freq_range = (0.5 * f_m0, 2.0 * f_m0)

        freqs = np.linspace(freq_range[0], freq_range[1], num_points)
        result = self.compute_effective_parameters(freqs)
        n_real = np.real(result["n_eff"])

        # Find contiguous regions where n_real < 0
        negative = n_real < 0
        if not np.any(negative):
            return None

        # Find first contiguous block
        changes = np.diff(negative.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if negative[0]:
            starts = np.concatenate(([0], starts))
        if negative[-1]:
            ends = np.concatenate((ends, [len(freqs) - 1]))

        if len(starts) == 0 or len(ends) == 0:
            return None

        # Return the widest band
        widths = [(ends[i] - starts[i], i) for i in range(min(len(starts), len(ends)))]
        widths.sort(reverse=True)
        best_i = widths[0][1]

        return (float(freqs[starts[best_i]]), float(freqs[ends[best_i]]))


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 65)
    print("  AETHER - Metamaterial EM Response Demo")
    print("=" * 65)

    # --- SRR + wire composite (potential NIM) --------------------------
    meta = MetamaterialSimulator(unit_cell_size=3e-3, geometry="SRR+wire")
    freqs = np.linspace(1e9, 20e9, 5000)

    print(f"\n  Geometry      : {meta.geometry}")
    print(f"  Unit cell     : {meta.a * 1e3:.1f} mm")
    print(f"  f_p (electric): {meta.params['f_p'] / 1e9:.1f} GHz")
    print(f"  f_m0 (magn.)  : {meta.params['f_m0'] / 1e9:.1f} GHz")

    eff = meta.compute_effective_parameters(freqs)

    # Find negative-index band
    nim_band = meta.find_negative_index_band()
    if nim_band:
        print(f"\n  * Negative-Index Band:")
        print(f"    {nim_band[0] / 1e9:.3f} GHz - {nim_band[1] / 1e9:.3f} GHz")
        print(f"    Bandwidth: {(nim_band[1] - nim_band[0]) / 1e9:.3f} GHz")
    else:
        print("\n  No negative-index band found in the scanned range.")

    # Transmission
    T_1 = meta.transmission_coefficient(freqs, num_layers=1)
    T_5 = meta.transmission_coefficient(freqs, num_layers=5)
    T_10 = meta.transmission_coefficient(freqs, num_layers=10)

    print(f"\n  Transmission at 10 GHz:")
    idx_10 = np.argmin(np.abs(freqs - 10e9))
    print(f"    1 layer  : {T_1[idx_10]:.4f}")
    print(f"    5 layers : {T_5[idx_10]:.4f}")
    print(f"    10 layers: {T_10[idx_10]:.4f}")

    # Summary of effective params at a few frequencies
    print("\n  Effective parameters at selected frequencies:")
    print(f"  {'Freq (GHz)':>12s}  {'Re(ε)':>10s}  {'Re(µ)':>10s}  {'Re(n)':>10s}  {'|T|² (5L)':>10s}")
    for f_ghz in [5.0, 8.0, 10.0, 12.0, 15.0]:
        idx = np.argmin(np.abs(freqs - f_ghz * 1e9))
        eps_r = np.real(eff["eps_eff"][idx])
        mu_r = np.real(eff["mu_eff"][idx])
        n_r = np.real(eff["n_eff"][idx])
        t5 = T_5[idx]
        print(f"  {f_ghz:12.1f}  {eps_r:10.3f}  {mu_r:10.3f}  {n_r:10.3f}  {t5:10.4f}")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        f_ghz = freqs / 1e9

        axes[0, 0].plot(f_ghz, np.real(eff["eps_eff"]), "b-", label="Re(ε)")
        axes[0, 0].plot(f_ghz, np.imag(eff["eps_eff"]), "b--", alpha=0.5, label="Im(ε)")
        axes[0, 0].axhline(0, color="k", lw=0.5)
        axes[0, 0].set_title("Effective Permittivity")
        axes[0, 0].set_xlabel("Frequency (GHz)")
        axes[0, 0].legend()

        axes[0, 1].plot(f_ghz, np.real(eff["mu_eff"]), "r-", label="Re(µ)")
        axes[0, 1].plot(f_ghz, np.imag(eff["mu_eff"]), "r--", alpha=0.5, label="Im(µ)")
        axes[0, 1].axhline(0, color="k", lw=0.5)
        axes[0, 1].set_title("Effective Permeability")
        axes[0, 1].set_xlabel("Frequency (GHz)")
        axes[0, 1].legend()

        axes[1, 0].plot(f_ghz, np.real(eff["n_eff"]), "g-", label="Re(n)")
        axes[1, 0].axhline(0, color="k", lw=0.5)
        if nim_band:
            axes[1, 0].axvspan(nim_band[0] / 1e9, nim_band[1] / 1e9,
                               alpha=0.2, color="red", label="NIM band")
        axes[1, 0].set_title("Effective Refractive Index")
        axes[1, 0].set_xlabel("Frequency (GHz)")
        axes[1, 0].legend()

        axes[1, 1].plot(f_ghz, T_1, label="1 layer")
        axes[1, 1].plot(f_ghz, T_5, label="5 layers")
        axes[1, 1].plot(f_ghz, T_10, label="10 layers")
        axes[1, 1].set_title("Transmission |T|²")
        axes[1, 1].set_xlabel("Frequency (GHz)")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig("metamaterial_demo_output.png", dpi=150)
        print("\n  Plot saved to metamaterial_demo_output.png")
    except ImportError:
        print("\n  (matplotlib not available - skipping plot)")

    print()
