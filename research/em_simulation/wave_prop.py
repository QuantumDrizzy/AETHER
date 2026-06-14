"""
Wave Propagation Through Layered Materials
=============================================
Transfer-matrix method for multi-layer dielectric/magnetic structures.
Computes transmission, reflection, and group velocity spectra.

Run standalone:
    python -m research.em_simulation.wave_prop
"""

from __future__ import annotations

from typing import Any

import numpy as np


class WavePropagator:
    """Plane-wave propagation through a stack of planar layers.

    Each layer is described by a dict with keys:
      - ``eps_r`` (float): relative permittivity
      - ``mu_r``  (float): relative permeability (default 1.0)
      - ``sigma`` (float): conductivity S/m (default 0.0)
      - ``thickness`` (float): layer thickness in metres

    Parameters
    ----------
    medium_properties : list[dict]
        Ordered list of layers (left -> right).  The surrounding medium
        is assumed to be vacuum.
    """

    def __init__(self, medium_properties: list[dict[str, float]]) -> None:
        self.layers = medium_properties
        self.c0 = 299_792_458.0
        self.eps0 = 8.854187817e-12
        self.mu0 = 4.0 * np.pi * 1e-7

    # ------------------------------------------------------------------
    # Transfer matrix at a single frequency
    # ------------------------------------------------------------------

    def transfer_matrix(self, freq: float) -> np.ndarray:
        """Compute the 2×2 transfer matrix for the entire stack at *freq*.

        Uses the standard transfer-matrix (Abeles) formalism for normal
        incidence.

        Parameters
        ----------
        freq : float
            Frequency (Hz).

        Returns
        -------
        np.ndarray, shape (2, 2), complex
            Total transfer matrix M = M_N · … · M_2 · M_1.
        """
        omega = 2.0 * np.pi * freq
        M_total = np.eye(2, dtype=np.complex128)

        for layer in self.layers:
            eps_r = layer.get("eps_r", 1.0)
            mu_r = layer.get("mu_r", 1.0)
            sigma = layer.get("sigma", 0.0)
            d = layer.get("thickness", 1e-3)

            # Complex permittivity with conductivity
            eps_c = eps_r - 1j * sigma / (omega * self.eps0 + 1e-30)
            n = np.sqrt(eps_c * mu_r + 0j)
            eta = np.sqrt(self.mu0 * mu_r / (self.eps0 * eps_c + 1e-30) + 0j)

            k = omega * n / self.c0
            phi = k * d

            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            # Layer transfer matrix (field-matching formalism)
            M_layer = np.array([
                [cos_phi,         1j * eta * sin_phi],
                [1j * sin_phi / (eta + 1e-30), cos_phi],
            ], dtype=np.complex128)

            M_total = M_total @ M_layer

        return M_total

    # ------------------------------------------------------------------
    # Spectra
    # ------------------------------------------------------------------

    def transmission_spectrum(self, freq_range: np.ndarray) -> np.ndarray:
        """Compute power transmission coefficient |T|² over a frequency range.

        Parameters
        ----------
        freq_range : np.ndarray
            Array of frequencies (Hz).

        Returns
        -------
        np.ndarray
            |T|² at each frequency.
        """
        eta0 = np.sqrt(self.mu0 / self.eps0)  # vacuum impedance ≈ 377 Ω
        T_power = np.zeros(len(freq_range), dtype=np.float64)

        for i, f in enumerate(freq_range):
            M = self.transfer_matrix(f)
            # T = 2 / (M11 + M12/η₀ + M21·η₀ + M22)
            denom = M[0, 0] + M[0, 1] / eta0 + M[1, 0] * eta0 + M[1, 1]
            T_complex = 2.0 / (denom + 1e-30)
            T_power[i] = float(np.abs(T_complex) ** 2)

        return T_power

    def reflection_spectrum(self, freq_range: np.ndarray) -> np.ndarray:
        """Compute power reflection coefficient |R|² over a frequency range.

        Parameters
        ----------
        freq_range : np.ndarray
            Array of frequencies (Hz).

        Returns
        -------
        np.ndarray
            |R|² at each frequency.
        """
        eta0 = np.sqrt(self.mu0 / self.eps0)
        R_power = np.zeros(len(freq_range), dtype=np.float64)

        for i, f in enumerate(freq_range):
            M = self.transfer_matrix(f)
            denom = M[0, 0] + M[0, 1] / eta0 + M[1, 0] * eta0 + M[1, 1]
            numer = M[0, 0] + M[0, 1] / eta0 - M[1, 0] * eta0 - M[1, 1]
            R_complex = numer / (denom + 1e-30)
            R_power[i] = float(np.abs(R_complex) ** 2)

        return R_power

    def group_velocity(
        self, freq_range: np.ndarray, df: float | None = None
    ) -> np.ndarray:
        """Compute group velocity v_g = dω/dk through the stack.

        Uses numerical differentiation of the phase of the transmission
        coefficient.

        Parameters
        ----------
        freq_range : np.ndarray
            Frequencies (Hz).
        df : float, optional
            Frequency step for numerical derivative.  Default: inferred
            from freq_range spacing.

        Returns
        -------
        np.ndarray
            Group velocity (m/s) at each frequency (interior points;
            boundary values are extrapolated).
        """
        eta0 = np.sqrt(self.mu0 / self.eps0)
        phases = np.zeros(len(freq_range), dtype=np.float64)

        for i, f in enumerate(freq_range):
            M = self.transfer_matrix(f)
            denom = M[0, 0] + M[0, 1] / eta0 + M[1, 0] * eta0 + M[1, 1]
            T_complex = 2.0 / (denom + 1e-30)
            phases[i] = np.unwrap([np.angle(T_complex)])[0]

        # Unwrap across the array
        phases = np.unwrap(phases)

        # Total thickness
        total_d = sum(l.get("thickness", 1e-3) for l in self.layers)

        # dφ/dω -> group delay τ_g, then v_g = d / τ_g
        omega = 2.0 * np.pi * freq_range
        dphi_domega = np.gradient(phases, omega)

        # Group delay (negative of phase derivative)
        tau_g = -dphi_domega
        # Avoid division by zero
        tau_g[np.abs(tau_g) < 1e-30] = 1e-30

        v_g = total_d / tau_g
        return v_g


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 65)
    print("  AETHER - Wave Propagation Demo")
    print("  Transfer-matrix method for layered structures")
    print("=" * 65)

    # Define a Bragg mirror: alternating high/low index layers
    n_pairs = 5
    lambda_design = 600e-9  # design wavelength 600 nm
    n_H, n_L = 2.35, 1.38   # TiO2, MgF2

    layers: list[dict[str, float]] = []
    for _ in range(n_pairs):
        layers.append({
            "eps_r": n_H ** 2,
            "thickness": lambda_design / (4.0 * n_H),
        })
        layers.append({
            "eps_r": n_L ** 2,
            "thickness": lambda_design / (4.0 * n_L),
        })

    prop = WavePropagator(layers)
    total_thickness = sum(l["thickness"] for l in layers)

    print(f"\n  Structure : {n_pairs}-pair Bragg mirror (TiO₂/MgF₂)")
    print(f"  Design λ  : {lambda_design * 1e9:.0f} nm")
    print(f"  Total d   : {total_thickness * 1e6:.2f} µm")
    print(f"  Layers    : {len(layers)}")

    # Frequency range: 300–900 nm equivalent
    c0 = 299_792_458.0
    f_min = c0 / 900e-9
    f_max = c0 / 300e-9
    freqs = np.linspace(f_min, f_max, 2000)
    wavelengths = c0 / freqs * 1e9  # nm

    T = prop.transmission_spectrum(freqs)
    R = prop.reflection_spectrum(freqs)
    v_g = prop.group_velocity(freqs)

    # Find stop-band
    stop_band = wavelengths[T < 0.01]
    if len(stop_band) > 0:
        print(f"\n  * Stop-band (|T|² < 1%):")
        print(f"    {stop_band[-1]:.1f} nm - {stop_band[0]:.1f} nm")
        print(f"    Width: {stop_band[0] - stop_band[-1]:.1f} nm")

    # Print at design wavelength
    idx_design = np.argmin(np.abs(wavelengths - lambda_design * 1e9))
    print(f"\n  At design wavelength ({lambda_design * 1e9:.0f} nm):")
    print(f"    |T|² = {T[idx_design]:.6f}")
    print(f"    |R|² = {R[idx_design]:.6f}")
    print(f"    v_g  = {v_g[idx_design]:.2e} m/s")

    # Energy conservation check
    print(f"\n  Energy conservation check (T+R at design λ): {T[idx_design] + R[idx_design]:.6f}")

    # --- Piezoelectric crystal stack ---
    print("\n--- Piezoelectric Crystal Stack ---")
    crystal_layers = [
        {"eps_r": 4.5, "mu_r": 1.0, "thickness": 0.5e-3, "sigma": 0.001},    # Quartz
        {"eps_r": 1700.0, "mu_r": 1.0, "thickness": 0.1e-3, "sigma": 0.01},   # BaTiO3
        {"eps_r": 4.5, "mu_r": 1.0, "thickness": 0.5e-3, "sigma": 0.001},    # Quartz
    ]
    prop2 = WavePropagator(crystal_layers)
    freqs2 = np.linspace(1e9, 50e9, 2000)
    T2 = prop2.transmission_spectrum(freqs2)
    R2 = prop2.reflection_spectrum(freqs2)

    print(f"  Stack: Quartz / BaTiO₃ / Quartz")
    print(f"  Transmission at 10 GHz: {T2[np.argmin(np.abs(freqs2 - 10e9))]:.4f}")
    print(f"  Transmission at 30 GHz: {T2[np.argmin(np.abs(freqs2 - 30e9))]:.4f}")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        ax = axes[0, 0]
        ax.plot(wavelengths, T, "b-", label="|T|²")
        ax.plot(wavelengths, R, "r-", label="|R|²")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Coefficient")
        ax.set_title(f"Bragg Mirror ({n_pairs} pairs)")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(wavelengths, v_g / c0, "g-")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("v_g / c")
        ax.set_title("Group Velocity (normalised)")
        ax.set_ylim(-2, 5)

        ax = axes[1, 0]
        ax.plot(freqs2 / 1e9, T2, "b-", label="|T|²")
        ax.plot(freqs2 / 1e9, R2, "r-", label="|R|²")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Coefficient")
        ax.set_title("Quartz/BaTiO₃/Quartz Stack")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(freqs2 / 1e9, T2 + R2, "k-")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("T + R")
        ax.set_title("Energy Conservation Check")
        ax.set_ylim(0.9, 1.1)

        plt.tight_layout()
        plt.savefig("wave_prop_demo_output.png", dpi=150)
        print("\n  Plot saved to wave_prop_demo_output.png")
    except ImportError:
        print("\n  (matplotlib not available - skipping plot)")

    print()
