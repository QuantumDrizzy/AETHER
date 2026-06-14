"""
Crystal Resonance & 1-D FDTD EM Simulation
=============================================
Pure-NumPy 1-D Finite-Difference Time-Domain simulator with PML
boundaries and a crystal resonance analyser built on top.

Run standalone:
    python -m research.em_simulation.resonance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class SimulationResult:
    """Container for FDTD simulation output."""

    E_field_history: np.ndarray       # (num_steps, grid_size)
    H_field_history: np.ndarray       # (num_steps, grid_size)
    time_array: np.ndarray            # (num_steps,)
    frequencies: np.ndarray | None = None
    spectrum: np.ndarray | None = None


# ======================================================================
# 1-D FDTD Simulator
# ======================================================================

class FDTDSimulator1D:
    """1-D Finite-Difference Time-Domain electromagnetic simulator.

    Implements the Yee algorithm with perfectly matched layer (PML)
    absorbing boundaries.

    Parameters
    ----------
    length_m : float
        Physical length of the simulation domain (metres).
    resolution : int
        Number of spatial grid cells.
    dt_factor : float
        Courant factor (dt = dt_factor * dx / c).  Must be ≤ 1 for
        stability.
    """

    def __init__(
        self,
        length_m: float = 0.01,
        resolution: int = 200,
        dt_factor: float = 0.99,
    ) -> None:
        self.length_m = length_m
        self.resolution = resolution
        self.dx = length_m / resolution
        self.c0 = 299_792_458.0          # m/s
        self.eps0 = 8.854187817e-12      # F/m
        self.mu0 = 4.0 * np.pi * 1e-7   # H/m
        self.dt = dt_factor * self.dx / self.c0

        # Field arrays
        self.Ez = np.zeros(resolution, dtype=np.float64)
        self.Hy = np.zeros(resolution, dtype=np.float64)

        # Material coefficient arrays (initialised to vacuum)
        self.epsilon_r = np.ones(resolution, dtype=np.float64)
        self.mu_r = np.ones(resolution, dtype=np.float64)
        self.sigma = np.zeros(resolution, dtype=np.float64)

        # PML parameters
        self.pml_layers = 20
        self._pml_sigma_e = np.zeros(resolution, dtype=np.float64)
        self._pml_sigma_h = np.zeros(resolution, dtype=np.float64)
        self._setup_pml()

        # Update coefficients (recomputed when material changes)
        self._ca = np.ones(resolution, dtype=np.float64)
        self._cb = np.ones(resolution, dtype=np.float64) * self.dt / (self.eps0 * self.dx)
        self._da = np.ones(resolution, dtype=np.float64)
        self._db = np.ones(resolution, dtype=np.float64) * self.dt / (self.mu0 * self.dx)

        # Source
        self._source_idx: int = 0
        self._source_freq: float = 1e9
        self._source_type: str = "gaussian_pulse"

        self._recompute_coefficients()

    # ------------------------------------------------------------------

    def setup_material(
        self,
        epsilon_r: float = 4.5,
        mu_r: float = 1.0,
        sigma: float = 0.0,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> None:
        """Place a material slab in the simulation domain.

        Parameters
        ----------
        epsilon_r, mu_r, sigma : float
            Relative permittivity, permeability, conductivity.
        start_idx, end_idx : int, optional
            Grid indices for material region.  Defaults to central third.
        """
        n = self.resolution
        if start_idx is None:
            start_idx = n // 3
        if end_idx is None:
            end_idx = 2 * n // 3

        self.epsilon_r[start_idx:end_idx] = epsilon_r
        self.mu_r[start_idx:end_idx] = mu_r
        self.sigma[start_idx:end_idx] = sigma
        self._recompute_coefficients()

    def add_source(
        self,
        position: float = 0.2,
        frequency: float = 1e9,
        source_type: str = "gaussian_pulse",
    ) -> None:
        """Configure the excitation source.

        Parameters
        ----------
        position : float
            Fractional position along the domain [0, 1].
        frequency : float
            Centre frequency (Hz).
        source_type : str
            ``'gaussian_pulse'``, ``'sinusoidal'``, or ``'ricker'``.
        """
        self._source_idx = int(position * self.resolution)
        self._source_freq = frequency
        self._source_type = source_type

    # ------------------------------------------------------------------
    # PML
    # ------------------------------------------------------------------

    def _setup_pml(self) -> None:
        """Initialise graded PML conductivity profile."""
        pml_max_sigma = 0.8 * (3.0 + 1) / (self.dx * np.sqrt(self.mu0 / self.eps0))
        for i in range(self.pml_layers):
            depth = (self.pml_layers - i) / self.pml_layers
            sig = pml_max_sigma * depth ** 3
            self._pml_sigma_e[i] = sig
            self._pml_sigma_e[self.resolution - 1 - i] = sig
            self._pml_sigma_h[i] = sig
            self._pml_sigma_h[self.resolution - 1 - i] = sig

    def _apply_pml(self) -> None:
        """Apply PML absorption (already folded into coefficients)."""
        # PML is applied via the _ca/_cb coefficients - no separate step needed.
        pass

    # ------------------------------------------------------------------
    # Coefficient computation
    # ------------------------------------------------------------------

    def _recompute_coefficients(self) -> None:
        """Recompute update coefficients from material parameters + PML."""
        eps = self.eps0 * self.epsilon_r
        mu = self.mu0 * self.mu_r
        total_sigma_e = self.sigma + self._pml_sigma_e

        # E-field coefficients:  ca = (1 - σΔt/2ε) / (1 + σΔt/2ε)
        #                        cb = (Δt/εΔx) / (1 + σΔt/2ε)
        denom_e = 1.0 + total_sigma_e * self.dt / (2.0 * eps)
        self._ca = (1.0 - total_sigma_e * self.dt / (2.0 * eps)) / denom_e
        self._cb = (self.dt / (eps * self.dx)) / denom_e

        # H-field coefficients (PML on H)
        denom_h = 1.0 + self._pml_sigma_h * self.dt / (2.0 * mu)
        self._da = (1.0 - self._pml_sigma_h * self.dt / (2.0 * mu)) / denom_h
        self._db = (self.dt / (mu * self.dx)) / denom_h

    # ------------------------------------------------------------------
    # Yee updates
    # ------------------------------------------------------------------

    def _update_H(self) -> None:
        """Yee update for H-field (Hy)."""
        self.Hy[:-1] = (self._da[:-1] * self.Hy[:-1]
                        - self._db[:-1] * (self.Ez[1:] - self.Ez[:-1]))

    def _update_E(self) -> None:
        """Yee update for E-field (Ez)."""
        self.Ez[1:] = (self._ca[1:] * self.Ez[1:]
                       - self._cb[1:] * (self.Hy[1:] - self.Hy[:-1]))

    # ------------------------------------------------------------------
    # Source waveforms
    # ------------------------------------------------------------------

    def _source_value(self, step: int) -> float:
        t = step * self.dt
        tau = 1.0 / self._source_freq
        t0 = 3.0 * tau

        if self._source_type == "gaussian_pulse":
            return float(np.exp(-0.5 * ((t - t0) / (tau / 3.0)) ** 2)
                         * np.sin(2.0 * np.pi * self._source_freq * t))
        elif self._source_type == "sinusoidal":
            ramp = min(1.0, t / (2.0 * tau))
            return float(ramp * np.sin(2.0 * np.pi * self._source_freq * t))
        elif self._source_type == "ricker":
            arg = np.pi * self._source_freq * (t - t0)
            return float((1.0 - 2.0 * arg ** 2) * np.exp(-arg ** 2))
        return 0.0

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self, num_steps: int = 1000) -> SimulationResult:
        """Run the FDTD simulation.

        Parameters
        ----------
        num_steps : int
            Number of time steps.

        Returns
        -------
        SimulationResult
        """
        self.Ez[:] = 0.0
        self.Hy[:] = 0.0

        E_hist = np.zeros((num_steps, self.resolution), dtype=np.float64)
        H_hist = np.zeros((num_steps, self.resolution), dtype=np.float64)
        time_arr = np.arange(num_steps) * self.dt

        # Probe at centre of domain for spectrum
        probe_idx = self.resolution // 2
        probe_signal = np.zeros(num_steps, dtype=np.float64)

        for step in range(num_steps):
            self._update_H()
            self._update_E()

            # Inject source (soft source)
            self.Ez[self._source_idx] += self._source_value(step)

            E_hist[step] = self.Ez.copy()
            H_hist[step] = self.Hy.copy()
            probe_signal[step] = self.Ez[probe_idx]

        # Compute frequency spectrum via FFT
        spectrum = np.abs(np.fft.rfft(probe_signal))
        freqs = np.fft.rfftfreq(num_steps, d=self.dt)

        return SimulationResult(
            E_field_history=E_hist,
            H_field_history=H_hist,
            time_array=time_arr,
            frequencies=freqs,
            spectrum=spectrum,
        )


# ======================================================================
# Crystal Resonance Simulator
# ======================================================================

class CrystalResonanceSimulator:
    """Analyse resonant behaviour of piezoelectric crystals.

    Parameters
    ----------
    crystal_properties : dict
        Must contain:
        - ``epsilon_r`` (float): relative permittivity
        - ``thickness`` (float): crystal thickness in metres
        - ``piezo_coeff`` (float): piezoelectric d₃₃ coefficient (pC/N)
        Optional:
        - ``density`` (float): kg/m³
        - ``acoustic_velocity`` (float): m/s
        - ``Q_factor`` (float): quality factor
    """

    def __init__(self, crystal_properties: dict[str, float]) -> None:
        self.props = crystal_properties
        self.epsilon_r = crystal_properties.get("epsilon_r", 4.5)
        self.thickness = crystal_properties.get("thickness", 1e-3)
        self.piezo_d33 = crystal_properties.get("piezo_coeff", 2.3e-12)
        self.density = crystal_properties.get("density", 2650.0)
        self.v_acoustic = crystal_properties.get("acoustic_velocity", 5760.0)
        self.Q_factor = crystal_properties.get("Q_factor", 100_000.0)
        self.c0 = 299_792_458.0
        self.eps0 = 8.854187817e-12

    def find_resonances(
        self,
        freq_range: tuple[float, float] = (1e6, 1e10),
        max_harmonics: int = 20,
    ) -> list[float]:
        """Find resonant frequencies of the crystal.

        Uses the acoustic resonance condition: f_n = n * v / (2 * t).

        Parameters
        ----------
        freq_range : tuple[float, float]
            (min_freq, max_freq) in Hz.
        max_harmonics : int
            Maximum harmonic number to consider.

        Returns
        -------
        list[float]
            Resonant frequencies within the specified range.
        """
        f_fundamental = self.v_acoustic / (2.0 * self.thickness)
        resonances: list[float] = []

        for n in range(1, max_harmonics + 1):
            f_n = n * f_fundamental
            if freq_range[0] <= f_n <= freq_range[1]:
                resonances.append(f_n)
            elif f_n > freq_range[1]:
                break

        return resonances

    def compute_impedance(self, frequencies: np.ndarray) -> np.ndarray:
        """Compute electrical impedance spectrum of the crystal.

        Uses the Butterworth-van Dyke (BVD) equivalent circuit model.

        Parameters
        ----------
        frequencies : np.ndarray
            Array of frequencies (Hz).

        Returns
        -------
        np.ndarray
            Complex impedance at each frequency.
        """
        omega = 2.0 * np.pi * frequencies
        f_s = self.v_acoustic / (2.0 * self.thickness)  # series resonance

        # BVD parameters
        C0 = self.eps0 * self.epsilon_r * 1e-4 / self.thickness  # static cap (assume 1 cm² area)
        k_eff_sq = self.piezo_d33 ** 2 / (self.eps0 * self.epsilon_r * 1e-12)  # effective coupling
        k_eff_sq = min(k_eff_sq, 0.1)  # physical limit

        # Motional parameters
        C1 = 8.0 * k_eff_sq * C0 / (np.pi ** 2)
        L1 = 1.0 / ((2.0 * np.pi * f_s) ** 2 * C1) if C1 > 0 else 1.0
        R1 = 2.0 * np.pi * f_s * L1 / self.Q_factor

        # Series branch impedance
        Z_motional = R1 + 1j * omega * L1 + 1.0 / (1j * omega * C1 + 1e-30)

        # Parallel with C0
        Z_C0 = 1.0 / (1j * omega * C0 + 1e-30)
        Z_total = (Z_motional * Z_C0) / (Z_motional + Z_C0 + 1e-30)

        return Z_total

    def simulate_piezo_response(
        self,
        excitation_freq: float,
        duration: float = 1e-4,
        num_points: int = 5000,
    ) -> dict[str, Any]:
        """Simulate piezoelectric response at a given excitation frequency.

        Parameters
        ----------
        excitation_freq : float
            Excitation frequency (Hz).
        duration : float
            Simulation duration (s).
        num_points : int
            Number of time points.

        Returns
        -------
        dict
            ``time``, ``voltage``, ``displacement``, ``charge``,
            ``resonance_factor``.
        """
        t = np.linspace(0, duration, num_points)
        omega = 2.0 * np.pi * excitation_freq

        # Find nearest resonance
        resonances = self.find_resonances()
        f_nearest = min(resonances, key=lambda f: abs(f - excitation_freq)) if resonances else excitation_freq

        # Resonance enhancement factor (Lorentzian)
        gamma = f_nearest / (2.0 * self.Q_factor)
        resonance_factor = gamma ** 2 / ((excitation_freq - f_nearest) ** 2 + gamma ** 2)

        # Applied voltage
        V_applied = np.sin(omega * t)

        # Displacement: d33 * V * resonance_enhancement
        d33_m_per_V = self.piezo_d33 * 1e-12  # convert pC/N to m/V (approximate)
        displacement = d33_m_per_V * V_applied * (1.0 + (self.Q_factor - 1) * resonance_factor)

        # Surface charge
        C0 = self.eps0 * self.epsilon_r * 1e-4 / self.thickness
        charge = C0 * V_applied

        return {
            "time": t,
            "voltage": V_applied,
            "displacement": displacement,
            "charge": charge,
            "resonance_factor": float(resonance_factor),
            "nearest_resonance_hz": float(f_nearest),
        }


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 65)
    print("  AETHER - Crystal Resonance & FDTD Demo")
    print("=" * 65)

    # ----- 1-D FDTD simulation ----------------------------------------
    print("\n--- 1-D FDTD Simulation ---")
    sim = FDTDSimulator1D(length_m=0.01, resolution=300, dt_factor=0.99)
    sim.setup_material(epsilon_r=4.5, mu_r=1.0, sigma=0.001)
    sim.add_source(position=0.15, frequency=5e9, source_type="gaussian_pulse")

    result = sim.run(num_steps=2000)
    print(f"  Grid cells     : {sim.resolution}")
    print(f"  dx             : {sim.dx * 1e6:.2f} µm")
    print(f"  dt             : {sim.dt * 1e12:.4f} ps")
    print(f"  Steps          : {result.time_array.shape[0]}")
    print(f"  Max |E| at end : {np.max(np.abs(result.E_field_history[-1])):.6f} V/m")

    if result.spectrum is not None and result.frequencies is not None:
        peak_idx = np.argmax(result.spectrum[1:]) + 1
        print(f"  Peak freq      : {result.frequencies[peak_idx] / 1e9:.3f} GHz")
        print(f"  Spectrum points: {len(result.spectrum)}")

    # ----- Crystal resonance -------------------------------------------
    print("\n--- Quartz Crystal Resonator ---")
    quartz = {
        "epsilon_r": 4.5,
        "thickness": 0.5e-3,         # 0.5 mm
        "piezo_coeff": 2.3,          # pC/N
        "density": 2650.0,
        "acoustic_velocity": 5760.0,
        "Q_factor": 100_000.0,
    }
    crystal = CrystalResonanceSimulator(quartz)

    resonances = crystal.find_resonances(freq_range=(1e6, 50e6))
    print(f"  Thickness  : {quartz['thickness'] * 1e3:.2f} mm")
    print(f"  ε_r        : {quartz['epsilon_r']}")
    print(f"  Q factor   : {quartz['Q_factor']:.0f}")
    print(f"  Resonances found: {len(resonances)}")
    for i, f in enumerate(resonances[:5]):
        print(f"    f_{i + 1} = {f / 1e6:.3f} MHz")

    # Impedance spectrum around fundamental
    f_fund = resonances[0] if resonances else 5.76e6
    f_scan = np.linspace(f_fund * 0.95, f_fund * 1.05, 2000)
    Z = crystal.compute_impedance(f_scan)
    Z_mag = np.abs(Z)
    print(f"\n  Impedance around f₁ = {f_fund / 1e6:.3f} MHz:")
    print(f"    Min |Z| = {np.min(Z_mag):.2f} Ω  (series resonance)")
    print(f"    Max |Z| = {np.max(Z_mag):.2f} Ω  (parallel resonance)")

    # Piezoelectric response
    resp = crystal.simulate_piezo_response(f_fund, duration=5e-6)
    print(f"\n  Piezo response at f₁:")
    print(f"    Resonance factor  : {resp['resonance_factor']:.4f}")
    print(f"    Max displacement  : {np.max(np.abs(resp['displacement'])) * 1e9:.4f} nm")
    print(f"    Max charge        : {np.max(np.abs(resp['charge'])) * 1e12:.4f} pC")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # E-field snapshot
        ax = axes[0, 0]
        steps_to_show = [100, 500, 1000, 1500]
        for s in steps_to_show:
            if s < result.E_field_history.shape[0]:
                ax.plot(result.E_field_history[s], alpha=0.7, label=f"step {s}")
        ax.set_title("1-D FDTD: E-field snapshots")
        ax.set_xlabel("Grid index")
        ax.set_ylabel("Ez (V/m)")
        ax.legend(fontsize=8)

        # Spectrum
        ax = axes[0, 1]
        mask = result.frequencies < 20e9
        ax.plot(result.frequencies[mask] / 1e9, result.spectrum[mask])
        ax.set_title("Frequency Spectrum")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("|FFT|")

        # Impedance
        ax = axes[1, 0]
        ax.semilogy(f_scan / 1e6, Z_mag)
        ax.set_title(f"Quartz Impedance (near {f_fund / 1e6:.1f} MHz)")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("|Z| (Ω)")

        # Piezo displacement
        ax = axes[1, 1]
        ax.plot(resp["time"] * 1e6, resp["displacement"] * 1e9)
        ax.set_title("Piezoelectric Displacement")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Displacement (nm)")

        plt.tight_layout()
        plt.savefig("resonance_demo_output.png", dpi=150)
        print("\n  Plot saved to resonance_demo_output.png")
    except ImportError:
        print("\n  (matplotlib not available - skipping plot)")

    print()
