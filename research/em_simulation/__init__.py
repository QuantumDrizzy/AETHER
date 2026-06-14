"""Electromagnetic simulation module - FDTD, crystal resonance, metamaterials."""

from research.em_simulation.resonance import FDTDSimulator1D, CrystalResonanceSimulator
from research.em_simulation.metamaterial import MetamaterialSimulator
from research.em_simulation.wave_prop import WavePropagator

__all__ = [
    "FDTDSimulator1D",
    "CrystalResonanceSimulator",
    "MetamaterialSimulator",
    "WavePropagator",
]
