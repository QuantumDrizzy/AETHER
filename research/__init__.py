"""
AETHER LAB - Advanced Materials Research Engine
================================================
Python research layer for quantum-informed materials science.

Modules
-------
- quantum_annealing : QUBO formulation + simulated quantum annealing
- reservoir_computing : Echo State Networks for material behavior prediction
- em_simulation : FDTD electromagnetic simulation & crystal resonance
- compatibility : Multi-metric material compatibility scoring
- knowledge : Pattern learning & cross-experiment correlation
"""

__version__ = "0.1.0"
__author__ = "AETHER LAB"

from typing import Final

PACKAGE_NAME: Final[str] = "aether-research"
DESCRIPTION: Final[str] = (
    "Python research layer for AETHER LAB - "
    "quantum-informed materials science platform"
)

__all__ = [
    "__version__",
    "PACKAGE_NAME",
    "DESCRIPTION",
]
