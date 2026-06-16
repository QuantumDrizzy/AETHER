"""Neuro — attractor networks (associative memory).

Hopfield network: Hebbian storage, attractor recall, and the Amit-Gutfreund-
Sompolinsky storage-capacity collapse (alpha_c ~ 0.138). The neural-memory face of
DRIFT's Ising engine; ties to the KHAOS neuroscience line.
"""

from research.neuro.hopfield import Hopfield, capacity_scan, random_patterns

__all__ = ["Hopfield", "capacity_scan", "random_patterns"]
