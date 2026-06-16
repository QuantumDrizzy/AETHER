"""Networks — graph structure (connectivity beside percolation/criticality).

Watts-Strogatz small-world: a little rewiring collapses path length while keeping
clustering high. Pure NumPy + BFS.
"""

from research.networks.small_world import (
    avg_path_length,
    clustering_coefficient,
    sweep_rewiring,
    watts_strogatz,
)

__all__ = ["watts_strogatz", "clustering_coefficient", "avg_path_length", "sweep_rewiring"]
