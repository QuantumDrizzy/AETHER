"""Mechanical metamaterials — effective properties set by structure, not material.

The re-entrant honeycomb and its negative (auxetic) Poisson's ratio: the same base
material gives ν > 0 or ν < 0 purely by changing the rib geometry. Complements the
electromagnetic metamaterial in research/em_simulation.
"""

from research.metamaterials.auxetic import (
    is_auxetic,
    poisson_ratio_12,
    poisson_ratio_21,
    sweep_angle,
)

__all__ = ["poisson_ratio_12", "poisson_ratio_21", "is_auxetic", "sweep_angle"]
