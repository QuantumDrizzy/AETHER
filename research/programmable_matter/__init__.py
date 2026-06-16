"""Programmable matter — self-reconfiguration and self-repair on a grid.

Modular units rearrange into a target shape by local moves + energy descent, and
recover gracefully under damage (the macro end of the daemons spectrum: the
Columbia "robot metabolism" abstraction, made measurable).
"""

from research.programmable_matter.reconfigure import (
    ReconfigurableMatter,
    square_target,
)

__all__ = ["ReconfigurableMatter", "square_target"]
