"""Electronic-structure models — the scientific spine of AETHER.

Currently: tight-binding graphene (the validatable reference). Future: general
tight-binding from crystal structures, GPU-batched k-space, interfaces to DFT /
Materials Project data. See docs/ADR-0001.
"""

from research.electronic_structure.graphene import GrapheneTB

__all__ = ["GrapheneTB"]
