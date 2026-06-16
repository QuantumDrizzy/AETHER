"""Active matter — self-propelled particles and emergent collective order.

The Vicsek flocking model: global order (a coherent swarm direction) emerging from
a local alignment rule, with a noise-driven phase transition. Agency in matter at
the mesoscale, between the micro daemon (Maxwell) and macro programmable matter.
"""

from research.active_matter.vicsek import VicsekModel, sweep_noise

__all__ = ["VicsekModel", "sweep_noise"]
