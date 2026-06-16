"""Universal criticality — the same order/disorder transition across Ising
(computronium), Vicsek (active matter) and reconfiguration (programmable matter).
The through-line of the daemons / computronium line: criticality is
substrate-independent."""

from research.criticality.universal import (
    ising_order,
    reconfig_order,
    transition_point,
    vicsek_order,
)

__all__ = ["ising_order", "vicsek_order", "reconfig_order", "transition_point"]
