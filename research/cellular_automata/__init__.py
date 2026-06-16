"""Cellular automata — discrete computational substrates (matter computes by rule).

Conway's Game of Life: a 2-state CA whose local rule yields still lifes,
oscillators, travelling gliders, and Turing-completeness. The discrete cousin of
DRIFT's Ising substrate.
"""

from research.cellular_automata.game_of_life import (
    blinker,
    block,
    density,
    glider,
    run,
    step,
    translated_equal,
)

__all__ = ["step", "run", "density", "block", "blinker", "glider", "translated_equal"]
