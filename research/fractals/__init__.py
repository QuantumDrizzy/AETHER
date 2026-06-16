"""Fractals — measuring self-similarity (box-counting dimension).

The estimator behind every "fractal" claim in the self-organisation line, validated
on deterministic fractals with known dimension (Sierpinski triangle/carpet).
"""

from research.fractals.box_counting import (
    SIERPINSKI_CARPET_DF,
    SIERPINSKI_TRIANGLE_DF,
    box_count,
    box_counting_dimension,
    filled_square,
    sierpinski_carpet,
    sierpinski_triangle,
)

__all__ = [
    "box_count",
    "box_counting_dimension",
    "sierpinski_triangle",
    "sierpinski_carpet",
    "filled_square",
    "SIERPINSKI_TRIANGLE_DF",
    "SIERPINSKI_CARPET_DF",
]
