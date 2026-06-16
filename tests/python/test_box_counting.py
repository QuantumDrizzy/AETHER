"""Validate box-counting dimension against fractals with known closed-form D_f."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.fractals.box_counting import (  # noqa: E402
    SIERPINSKI_CARPET_DF,
    SIERPINSKI_TRIANGLE_DF,
    box_count,
    box_counting_dimension,
    filled_square,
    sierpinski_carpet,
    sierpinski_triangle,
)


def test_sierpinski_triangle_dimension():
    df = box_counting_dimension(sierpinski_triangle(8), [1, 2, 4, 8, 16, 32, 64])
    assert abs(df - SIERPINSKI_TRIANGLE_DF) < 0.02      # log3/log2 ~ 1.585


def test_sierpinski_carpet_dimension():
    df = box_counting_dimension(sierpinski_carpet(5), [1, 3, 9, 27, 81])
    assert abs(df - SIERPINSKI_CARPET_DF) < 0.02        # log8/log3 ~ 1.893


def test_filled_square_is_dimension_two():
    df = box_counting_dimension(filled_square(256), [1, 2, 4, 8, 16, 32, 64])
    assert abs(df - 2.0) < 0.02


def test_box_count_increases_at_finer_scale():
    g = sierpinski_triangle(7)
    assert box_count(g, 1) > box_count(g, 8) > box_count(g, 64)
