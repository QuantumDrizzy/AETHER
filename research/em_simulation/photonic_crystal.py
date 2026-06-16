"""1D photonic crystal — a photonic band gap from periodic structure (Bragg stack).

The complement to the effective-medium negative-index metamaterial in
`metamaterial.py`: here the gap is *structural*, not resonant. A periodic stack of
two dielectric layers (indices n1, n2) reflects light in a band of frequencies —
the photonic band gap — by Bragg interference, exactly as the auxetic honeycomb's
property came from geometry. The propagating Bloch modes satisfy

    cos(K·Λ) = ½ Tr(M_cell),     M_cell = M(n1, d1) · M(n2, d2),

with the characteristic layer matrix M(n,d) = [[cos δ, i sinδ/n],[i n sinδ, cos δ]],
δ = 2π f n d (units c = 1). When |½ Tr(M)| > 1 there is no real Bloch wavevector:
light cannot propagate — a band gap. A quarter-wave stack (n1 d1 = n2 d2 = λ0/4)
opens its first gap centred at the design frequency f0; the gap widens with index
contrast, and a uniform medium (n1 = n2) has no gap at all.
"""

from __future__ import annotations

import numpy as np


class PhotonicCrystal:
    def __init__(self, n1: float = 1.0, n2: float = 2.0, f0: float = 1.0):
        self.n1, self.n2, self.f0 = n1, n2, f0
        # quarter-wave optical thickness at the design frequency (c = 1): n·d = 1/(4 f0)
        self.d1 = 1.0 / (4.0 * n1 * f0)
        self.d2 = 1.0 / (4.0 * n2 * f0)
        self.period = self.d1 + self.d2

    @staticmethod
    def _layer_matrix(n: float, d: float, f: float) -> np.ndarray:
        delta = 2.0 * np.pi * f * n * d
        return np.array([[np.cos(delta), 1j * np.sin(delta) / n],
                         [1j * n * np.sin(delta), np.cos(delta)]])

    def bloch_cos(self, f: float) -> float:
        """½ Tr(M_cell) = cos(K·Λ). |·| > 1 means a band gap (evanescent)."""
        m = self._layer_matrix(self.n1, self.d1, f) @ self._layer_matrix(self.n2, self.d2, f)
        return float((m[0, 0] + m[1, 1]).real / 2.0)

    def is_gap(self, f: float) -> bool:
        return abs(self.bloch_cos(f)) > 1.0

    def band_gaps(self, f_max: float = 3.0, n: int = 4000):
        """Return the list of (f_low, f_high) photonic band gaps up to f_max."""
        fs = np.linspace(1e-4, f_max, n)
        gap = np.array([self.is_gap(f) for f in fs])
        out, start = [], None
        for i, g in enumerate(gap):
            if g and start is None:
                start = fs[i]
            elif not g and start is not None:
                out.append((start, fs[i - 1]))
                start = None
        if start is not None:
            out.append((start, fs[-1]))
        return out

    def first_gap(self, f_max: float = 2.0):
        gaps = self.band_gaps(f_max)
        return gaps[0] if gaps else None
