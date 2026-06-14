"""
QUBO Formulation for Material Compatibility Optimization
=========================================================
Quadratic Unconstrained Binary Optimization formulation that encodes
material selection as a combinatorial optimization problem suitable
for quantum/simulated annealing.

Run standalone:
    python -m research.quantum_annealing.qubo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MaterialSpec:
    """Canonical material specification used across AETHER modules."""

    name: str
    properties: dict[str, float] = field(default_factory=dict)

    # Convenience accessors for common properties --------------------------
    def get(self, key: str, default: float = 0.0) -> float:
        return self.properties.get(key, default)


class QUBOFormulator:
    """Build a QUBO (Q-matrix) for selecting *k* most compatible materials.

    The Q-matrix encodes:
      * **Linear terms** (diagonal): individual material quality scores
        (negated so the solver *minimises* energy -> *maximises* quality).
      * **Quadratic terms** (off-diagonal): pairwise compatibility scores
        (negated likewise).
      * **Constraint penalty**: ``penalty_weight * (sum(x_i) - k)^2``
        ensures exactly *k* materials are selected.

    Parameters
    ----------
    materials : list[dict]
        Each dict must have ``name`` (str) and ``properties`` (dict[str, float]).
    compatibility_matrix : np.ndarray
        Symmetric (N×N) matrix of pairwise compatibility, values in [0, 1].
    """

    def __init__(
        self,
        materials: list[dict[str, Any]],
        compatibility_matrix: np.ndarray,
    ) -> None:
        self.materials = materials
        self.n = len(materials)
        if compatibility_matrix.shape != (self.n, self.n):
            raise ValueError(
                f"compatibility_matrix shape {compatibility_matrix.shape} "
                f"does not match {self.n} materials"
            )
        self.compatibility = compatibility_matrix
        self._individual_scores: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def formulate(
        self,
        select_k: int = 3,
        penalty_weight: float = 10.0,
    ) -> dict[tuple[int, int], float]:
        """Return the QUBO Q-matrix as ``{(i, j): coefficient}``.

        Parameters
        ----------
        select_k : int
            Number of materials to select.
        penalty_weight : float
            Lagrange multiplier for the cardinality constraint.

        Returns
        -------
        dict[tuple[int, int], float]
            Sparse QUBO dictionary.
        """
        Q: dict[tuple[int, int], float] = {}
        indiv = self._compute_individual_scores()
        pair = self._compute_pairwise_scores()

        # --- Objective: maximise quality -> minimise -quality -------------
        for i in range(self.n):
            Q[(i, i)] = Q.get((i, i), 0.0) - indiv[i]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                Q[(i, j)] = Q.get((i, j), 0.0) - pair[i, j]

        # --- Constraint: sum(x_i) == k via penalty (sum - k)^2 ----------
        # Expand: penalty * [sum_i x_i^2 + 2*sum_{i<j} x_i*x_j
        #                     - 2k*sum_i x_i + k^2]
        # x_i^2 == x_i (binary), so linear terms += penalty*(1 - 2k)
        for i in range(self.n):
            Q[(i, i)] = Q.get((i, i), 0.0) + penalty_weight * (1.0 - 2.0 * select_k)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                Q[(i, j)] = Q.get((i, j), 0.0) + 2.0 * penalty_weight
        # constant k^2 * penalty is irrelevant for optimisation

        return Q

    def decode_solution(self, sample: dict[int, int]) -> list[int]:
        """Return list of selected material indices from a binary sample."""
        return sorted(i for i, v in sample.items() if v == 1)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_individual_scores(self) -> np.ndarray:
        """Score each material individually based on its properties.

        Combines normalised physical properties into a single quality
        metric in [0, 1].
        """
        if self._individual_scores is not None:
            return self._individual_scores

        scores = np.zeros(self.n, dtype=np.float64)

        # Collect all numeric property values across materials
        all_keys: set[str] = set()
        for m in self.materials:
            all_keys.update(m.get("properties", {}).keys())
        if not all_keys:
            self._individual_scores = np.ones(self.n) / self.n
            return self._individual_scores

        # Build feature matrix and z-score normalise per property
        raw = np.zeros((self.n, len(all_keys)), dtype=np.float64)
        keys_list = sorted(all_keys)
        for i, m in enumerate(self.materials):
            props = m.get("properties", {})
            for j, k in enumerate(keys_list):
                raw[i, j] = float(props.get(k, 0.0))

        # Z-score per column, then sigmoid -> [0, 1]
        means = raw.mean(axis=0)
        stds = raw.std(axis=0)
        stds[stds == 0] = 1.0
        z = (raw - means) / stds
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        scores = sigmoid.mean(axis=1)

        self._individual_scores = scores
        return scores

    def _compute_pairwise_scores(self) -> np.ndarray:
        """Pairwise compatibility scores (delegates to the supplied matrix)."""
        return self.compatibility.copy()


# ======================================================================
# Demo materials database
# ======================================================================

DEMO_MATERIALS: list[dict[str, Any]] = [
    {
        "name": "Quartz (SiO2)",
        "properties": {
            "piezo_d33": 2.3,       # pC/N
            "epsilon_r": 4.5,
            "hardness": 7.0,        # Mohs
            "density": 2650.0,      # kg/m^3
            "thermal_expansion": 0.55e-6,  # 1/K
            "lattice_a": 4.913,     # Å
            "bandgap": 8.9,         # eV
            "acoustic_velocity": 5760.0,  # m/s
        },
    },
    {
        "name": "Tourmaline",
        "properties": {
            "piezo_d33": 1.8,
            "epsilon_r": 7.1,
            "hardness": 7.5,
            "density": 3100.0,
            "thermal_expansion": 4.0e-6,
            "lattice_a": 15.84,
            "bandgap": 3.2,
            "acoustic_velocity": 7500.0,
        },
    },
    {
        "name": "PZT (Pb[Zr,Ti]O3)",
        "properties": {
            "piezo_d33": 593.0,
            "epsilon_r": 3400.0,
            "hardness": 5.5,
            "density": 7600.0,
            "thermal_expansion": 2.0e-6,
            "lattice_a": 4.036,
            "bandgap": 3.4,
            "acoustic_velocity": 4600.0,
        },
    },
    {
        "name": "BaTiO3",
        "properties": {
            "piezo_d33": 190.0,
            "epsilon_r": 1700.0,
            "hardness": 5.0,
            "density": 6020.0,
            "thermal_expansion": 6.3e-6,
            "lattice_a": 3.996,
            "bandgap": 3.2,
            "acoustic_velocity": 5200.0,
        },
    },
    {
        "name": "Graphene",
        "properties": {
            "piezo_d33": 0.0,
            "epsilon_r": 3.3,
            "hardness": 10.0,       # effective
            "density": 2267.0,
            "thermal_expansion": -8.0e-6,
            "lattice_a": 2.46,
            "bandgap": 0.0,
            "acoustic_velocity": 21300.0,
        },
    },
    {
        "name": "Bismuth (Bi)",
        "properties": {
            "piezo_d33": 0.0,
            "epsilon_r": 100.0,
            "hardness": 2.25,
            "density": 9780.0,
            "thermal_expansion": 13.4e-6,
            "lattice_a": 4.546,
            "bandgap": 0.014,
            "acoustic_velocity": 1790.0,
        },
    },
]


def _build_demo_compatibility(n: int, seed: int = 42) -> np.ndarray:
    """Build a synthetic but plausible pairwise compatibility matrix."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 0.95, size=(n, n))
    sym = (raw + raw.T) / 2.0
    np.fill_diagonal(sym, 1.0)
    return sym


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    np.random.seed(42)

    materials = DEMO_MATERIALS
    compat = _build_demo_compatibility(len(materials))

    print("=" * 65)
    print("  AETHER - QUBO Formulation Demo")
    print("=" * 65)
    print(f"\nMaterials ({len(materials)}):")
    for i, m in enumerate(materials):
        print(f"  [{i}] {m['name']}")

    formulator = QUBOFormulator(materials, compat)

    print("\n--- Individual quality scores ---")
    scores = formulator._compute_individual_scores()
    for i, m in enumerate(materials):
        print(f"  {m['name']:25s}  ->  {scores[i]:.4f}")

    print("\n--- Pairwise compatibility (top-5) ---")
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(materials)):
        for j in range(i + 1, len(materials)):
            pairs.append((compat[i, j], i, j))
    pairs.sort(reverse=True)
    for score, i, j in pairs[:5]:
        print(f"  {materials[i]['name']:20s} ↔ {materials[j]['name']:20s}  {score:.4f}")

    Q = formulator.formulate(select_k=3, penalty_weight=8.0)
    print(f"\nQUBO size: {len(Q)} non-zero entries")
    print(f"Diagonal (linear) terms: {sum(1 for k in Q if k[0] == k[1])}")
    print(f"Off-diag (quadratic) terms: {sum(1 for k in Q if k[0] != k[1])}")

    # Brute-force optimal (feasible for 6 materials)
    from itertools import combinations

    best_energy = float("inf")
    best_combo: tuple[int, ...] = ()
    for combo in combinations(range(len(materials)), 3):
        sample = {i: (1 if i in combo else 0) for i in range(len(materials))}
        energy = sum(Q.get((i, j), 0.0) * sample[i] * sample[j]
                     for i in range(len(materials))
                     for j in range(i, len(materials)))
        if energy < best_energy:
            best_energy = energy
            best_combo = combo

    print(f"\n* Optimal 3-material combination (brute-force):")
    print(f"  Energy = {best_energy:.4f}")
    for idx in best_combo:
        print(f"    -> {materials[idx]['name']}")
    print()
