"""
Multi-Objective Material Combination Optimizer
================================================
Pure-NumPy evolutionary Pareto optimizer - no pymoo dependency required.
Uses NSGA-II–style non-dominated sorting and crowding distance.

Run standalone:
    python -m research.quantum_annealing.optimizer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ParetoSolution:
    """A single Pareto-optimal solution."""

    indices: list[int]
    materials: list[str]
    objectives: dict[str, float]
    rank: int = 0
    crowding: float = 0.0


class CombinationOptimizer:
    """Find Pareto-optimal material combinations over multiple objectives.

    Uses a simple evolutionary approach with non-dominated sorting
    (NSGA-II style) - pure NumPy, no pymoo needed.

    Parameters
    ----------
    materials : list[dict]
        Material specifications with ``name`` and ``properties`` keys.
    """

    # Supported objective names -> (property key, direction +1=maximise -1=minimise)
    OBJECTIVE_MAP: dict[str, tuple[str, int]] = {
        "piezoelectric": ("piezo_d33", 1),
        "hardness": ("hardness", 1),
        "density_low": ("density", -1),
        "bandgap": ("bandgap", 1),
        "acoustic_velocity": ("acoustic_velocity", 1),
        "thermal_stability": ("thermal_expansion", -1),  # lower CTE = more stable
        "permittivity": ("epsilon_r", 1),
    }

    def __init__(self, materials: list[dict[str, Any]]) -> None:
        self.materials = materials
        self.n = len(materials)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        objectives: list[str],
        k: int = 3,
        generations: int = 100,
        pop_size: int = 50,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Run multi-objective optimisation over material combinations.

        Parameters
        ----------
        objectives : list[str]
            Names of objectives (see ``OBJECTIVE_MAP``).
        k : int
            Number of materials to select per combination.
        generations : int
            Number of evolutionary generations.
        pop_size : int
            Population size.
        seed : int
            Random seed.

        Returns
        -------
        list[dict]
            Pareto-optimal solutions with ``indices``, ``materials``,
            ``objectives``, ``rank``, ``crowding``.
        """
        rng = np.random.default_rng(seed)
        obj_specs = [(self.OBJECTIVE_MAP[o], o) for o in objectives if o in self.OBJECTIVE_MAP]
        n_obj = len(obj_specs)
        if n_obj == 0:
            raise ValueError(f"No valid objectives. Choose from {list(self.OBJECTIVE_MAP)}")

        from itertools import combinations as _combs

        all_combos = list(_combs(range(self.n), k))
        n_combos = len(all_combos)

        if n_combos <= pop_size:
            # Small enough to evaluate exhaustively
            population = list(range(n_combos))
        else:
            population = rng.choice(n_combos, size=pop_size, replace=False).tolist()

        def _evaluate(combo_idx: int) -> np.ndarray:
            combo = all_combos[combo_idx]
            scores = np.zeros(n_obj, dtype=np.float64)
            for oi, ((prop_key, direction), _name) in enumerate(obj_specs):
                vals = [
                    float(self.materials[i].get("properties", {}).get(prop_key, 0.0))
                    for i in combo
                ]
                # Aggregate: mean value × direction (we minimise, so negate if maximising)
                scores[oi] = -direction * np.mean(vals)
            return scores

        # Evolutionary loop
        for _gen in range(generations):
            # Evaluate population
            fitnesses = np.array([_evaluate(idx) for idx in population])
            ranks, crowding = self._nds_crowding(fitnesses)

            # Selection + offspring
            new_pop: list[int] = []
            while len(new_pop) < len(population):
                # Binary tournament
                i1, i2 = rng.choice(len(population), size=2, replace=False)
                winner = i1 if (ranks[i1] < ranks[i2] or
                                (ranks[i1] == ranks[i2] and crowding[i1] > crowding[i2])) else i2
                new_pop.append(population[winner])

            # Mutation: swap a random combo member with probability 0.3
            mutated: list[int] = []
            for idx in new_pop:
                if rng.random() < 0.3 and n_combos > 1:
                    mutated.append(int(rng.choice(n_combos)))
                else:
                    mutated.append(idx)

            population = mutated

        # Final evaluation and extract Pareto front
        fitnesses = np.array([_evaluate(idx) for idx in population])
        ranks, crowding = self._nds_crowding(fitnesses)

        results: list[dict[str, Any]] = []
        seen: set[tuple[int, ...]] = set()
        for pi, idx in enumerate(population):
            if ranks[pi] == 0:
                combo = all_combos[idx]
                if combo in seen:
                    continue
                seen.add(combo)
                obj_values = {}
                for oi, ((_prop_key, _dir), name) in enumerate(obj_specs):
                    obj_values[name] = -fitnesses[pi, oi]  # un-negate
                results.append({
                    "indices": list(combo),
                    "materials": [self.materials[i]["name"] for i in combo],
                    "objectives": obj_values,
                    "rank": int(ranks[pi]),
                    "crowding": float(crowding[pi]),
                })

        # Sort by crowding distance (most spread-out first)
        results.sort(key=lambda r: -r["crowding"])
        return results

    # ------------------------------------------------------------------
    # NSGA-II Non-Dominated Sorting + Crowding Distance
    # ------------------------------------------------------------------

    @staticmethod
    def _nds_crowding(fitnesses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Non-dominated sorting and crowding distance.

        Parameters
        ----------
        fitnesses : np.ndarray, shape (N, M)
            Objective values (minimisation).

        Returns
        -------
        ranks : np.ndarray, shape (N,)
        crowding : np.ndarray, shape (N,)
        """
        n = fitnesses.shape[0]
        ranks = np.full(n, -1, dtype=np.int32)
        crowding = np.zeros(n, dtype=np.float64)

        remaining = set(range(n))
        rank = 0

        while remaining:
            non_dominated: list[int] = []
            for p in remaining:
                is_dominated = False
                for q in remaining:
                    if p == q:
                        continue
                    # q dominates p if q <= p on all objectives and q < p on at least one
                    if np.all(fitnesses[q] <= fitnesses[p]) and np.any(fitnesses[q] < fitnesses[p]):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(p)

            for p in non_dominated:
                ranks[p] = rank
                remaining.discard(p)

            # Crowding distance for this front
            if len(non_dominated) > 2:
                front_fit = fitnesses[non_dominated]
                m = front_fit.shape[1]
                cd = np.zeros(len(non_dominated), dtype=np.float64)
                for obj_i in range(m):
                    order = np.argsort(front_fit[:, obj_i])
                    cd[order[0]] = np.inf
                    cd[order[-1]] = np.inf
                    f_range = front_fit[order[-1], obj_i] - front_fit[order[0], obj_i]
                    if f_range > 0:
                        for si in range(1, len(order) - 1):
                            cd[order[si]] += (
                                (front_fit[order[si + 1], obj_i] - front_fit[order[si - 1], obj_i])
                                / f_range
                            )
                for ci, p in enumerate(non_dominated):
                    crowding[p] = cd[ci]
            elif len(non_dominated) <= 2:
                for p in non_dominated:
                    crowding[p] = np.inf

            rank += 1

        return ranks, crowding


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    from research.quantum_annealing.qubo import DEMO_MATERIALS

    np.random.seed(42)

    print("=" * 65)
    print("  AETHER - Multi-Objective Combination Optimizer Demo")
    print("=" * 65)

    optimizer = CombinationOptimizer(DEMO_MATERIALS)

    objectives = ["piezoelectric", "hardness", "thermal_stability"]
    print(f"\nObjectives: {objectives}")
    print(f"Selecting k=3 from {len(DEMO_MATERIALS)} materials\n")

    results = optimizer.optimize(objectives, k=3, generations=80, pop_size=40)

    print(f"Found {len(results)} Pareto-optimal combinations:\n")
    for i, sol in enumerate(results[:8]):
        print(f"  Solution {i + 1}:")
        print(f"    Materials : {', '.join(sol['materials'])}")
        for obj_name, obj_val in sol["objectives"].items():
            print(f"    {obj_name:22s} = {obj_val:.4f}")
        print(f"    Crowding distance    = {sol['crowding']:.4f}")
        print()
