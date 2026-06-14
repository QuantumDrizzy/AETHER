"""
Simulated Quantum Annealing Engine
====================================
Provides both OpenJij-based SQA and a pure-NumPy simulated annealing
fallback that works without any external quantum-computing libraries.

Run standalone:
    python -m research.quantum_annealing.annealer
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AnnealingResult:
    """Result container from an annealing run."""

    sample: dict[int, int]
    energy: float
    num_reads: int
    timing_ms: float
    method: str
    all_energies: list[float] | None = None

    def selected_indices(self) -> list[int]:
        """Return sorted list of indices where sample == 1."""
        return sorted(k for k, v in self.sample.items() if v == 1)


class MaterialAnnealer:
    """Run simulated (quantum) annealing on a QUBO problem.

    Parameters
    ----------
    method : str
        ``'SQA'`` -> try OpenJij SQA first, fall back to NumPy SA.
        ``'SA'``  -> always use the pure-NumPy simulated annealing.
    num_reads : int
        Number of independent annealing runs.
    seed : int
        Random seed for reproducibility.
    """

    SUPPORTED_METHODS = ("SQA", "SA")

    def __init__(
        self,
        method: str = "SQA",
        num_reads: int = 100,
        seed: int = 42,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {self.SUPPORTED_METHODS}")
        self.method = method
        self.num_reads = num_reads
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anneal(self, qubo: dict[tuple[int, int], float]) -> AnnealingResult:
        """Run annealing on the given QUBO and return the best result."""
        if self.method == "SQA":
            return self._run_openjij_sqa(qubo)
        return self._run_numpy_sa(qubo)

    # ------------------------------------------------------------------
    # OpenJij SQA (optional dependency)
    # ------------------------------------------------------------------

    def _run_openjij_sqa(
        self, qubo: dict[tuple[int, int], float]
    ) -> AnnealingResult:
        """Attempt OpenJij Simulated Quantum Annealing; fall back to NumPy SA."""
        try:
            import openjij as oj  # type: ignore[import-untyped]

            sampler = oj.SQASampler()
            t0 = time.perf_counter()

            # OpenJij expects h, J or QUBO dict
            response = sampler.sample_qubo(qubo, seed=self.seed, num_reads=self.num_reads)
            elapsed = (time.perf_counter() - t0) * 1000.0

            best = response.first
            return AnnealingResult(
                sample=dict(best.sample),
                energy=float(best.energy),
                num_reads=self.num_reads,
                timing_ms=elapsed,
                method="OpenJij-SQA",
            )

        except ImportError:
            print("[MaterialAnnealer] OpenJij not available - falling back to NumPy SA")
            return self._run_numpy_sa(qubo)

        except Exception as exc:  # noqa: BLE001
            print(f"[MaterialAnnealer] OpenJij error ({exc}) - falling back to NumPy SA")
            return self._run_numpy_sa(qubo)

    # ------------------------------------------------------------------
    # Pure-NumPy Simulated Annealing (ALWAYS works)
    # ------------------------------------------------------------------

    def _run_numpy_sa(
        self,
        qubo: dict[tuple[int, int], float],
        *,
        t_init: float = 5.0,
        t_final: float = 0.001,
        sweeps: int = 1000,
    ) -> AnnealingResult:
        """Pure-NumPy simulated annealing - no external dependencies.

        Uses geometric cooling schedule with single-bit-flip Metropolis
        updates.
        """
        rng = np.random.default_rng(self.seed)

        # Determine problem size
        variables: set[int] = set()
        for (i, j) in qubo:
            variables.add(i)
            variables.add(j)
        n = max(variables) + 1 if variables else 0
        if n == 0:
            return AnnealingResult({}, 0.0, self.num_reads, 0.0, "NumPy-SA")

        # Build dense Q matrix for fast energy computation
        Q = np.zeros((n, n), dtype=np.float64)
        for (i, j), val in qubo.items():
            Q[i, j] += val
            if i != j:
                Q[j, i] += val  # symmetrise for delta-E calc

        def _energy(state: np.ndarray) -> float:
            return float(state @ Q @ state) / 2.0 + float(np.diag(Q) @ state) / 2.0

        def _energy_exact(state: np.ndarray) -> float:
            """Exact QUBO energy using the original upper-triangular dict."""
            e = 0.0
            for (i, j), val in qubo.items():
                e += val * state[i] * state[j]
            return e

        t0 = time.perf_counter()
        best_sample = np.zeros(n, dtype=np.int8)
        best_energy = float("inf")
        all_energies: list[float] = []
        cooling = (t_final / t_init) ** (1.0 / max(sweeps, 1))

        for _read in range(self.num_reads):
            state = rng.integers(0, 2, size=n, dtype=np.int8)
            current_e = _energy_exact(state)
            temp = t_init

            for _sweep in range(sweeps):
                for flip_idx in rng.permutation(n):
                    # Compute delta energy for flipping bit flip_idx
                    s_i = state[flip_idx]
                    delta = 0.0
                    if s_i == 0:
                        # Flipping 0 -> 1
                        delta = Q[flip_idx, flip_idx]
                        for j2 in range(n):
                            if j2 != flip_idx and state[j2]:
                                delta += Q[flip_idx, j2] + Q[j2, flip_idx]
                    else:
                        # Flipping 1 -> 0
                        delta = -Q[flip_idx, flip_idx]
                        for j2 in range(n):
                            if j2 != flip_idx and state[j2]:
                                delta -= Q[flip_idx, j2] + Q[j2, flip_idx]

                    # Metropolis acceptance
                    if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-30)):
                        state[flip_idx] = 1 - s_i
                        current_e += delta

                temp *= cooling

            final_e = _energy_exact(state)
            all_energies.append(final_e)
            if final_e < best_energy:
                best_energy = final_e
                best_sample = state.copy()

        elapsed = (time.perf_counter() - t0) * 1000.0

        sample_dict = {i: int(best_sample[i]) for i in range(n)}
        return AnnealingResult(
            sample=sample_dict,
            energy=best_energy,
            num_reads=self.num_reads,
            timing_ms=elapsed,
            method="NumPy-SA",
            all_energies=all_energies,
        )


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    from research.quantum_annealing.qubo import (
        DEMO_MATERIALS,
        QUBOFormulator,
        _build_demo_compatibility,
    )

    np.random.seed(42)
    materials = DEMO_MATERIALS
    compat = _build_demo_compatibility(len(materials))
    formulator = QUBOFormulator(materials, compat)
    Q = formulator.formulate(select_k=3, penalty_weight=8.0)

    print("=" * 65)
    print("  AETHER * Simulated Quantum Annealing Demo")
    print("=" * 65)

    # Always use NumPy SA to ensure the demo works
    annealer = MaterialAnnealer(method="SA", num_reads=50, seed=42)
    result = annealer.anneal(Q)

    print(f"\n  Method      : {result.method}")
    print(f"  Best energy : {result.energy:.4f}")
    print(f"  Reads       : {result.num_reads}")
    print(f"  Time        : {result.timing_ms:.1f} ms")
    print(f"  Selected    : {result.selected_indices()}")
    print()

    selected = formulator.decode_solution(result.sample)
    print("  * Optimal material combination:")
    for idx in selected:
        print(f"      -> {materials[idx]['name']}")

    if result.all_energies:
        energies = np.array(result.all_energies)
        print(f"\n  Energy stats across {len(energies)} reads:")
        print(f"    min={energies.min():.4f}  mean={energies.mean():.4f}"
              f"  std={energies.std():.4f}")

    # Try SQA (will fall back to SA if OpenJij not installed)
    print("\n--- Attempting SQA method ---")
    annealer_sqa = MaterialAnnealer(method="SQA", num_reads=30, seed=42)
    result_sqa = annealer_sqa.anneal(Q)
    print(f"  Method used : {result_sqa.method}")
    print(f"  Best energy : {result_sqa.energy:.4f}")
    print(f"  Selected    : {result_sqa.selected_indices()}")
    print()
