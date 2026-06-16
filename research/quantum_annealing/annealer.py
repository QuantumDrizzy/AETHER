"""
Material Annealer — thin consumer of the `anneal` spine
========================================================
The simulated-annealing / QUBO core lives in the shared **temp0r** spine
(`QuantumDrizzy/temp0r`, twin of Spectra). This module keeps AETHER's
material-facing API (`MaterialAnnealer`, `AnnealingResult`) but delegates the
actual optimisation to the spine — one owned, audited SA/QUBO core, not a copy
per lab. See docs/ADR-0002-extract-annealing-spine.md.

Install the spine (editable):  pip install -e ../temp0r

Run standalone:
    python -m research.quantum_annealing.annealer
"""

from __future__ import annotations

from dataclasses import dataclass

from temp0r import anneal as _spine_anneal


@dataclass
class AnnealingResult:
    """Result container from an annealing run (AETHER-facing shape)."""

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
    """Run (simulated quantum) annealing on a QUBO via the `anneal` spine.

    Parameters
    ----------
    method : str
        ``'SQA'`` -> spine ``backend='sqa'`` (OpenJij if present, else SA).
        ``'SA'``  -> spine ``backend='sa'`` (pure-NumPy simulated annealing).
    num_reads : int
        Number of independent annealing runs.
    seed : int
        Random seed for reproducibility.
    """

    SUPPORTED_METHODS = ("SQA", "SA")

    def __init__(self, method: str = "SQA", num_reads: int = 100, seed: int = 42) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {self.SUPPORTED_METHODS}")
        self.method = method
        self.num_reads = num_reads
        self.seed = seed

    def anneal(self, qubo: dict[tuple[int, int], float]) -> AnnealingResult:
        """Run annealing on the given QUBO and return the best result."""
        backend = "sqa" if self.method == "SQA" else "sa"
        r = _spine_anneal(qubo, backend=backend, num_reads=self.num_reads, seed=self.seed)
        return AnnealingResult(
            sample=r.sample,
            energy=r.energy,
            num_reads=r.num_reads,
            timing_ms=r.timing_ms,
            method=r.method,
            all_energies=r.all_energies,
        )


# ======================================================================
# __main__ demo
# ======================================================================

if __name__ == "__main__":
    import numpy as np

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
    print("  AETHER * Simulated Quantum Annealing Demo (via anneal spine)")
    print("=" * 65)

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
