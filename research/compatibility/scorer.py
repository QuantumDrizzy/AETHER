from dataclasses import dataclass
import numpy as np
from research.compatibility.metrics import MetricCalculator

@dataclass
class CompatibilityReport:
    overall_score: float
    dimensions: dict          # only the dimensions actually computed
    synergies: list
    conflicts: list
    recommendation: str
    unimplemented: list       # dimensions skipped because they are not implemented yet

class CompatibilityScorer:
    def __init__(self, weights: dict = None):
        if weights is None:
            self.weights = {
                "crystallographic": 1.0,
                "thermal": 1.2,
                "electromagnetic": 1.5,
                "resonant": 1.3,
                "piezoelectric": 1.4,
                "mechanical": 0.8,
                "bandgap": 1.0
            }
        else:
            self.weights = weights

    def score(self, mat_a: dict, mat_b: dict) -> CompatibilityReport:
        raw = {
            "crystallographic": MetricCalculator.lattice_mismatch(mat_a, mat_b),
            "thermal": MetricCalculator.thermal_expansion_mismatch(mat_a, mat_b),
            "electromagnetic": MetricCalculator.impedance_matching(mat_a, mat_b),
            "resonant": MetricCalculator.resonance_overlap(mat_a, mat_b),
            "piezoelectric": MetricCalculator.piezoelectric_coupling(mat_a, mat_b),
            "mechanical": MetricCalculator.mechanical_compatibility(mat_a, mat_b),
            "bandgap": MetricCalculator.bandgap_alignment(mat_a, mat_b),
        }
        # Aggregate ONLY the implemented dimensions; never invent a neutral value.
        scores = {k: v for k, v in raw.items() if v is not None}
        unimplemented = [k for k, v in raw.items() if v is None]

        total_weight = sum(self.weights.get(k, 0.0) for k in scores)
        weighted_sum = sum(scores[k] * self.weights.get(k, 0.0) for k in scores)
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        recommendation = "Neutral"
        if overall > 0.8: recommendation = "HighlyCompatible"
        elif overall > 0.6: recommendation = "Compatible"
        elif overall < 0.2: recommendation = "HighlyIncompatible"
        elif overall < 0.4: recommendation = "Incompatible"

        return CompatibilityReport(overall, scores, [], [], recommendation, unimplemented)

    def score_matrix(self, materials: list[dict]) -> np.ndarray:
        n = len(materials)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = self.score(materials[i], materials[j]).overall_score
        return matrix

    def find_best_pairs(self, materials: list[dict], top_k: int = 5) -> list:
        """Return the top_k most compatible (name_a, name_b, score) pairs."""
        scored = []
        for i in range(len(materials)):
            for j in range(i + 1, len(materials)):
                s = self.score(materials[i], materials[j]).overall_score
                scored.append((materials[i].get("name", i), materials[j].get("name", j), s))
        scored.sort(key=lambda t: t[2], reverse=True)
        return scored[:top_k]

    def find_best_combination(self, materials: list[dict], k: int = 3) -> list:
        """Exhaustive search for the k-material set with the best mean pairwise
        compatibility. O(C(n, k)) — intended for small candidate sets."""
        from itertools import combinations

        best_names, best_score = [], -1.0
        for combo in combinations(range(len(materials)), k):
            pair_scores = [
                self.score(materials[a], materials[b]).overall_score
                for a, b in combinations(combo, 2)
            ]
            avg = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
            if avg > best_score:
                best_score = avg
                best_names = [materials[i].get("name", i) for i in combo]
        return best_names

if __name__ == "__main__":
    scorer = CompatibilityScorer()
    mat_a = {"name": "Quartz", "crystal": {"lattice_params": {"a": 4.914}}, "physical": {"thermal_expansion": 13.2e-6}}
    mat_b = {"name": "BaTiO3", "crystal": {"lattice_params": {"a": 3.992}}, "physical": {"thermal_expansion": 14.0e-6}}
    report = scorer.score(mat_a, mat_b)
    print(f"Overall compatibility: {report.overall_score:.2f} ({report.recommendation})")
