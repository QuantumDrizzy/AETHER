from dataclasses import dataclass
import numpy as np
from research.compatibility.metrics import MetricCalculator

@dataclass
class CompatibilityReport:
    overall_score: float
    dimensions: dict
    synergies: list
    conflicts: list
    recommendation: str

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
        scores = {
            "crystallographic": MetricCalculator.lattice_mismatch(mat_a, mat_b),
            "thermal": MetricCalculator.thermal_expansion_mismatch(mat_a, mat_b),
            "electromagnetic": MetricCalculator.impedance_matching(mat_a, mat_b),
            "resonant": MetricCalculator.resonance_overlap(mat_a, mat_b),
            "piezoelectric": MetricCalculator.piezoelectric_coupling(mat_a, mat_b),
            "mechanical": MetricCalculator.mechanical_compatibility(mat_a, mat_b),
            "bandgap": MetricCalculator.bandgap_alignment(mat_a, mat_b),
        }
        
        weighted_sum = sum(scores[k] * self.weights[k] for k in scores)
        total_weight = sum(self.weights.values())
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        recommendation = "Neutral"
        if overall > 0.8: recommendation = "HighlyCompatible"
        elif overall > 0.6: recommendation = "Compatible"
        elif overall < 0.2: recommendation = "HighlyIncompatible"
        elif overall < 0.4: recommendation = "Incompatible"

        return CompatibilityReport(overall, scores, [], [], recommendation)

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
        return []

    def find_best_combination(self, materials: list[dict], k: int = 3) -> list:
        return []

if __name__ == "__main__":
    scorer = CompatibilityScorer()
    mat_a = {"name": "Quartz", "crystal": {"lattice_params": {"a": 4.914}}, "physical": {"thermal_expansion": 13.2e-6}}
    mat_b = {"name": "BaTiO3", "crystal": {"lattice_params": {"a": 3.992}}, "physical": {"thermal_expansion": 14.0e-6}}
    report = scorer.score(mat_a, mat_b)
    print(f"Overall compatibility: {report.overall_score:.2f} ({report.recommendation})")
