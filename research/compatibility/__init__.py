"""Material compatibility scoring - multi-metric physical and subjective analysis."""

from research.compatibility.metrics import MetricCalculator
from research.compatibility.scorer import CompatibilityScorer, CompatibilityReport
from research.compatibility.subjective import SubjectiveHandler

__all__ = [
    "MetricCalculator",
    "CompatibilityScorer",
    "CompatibilityReport",
    "SubjectiveHandler",
]
