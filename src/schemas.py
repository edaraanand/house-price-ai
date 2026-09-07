from dataclasses import dataclass
from typing import Dict


@dataclass
class TrainingResult:
    model: object
    metrics: Dict[str, float]
    params: Dict[str, object]


@dataclass
class EvaluationResult:
    candidate_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    rmse_improvement: float
    passed: bool
