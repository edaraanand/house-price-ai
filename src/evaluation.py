from typing import Dict


def calculate_rmse_improvement(
    candidate_rmse: float,
    baseline_rmse: float,
) -> float:
    """
    Calculate relative RMSE improvement.

    Positive means the candidate is better.
    """

    if baseline_rmse <= 0:
        raise ValueError("Baseline RMSE must be greater than zero.")

    return (baseline_rmse - candidate_rmse) / baseline_rmse


def passes_promotion_gate(
    candidate_rmse: float,
    baseline_rmse: float,
    candidate_r2: float,
    min_rmse_improvement: float,
    min_r2: float,
) -> bool:
    """
    Determine whether a candidate model should be promoted.
    """

    improvement = calculate_rmse_improvement(
        candidate_rmse=candidate_rmse,
        baseline_rmse=baseline_rmse,
    )

    return (
        improvement >= min_rmse_improvement
        and candidate_r2 >= min_r2
    )
