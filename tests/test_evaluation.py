from src.evaluation import (
    calculate_rmse_improvement,
    passes_promotion_gate,
)


def test_rmse_improvement():
    improvement = calculate_rmse_improvement(
        candidate_rmse=1.40,
        baseline_rmse=1.50,
    )

    assert improvement > 0.06


def test_candidate_passes_gate():
    passed = passes_promotion_gate(
        candidate_rmse=1.40,
        baseline_rmse=1.50,
        candidate_r2=0.60,
        min_rmse_improvement=0.05,
        min_r2=0.50,
    )

    assert passed is True


def test_candidate_fails_rmse_gate():
    passed = passes_promotion_gate(
        candidate_rmse=1.45,
        baseline_rmse=1.50,
        candidate_r2=0.60,
        min_rmse_improvement=0.05,
        min_r2=0.50,
    )

    assert passed is False


def test_candidate_fails_r2_gate():
    passed = passes_promotion_gate(
        candidate_rmse=1.30,
        baseline_rmse=1.50,
        candidate_r2=0.40,
        min_rmse_improvement=0.05,
        min_r2=0.50,
    )

    assert passed is False


def test_worse_model_fails():
    passed = passes_promotion_gate(
        candidate_rmse=1.70,
        baseline_rmse=1.50,
        candidate_r2=0.60,
        min_rmse_improvement=0.05,
        min_r2=0.50,
    )

    assert passed is False
