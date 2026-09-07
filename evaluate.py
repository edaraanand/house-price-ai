import argparse
import os
from pathlib import Path

import mlflow
import yaml

from src.data import load_california_housing, split_data
from src.evaluation import (
    calculate_rmse_improvement,
    passes_promotion_gate,
)


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a registered MLflow model."
    )

    parser.add_argument(
        "--candidate-version",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--baseline-version",
        type=int,
        required=True,
    )

    return parser.parse_args()


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def main():
    args = parse_args()
    config = load_config()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    mlflow.set_tracking_uri(tracking_uri)

    model_name = os.getenv(
        "MLFLOW_MODEL_NAME",
        config["model"]["name"],
    )

    model_base_uri = f"models:/{model_name}"

    candidate_uri = (
        f"{model_base_uri}/{args.candidate_version}"
    )

    baseline_uri = (
        f"{model_base_uri}/{args.baseline_version}"
    )

    print("Loading candidate model...")
    candidate_model = mlflow.pyfunc.load_model(
        candidate_uri
    )

    print("Loading baseline model...")
    baseline_model = mlflow.pyfunc.load_model(
        baseline_uri
    )

    X, y = load_california_housing()

    dataset_config = config["dataset"]

    _, X_test, _, y_test = split_data(
        X,
        y,
        test_size=dataset_config["test_size"],
        random_state=dataset_config["random_state"],
    )

    candidate_metrics = evaluate_model(
        candidate_model,
        X_test,
        y_test,
    )

    baseline_metrics = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    improvement = calculate_rmse_improvement(
        candidate_rmse=candidate_metrics["rmse"],
        baseline_rmse=baseline_metrics["rmse"],
    )

    evaluation_config = config["evaluation"]

    passed = passes_promotion_gate(
        candidate_rmse=candidate_metrics["rmse"],
        baseline_rmse=baseline_metrics["rmse"],
        candidate_r2=candidate_metrics["r2"],
        min_rmse_improvement=evaluation_config[
            "min_rmse_improvement"
        ],
        min_r2=evaluation_config["min_r2"],
    )

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        f"Candidate version: {args.candidate_version}"
    )

    print(
        f"Baseline version:  {args.baseline_version}"
    )

    print()
    print("Candidate:")
    print(
        f"  RMSE: {candidate_metrics['rmse']:.4f}"
    )
    print(
        f"  MAE:  {candidate_metrics['mae']:.4f}"
    )
    print(
        f"  R2:   {candidate_metrics['r2']:.4f}"
    )

    print()
    print("Baseline:")
    print(
        f"  RMSE: {baseline_metrics['rmse']:.4f}"
    )
    print(
        f"  MAE:  {baseline_metrics['mae']:.4f}"
    )
    print(
        f"  R2:   {baseline_metrics['r2']:.4f}"
    )

    print()
    print(
        f"RMSE improvement: {improvement * 100:.2f}%"
    )

    print()

    if passed:
        print("✓ PROMOTION GATE PASSED")
        print("Candidate is eligible for promotion.")
    else:
        print("✗ PROMOTION GATE FAILED")
        print("Candidate must not be promoted.")

    print("=" * 60)


if __name__ == "__main__":
    main()
