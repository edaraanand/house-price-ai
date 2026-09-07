"""
tune.py

Run hyperparameter search, select the best MLflow run,
and register only the winning model.

Flow:

    generate 18 parameter combinations
        ↓
    train 18 MLflow runs
        ↓
    select lowest RMSE
        ↓
    register winning model
        ↓
    write model version for pipeline
"""
"""
tune.py

Run 18 hyperparameter combinations, select the best run,
register only the winning model, and expose the registered
version to the pipeline.
"""

import os
from itertools import product
from pathlib import Path

import mlflow
import yaml

from train import main


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

PIPELINE_OUTPUT_DIR = os.getenv(
    "PIPELINE_OUTPUT_DIR"
)


PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [300, 600],
}


def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def generate_params():
    keys = PARAM_GRID.keys()

    for values in product(*PARAM_GRID.values()):
        yield dict(zip(keys, values))


def write_pipeline_outputs(
    run_id,
    model_version,
    metrics,
):
    if not PIPELINE_OUTPUT_DIR:
        return

    output_dir = Path(
        PIPELINE_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "run_id").write_text(
        str(run_id)
    )

    (output_dir / "model_version").write_text(
        str(model_version)
    )

    (output_dir / "rmse").write_text(
        f"{metrics['rmse']:.6f}"
    )


def main_tuning():

    config = load_config()

    tracking_uri = os.environ[
        "MLFLOW_TRACKING_URI"
    ]

    mlflow.set_tracking_uri(
        tracking_uri
    )

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        config["model"]["name"],
    )

    model_name = os.getenv(
        "MLFLOW_MODEL_NAME",
        config["model"]["name"],
    )

    mlflow.set_experiment(
        experiment_name
    )

    parameter_sets = list(
        generate_params()
    )

    total_experiments = len(
        parameter_sets
    )

    results = []

    # ==================================================
    # HYPERPARAMETER SEARCH
    # ==================================================

    for i, params in enumerate(
        parameter_sets,
        start=1,
    ):

        run_name = (
            f"tuning-{i:02d}"
            f"-depth={params['max_depth']}"
            f"-lr={params['learning_rate']}"
            f"-estimators={params['n_estimators']}"
        )

        print()
        print("=" * 60)
        print(
            f"EXPERIMENT "
            f"{i}/{total_experiments}"
        )
        print("=" * 60)

        print(f"Run name:   {run_name}")
        print(f"Parameters: {params}")

        result = main(
            param_overrides=params,
            run_name=run_name,
            register_model=False,
            run_type="hyperparameter_tuning",
        )

        results.append(result)

        print(
            f"Run ID: {result['run_id']}"
        )

        print(
            f"RMSE:   "
            f"{result['metrics']['rmse']:.4f}"
        )

        print(
            f"MAE:    "
            f"{result['metrics']['mae']:.4f}"
        )

        print(
            f"R2:     "
            f"{result['metrics']['r2']:.4f}"
        )

    # ==================================================
    # SELECT BEST
    # ==================================================

    best_result = min(
        results,
        key=lambda result: result[
            "metrics"
        ]["rmse"],
    )

    best_run_id = best_result[
        "run_id"
    ]

    best_model_uri = best_result[
        "model_uri"
    ]

    best_metrics = best_result[
        "metrics"
    ]

    print()
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Run ID: {best_run_id}"
    )

    print(
        f"Model URI: {best_model_uri}"
    )

    print(
        f"RMSE: {best_metrics['rmse']:.4f}"
    )

    print(
        f"MAE:  {best_metrics['mae']:.4f}"
    )

    print(
        f"R2:   {best_metrics['r2']:.4f}"
    )

    print("=" * 60)

    # ==================================================
    # REGISTER WINNER
    # ==================================================

    print()
    print("Registering winning model...")

    registered_model = mlflow.register_model(
        model_uri=best_model_uri,
        name=model_name,
    )

    model_version = registered_model.version

    print(
        f"Registered model: "
        f"{model_name}"
    )

    print(
        f"Model version: "
        f"{model_version}"
    )

    # ==================================================
    # PIPELINE OUTPUTS
    # ==================================================

    write_pipeline_outputs(
        run_id=best_run_id,
        model_version=model_version,
        metrics=best_metrics,
    )

    return {
        "run_id": best_run_id,
        "model_version": model_version,
        "metrics": best_metrics,
        "model_uri": best_model_uri,
    }


if __name__ == "__main__":
    main_tuning()
