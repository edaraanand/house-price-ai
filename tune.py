"""
tune.py

Run algorithm-specific hyperparameter searches in parallel,
compare all MLflow runs, and register only the overall
winning model.

Flow:

    load config
        ↓
    generate parameter combinations
        ↓
    create tuning jobs
        ↓
    run jobs in parallel
        ↓
    XGBoost / Random Forest / HistGradientBoosting
        ↓
    collect all MLflow results
        ↓
    compare all runs
        ↓
    select lowest RMSE
        ↓
    register winning model
        ↓
    write model version for pipeline

Example:

    3 algorithms
        ×
    N hyperparameter combinations
        =
    all jobs executed concurrently

Concurrency is controlled by:

    tuning:
      max_workers: 4
"""

import os
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from itertools import product
from pathlib import Path

import mlflow
import yaml

from multi_train import main


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"

PIPELINE_OUTPUT_DIR = os.getenv(
    "PIPELINE_OUTPUT_DIR"
)


# ==========================================================
# CONFIG
# ==========================================================


def load_config():
    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


# ==========================================================
# HYPERPARAMETER GENERATION
# ==========================================================


def generate_params(param_grid):
    """
    Generate every combination from a parameter grid.

    Example:

        {
            "max_depth": [4, 6],
            "learning_rate": [0.01, 0.05],
        }

    produces:

        {
            "max_depth": 4,
            "learning_rate": 0.01,
        }

        {
            "max_depth": 4,
            "learning_rate": 0.05,
        }

        {
            "max_depth": 6,
            "learning_rate": 0.01,
        }

        {
            "max_depth": 6,
            "learning_rate": 0.05,
        }
    """

    keys = list(param_grid.keys())

    values = [
        param_grid[key]
        for key in keys
    ]

    for combination in product(*values):
        yield dict(
            zip(keys, combination)
        )


# ==========================================================
# WORKER
# ==========================================================


def run_tuning_job(job):
    """
    Execute one hyperparameter tuning job.

    This function runs inside a separate process.

    Important:
        MLflow runs are created independently inside
        each process. No active MLflow run is shared
        between processes.
    """

    algorithm = job["algorithm"]
    params = job["params"]
    run_name = job["run_name"]

    return main(
        algorithm=algorithm,
        param_overrides=params,
        run_name=run_name,
        register_model=False,
        run_type="hyperparameter_tuning",
    )


# ==========================================================
# PIPELINE OUTPUTS
# ==========================================================


def write_pipeline_outputs(
    run_id,
    model_version,
    metrics,
):
    """
    Write the winning model information for CI/CD
    or downstream pipeline steps.
    """

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


# ==========================================================
# MAIN TUNING
# ==========================================================


def main_tuning():

    config = load_config()

    # ==================================================
    # MLFLOW CONFIGURATION
    # ==================================================

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

    # ==================================================
    # TRAINING CONFIGURATION
    # ==================================================

    training_config = config.get(
        "training",
        {},
    )

    if not training_config:
        raise ValueError(
            "No 'training' section found "
            "in config.yaml"
        )

    # ==================================================
    # TUNING CONFIGURATION
    # ==================================================

    tuning_config = config.get(
        "tuning",
        {},
    )

    if not tuning_config:
        raise ValueError(
            "No 'tuning' section found "
            "in config.yaml"
        )

    # max_workers controls how many training processes
    # run simultaneously.
    #
    # Example:
    #
    # tuning:
    #   max_workers: 4

    max_workers = int(
        os.getenv(
            "TUNING_MAX_WORKERS",
            tuning_config.get("max_workers", 4),
        )
    )

    # ==================================================
    # BUILD ALL JOBS
    # ==================================================

    jobs = []

    print()
    print("=" * 70)
    print("BUILDING HYPERPARAMETER SEARCH")
    print("=" * 70)

    for algorithm, algorithm_tuning in (
        tuning_config.items()
    ):

        # max_workers is configuration for the tuner,
        # not an algorithm.
        if algorithm == "max_workers":
            continue

        # Make sure this algorithm exists in training.
        if algorithm not in training_config:
            raise ValueError(
                f"Tuning configured for "
                f"'{algorithm}', but no "
                f"training configuration "
                f"exists for it."
            )

        parameter_sets = list(
            generate_params(
                algorithm_tuning
            )
        )

        total_experiments = len(
            parameter_sets
        )

        print(
            f"{algorithm:<25} "
            f"{total_experiments} combinations"
        )

        # --------------------------------------------------
        # Create jobs
        # --------------------------------------------------

        for i, params in enumerate(
            parameter_sets,
            start=1,
        ):

            run_name = (
                f"tuning-"
                f"{algorithm}-"
                f"{i:03d}"
            )

            jobs.append(
                {
                    "algorithm": algorithm,
                    "params": params,
                    "run_name": run_name,
                }
            )

    total_jobs = len(jobs)

    if total_jobs == 0:
        raise RuntimeError(
            "No hyperparameter tuning jobs "
            "were generated."
        )

    print()
    print(
        f"TOTAL JOBS:    {total_jobs}"
    )

    print(
        f"MAX WORKERS:   {max_workers}"
    )

    print("=" * 70)

    # ==================================================
    # PARALLEL EXECUTION
    # ==================================================

    results = []

    completed = 0
    failed = 0

    print()
    print("=" * 70)
    print("STARTING PARALLEL HYPERPARAMETER SEARCH")
    print("=" * 70)

    # --------------------------------------------------
    # ProcessPoolExecutor
    # --------------------------------------------------
    #
    # Each worker is a separate Python process.
    #
    # This is preferable to threads for CPU-heavy
    # machine learning training.
    #
    # max_workers prevents all jobs from starting
    # simultaneously.

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_to_job = {
            executor.submit(
                run_tuning_job,
                job,
            ): job
            for job in jobs
        }

        # --------------------------------------------------
        # Collect results as they finish
        # --------------------------------------------------

        for future in as_completed(
            future_to_job
        ):

            job = future_to_job[
                future
            ]

            algorithm = job[
                "algorithm"
            ]

            run_name = job[
                "run_name"
            ]

            try:

                result = future.result()

                results.append(
                    result
                )

                completed += 1

                print()
                print(
                    "-" * 70
                )

                print(
                    f"COMPLETED "
                    f"{completed + failed}/{total_jobs}"
                )

                print(
                    f"Algorithm: {algorithm}"
                )

                print(
                    f"Run:       {run_name}"
                )

                print(
                    f"Run ID:    "
                    f"{result['run_id']}"
                )

                print(
                    f"RMSE:      "
                    f"{result['metrics']['rmse']:.4f}"
                )

                print(
                    f"MAE:       "
                    f"{result['metrics']['mae']:.4f}"
                )

                print(
                    f"R2:        "
                    f"{result['metrics']['r2']:.4f}"
                )

                print(
                    f"Progress:  "
                    f"{completed + failed}/"
                    f"{total_jobs}"
                )

            except Exception as exc:

                failed += 1

                print()
                print(
                    "-" * 70
                )

                print(
                    f"FAILED "
                    f"{completed + failed}/{total_jobs}"
                )

                print(
                    f"Algorithm: {algorithm}"
                )

                print(
                    f"Run:       {run_name}"
                )

                print(
                    f"Error:     {exc}"
                )

    # ==================================================
    # VALIDATE RESULTS
    # ==================================================

    print()
    print("=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)

    print(
        f"Successful: {completed}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Total:      {total_jobs}"
    )

    print("=" * 70)

    if not results:
        raise RuntimeError(
            "All hyperparameter tuning jobs failed."
        )

    # ==================================================
    # SELECT BEST MODEL
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

    best_algorithm = best_result[
        "algorithm"
    ]

    # ==================================================
    # FINAL MODEL COMPARISON
    # ==================================================

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    sorted_results = sorted(
        results,
        key=lambda result: result[
            "metrics"
        ]["rmse"],
    )

    for index, result in enumerate(
        sorted_results,
        start=1,
    ):

        algorithm = result[
            "algorithm"
        ]

        metrics = result[
            "metrics"
        ]

        print(
            f"{index:>3}. "
            f"{algorithm:<25} "
            f"RMSE={metrics['rmse']:.4f} "
            f"MAE={metrics['mae']:.4f} "
            f"R2={metrics['r2']:.4f}"
        )

    # ==================================================
    # BEST MODEL
    # ==================================================

    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Algorithm: {best_algorithm}"
    )

    print(
        f"Run ID:    {best_run_id}"
    )

    print(
        f"Model URI: {best_model_uri}"
    )

    print(
        f"RMSE:      "
        f"{best_metrics['rmse']:.4f}"
    )

    print(
        f"MAE:       "
        f"{best_metrics['mae']:.4f}"
    )

    print(
        f"R2:        "
        f"{best_metrics['r2']:.4f}"
    )

    print("=" * 70)

    # ==================================================
    # REGISTER WINNER ONLY
    # ==================================================

    print()
    print(
        "Registering winning model..."
    )

    registered_model = (
        mlflow.register_model(
            model_uri=best_model_uri,
            name=model_name,
        )
    )

    model_version = (
        registered_model.version
    )

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

    # ==================================================
    # RETURN
    # ==================================================

    return {
        "run_id": best_run_id,
        "algorithm": best_algorithm,
        "model_version": model_version,
        "metrics": best_metrics,
        "model_uri": best_model_uri,
        "successful_jobs": completed,
        "failed_jobs": failed,
        "total_jobs": total_jobs,
    }


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main_tuning()
