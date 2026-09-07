"""
multi_train.py

Train one or multiple algorithms and log each model to MLflow.

Modes:

    python multi_train.py
        -> trains all configured algorithms

    main(
        algorithm="xgboost",
        param_overrides={...}
    )
        -> trains only XGBoost with the supplied overrides

MLflow design:

    One MLflow experiment
        |
        +-- xgboost-baseline
        +-- random_forest-baseline
        +-- hist_gradient_boosting-baseline
        |
        +-- tuning-xgboost-001
        +-- tuning-xgboost-002
        +-- ...
"""

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import sklearn
import xgboost as xgb
import yaml

from src.data import (
    load_california_housing,
    split_data,
)

from src.training import train_model


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
# PIPELINE OUTPUTS
# ==========================================================


def write_pipeline_outputs(
    run_id,
    model_version,
    metrics,
):
    """
    Write MLflow information for downstream
    CI/CD pipeline stages.
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

    (output_dir / "model_version").write_text(
        str(model_version)
    )

    (output_dir / "run_id").write_text(
        str(run_id)
    )

    (output_dir / "rmse").write_text(
        f"{metrics['rmse']:.6f}"
    )


# ==========================================================
# TRAIN ONE ALGORITHM
# ==========================================================


def train_one_algorithm(
    algorithm,
    configured_params,
    param_overrides,
    X_train,
    y_train,
    X_test,
    y_test,
    dataset_config,
    model_name,
    run_name,
    register_model,
    run_type,
):
    """
    Train exactly one algorithm.

    configured_params:
        Base parameters from config.yaml.

    param_overrides:
        Hyperparameter tuning overrides.

    Example:

        configured_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            ...
        }

        param_overrides = {
            "max_depth": 8,
            "learning_rate": 0.1,
        }

    Result:

        {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.1,
            ...
        }
    """

    # ------------------------------------------------------
    # Merge parameters
    # ------------------------------------------------------

    training_params = {
        **configured_params,
        **(param_overrides or {}),
    }

    # ------------------------------------------------------
    # Run name
    # ------------------------------------------------------

    if run_name is None:

        if run_type == "training":
            run_name = (
                f"{algorithm}-baseline"
            )

        else:
            run_name = (
                f"{run_type}-{algorithm}"
            )

    # ------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        result = train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            algorithm=algorithm,
            params=training_params,
        )

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        mlflow.log_params(
            training_params
        )

        mlflow.log_params(
            {
                "dataset": dataset_config[
                    "name"
                ],
                "dataset_version": (
                    "sklearn_builtin"
                ),
                "train_rows": len(
                    X_train
                ),
                "test_rows": len(
                    X_test
                ),
                "feature_count": (
                    X_train.shape[1]
                ),
                "random_state": (
                    dataset_config[
                        "random_state"
                    ]
                ),
                "python_version": (
                    sys.version.split(".")[0]
                    + "."
                    + sys.version.split(".")[1]
                ),
                "sklearn_version": (
                    sklearn.__version__
                ),
                "xgboost_version": (
                    xgb.__version__
                ),
            }
        )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        mlflow.log_metrics(
            result.metrics
        )

        # --------------------------------------------------
        # Tags
        # --------------------------------------------------

        mlflow.set_tags(
            {
                "algorithm": algorithm,
                "model_name": model_name,
                "dataset": dataset_config[
                    "name"
                ],
                "run_type": run_type,
                "lifecycle": "candidate",
            }
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=result.model,
            name="model",
        )

        model_uri = (
            f"runs:/{run.info.run_id}/model"
        )

        # --------------------------------------------------
        # Optional registration
        # --------------------------------------------------

        registered_model = None

        if register_model:

            registered_model = (
                mlflow.register_model(
                    model_uri=model_uri,
                    name=model_name,
                )
            )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        print(
            f"Algorithm:       {algorithm}"
        )

        print(
            f"Run ID:          {run.info.run_id}"
        )

        print(
            f"Run Name:        {run_name}"
        )

        print(
            f"Model Name:      {model_name}"
        )

        print(
            f"RMSE:            "
            f"{result.metrics['rmse']:.4f}"
        )

        print(
            f"MAE:             "
            f"{result.metrics['mae']:.4f}"
        )

        print(
            f"R2:              "
            f"{result.metrics['r2']:.4f}"
        )

        print(
            f"Model URI:       "
            f"{model_uri}"
        )

        if registered_model:

            print(
                f"Model Version:   "
                f"{registered_model.version}"
            )

        print("=" * 60)

        # --------------------------------------------------
        # Pipeline output
        # --------------------------------------------------

        if registered_model:

            write_pipeline_outputs(
                run_id=run.info.run_id,
                model_version=(
                    registered_model.version
                ),
                metrics=result.metrics,
            )

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "algorithm": algorithm,
            "metrics": result.metrics,
            "model_uri": model_uri,
            "model_version": (
                registered_model.version
                if registered_model
                else None
            ),
        }


# ==========================================================
# MAIN
# ==========================================================


def main(
    algorithm=None,
    param_overrides=None,
    run_name=None,
    register_model=False,
    run_type="training",
):
    """
    Train one or all algorithms.

    Examples
    --------

    Train all algorithms:

        main()

    Train XGBoost:

        main(
            algorithm="xgboost"
        )

    Hyperparameter tuning:

        main(
            algorithm="xgboost",
            param_overrides={
                "max_depth": 8,
                "learning_rate": 0.1,
                "n_estimators": 600,
            },
            register_model=False,
            run_type="hyperparameter_tuning",
        )
    """

    config = load_config()

    # ------------------------------------------------------
    # MLflow
    # ------------------------------------------------------

    tracking_uri = os.environ[
        "MLFLOW_TRACKING_URI"
    ]

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        config["model"]["name"],
    )

    model_name = os.getenv(
        "MLFLOW_MODEL_NAME",
        config["model"]["name"],
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        experiment_name
    )

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    X, y = load_california_housing()

    dataset_config = config[
        "dataset"
    ]

    X_train, X_test, y_train, y_test = (
        split_data(
            X,
            y,
            test_size=dataset_config[
                "test_size"
            ],
            random_state=dataset_config[
                "random_state"
            ],
        )
    )

    # ------------------------------------------------------
    # Algorithms
    # ------------------------------------------------------

    algorithms = config[
        "training"
    ]

    # ------------------------------------------------------
    # Single algorithm mode
    # ------------------------------------------------------

    if algorithm:

        algorithm = algorithm.lower()

        if algorithm not in algorithms:

            raise ValueError(
                f"Unknown algorithm: "
                f"{algorithm}. "
                f"Available algorithms: "
                f"{list(algorithms.keys())}"
            )

        print()
        print("=" * 60)
        print(
            f"TRAINING: {algorithm}"
        )
        print("=" * 60)

        return train_one_algorithm(
            algorithm=algorithm,
            configured_params=algorithms[
                algorithm
            ],
            param_overrides=param_overrides,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_config=dataset_config,
            model_name=model_name,
            run_name=run_name,
            register_model=register_model,
            run_type=run_type,
        )

    # ------------------------------------------------------
    # Multi algorithm mode
    # ------------------------------------------------------

    results = []

    for algorithm, configured_params in (
        algorithms.items()
    ):

        print()
        print("=" * 60)
        print(
            f"TRAINING: {algorithm}"
        )
        print("=" * 60)

        result = train_one_algorithm(
            algorithm=algorithm,
            configured_params=configured_params,
            param_overrides=None,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_config=dataset_config,
            model_name=model_name,
            run_name=f"{algorithm}-baseline",
            register_model=register_model,
            run_type=run_type,
        )

        results.append(result)

    # ------------------------------------------------------
    # Compare algorithms
    # ------------------------------------------------------

    sorted_results = sorted(
        results,
        key=lambda result: result[
            "metrics"
        ]["rmse"],
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for index, result in enumerate(
        sorted_results,
        start=1,
    ):

        print(
            f"{index}. "
            f"{result['algorithm']:<25} "
            f"RMSE="
            f"{result['metrics']['rmse']:.4f} "
            f"R2="
            f"{result['metrics']['r2']:.4f}"
        )

    # ------------------------------------------------------
    # Best baseline
    # ------------------------------------------------------

    best_result = sorted_results[0]

    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Algorithm:       "
        f"{best_result['algorithm']}"
    )

    print(
        f"Run ID:          "
        f"{best_result['run_id']}"
    )

    print(
        f"RMSE:            "
        f"{best_result['metrics']['rmse']:.4f}"
    )

    print(
        f"MAE:             "
        f"{best_result['metrics']['mae']:.4f}"
    )

    print(
        f"R2:              "
        f"{best_result['metrics']['r2']:.4f}"
    )

    print(
        f"Model URI:       "
        f"{best_result['model_uri']}"
    )

    print("=" * 70)

    return results


# ==========================================================
# SCRIPT ENTRY POINT
# ==========================================================


if __name__ == "__main__":

    main(
        register_model=False,
        run_type="training",
    )
