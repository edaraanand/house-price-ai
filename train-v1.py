"""
train.py

Single-model training entry point.

Flow:
    load config
        ↓
    apply parameter overrides
        ↓
    start MLflow run
        ↓
    load dataset
        ↓
    split dataset
        ↓
    train XGBoost
        ↓
    log parameters
        ↓
    log metrics
        ↓
    log model artifact
        ↓
    optionally register model
"""

"""
train.py

Train a single XGBoost model and log it to MLflow.

This module can be used by:
    - normal training
    - hyperparameter tuning

Registration is optional. During tuning, models are logged but not
registered. tune.py selects the best run and registers only the winner.
"""

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import sklearn
import xgboost as xgb
import yaml

from src.data import load_california_housing, split_data
from src.training_v1 import train_model


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config-v1.yaml"

PIPELINE_OUTPUT_DIR = os.getenv("PIPELINE_OUTPUT_DIR")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_param_overrides(config, overrides):
    config = dict(config)

    config["training"] = {
        **config["training"],
        **overrides,
    }

    return config


def write_pipeline_outputs(run_id, model_version, metrics):
    if not PIPELINE_OUTPUT_DIR:
        return

    output_dir = Path(PIPELINE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "model_version").write_text(
        str(model_version)
    )

    (output_dir / "run_id").write_text(
        str(run_id)
    )

    (output_dir / "rmse").write_text(
        f"{metrics['rmse']:.6f}"
    )


def main(
    param_overrides=None,
    run_name=None,
    register_model=True,
    run_type="training",
):
    config = load_config()

    if param_overrides:
        config = apply_param_overrides(
            config,
            param_overrides,
        )

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        config["model"]["name"],
    )

    model_name = os.getenv(
        "MLFLOW_MODEL_NAME",
        config["model"]["name"],
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    X, y = load_california_housing()

    dataset_config = config["dataset"]

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=dataset_config["test_size"],
        random_state=dataset_config["random_state"],
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    training_params = config["training"]

    if run_name is None:
        run_name = "xgb-" + "-".join(
            f"{key}={value}"
            for key, value in training_params.items()
        )

    with mlflow.start_run(run_name=run_name) as run:

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        result = train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params=training_params,
        )

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        mlflow.log_params(training_params)

        mlflow.log_params(
            {
                "dataset": dataset_config["name"],
                "dataset_version": "sklearn_builtin",
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": X_train.shape[1],
                "random_state": dataset_config["random_state"],
                "python_version": sys.version.split()[0],
                "xgboost_version": xgb.__version__,
                "sklearn_version": sklearn.__version__,
            }
        )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        mlflow.log_metrics(result.metrics)

        # --------------------------------------------------
        # Tags
        # --------------------------------------------------

        mlflow.set_tags(
            {
                "model_name": model_name,
                "dataset": dataset_config["name"],
                "run_type": run_type,
                "lifecycle": "candidate",
            }
        )

        # --------------------------------------------------
        # Model artifact
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=result.model,
            name="model",
        )

        # --------------------------------------------------
        # Optional registration
        # --------------------------------------------------

        registered_model = None

        if register_model:
            model_uri = (
                f"runs:/{run.info.run_id}/model"
            )

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        print(f"Run ID:          {run.info.run_id}")
        print(f"Run Name:        {run_name}")
        print(f"Model Name:      {model_name}")

        if registered_model:
            print(
                f"Model Version:   "
                f"{registered_model.version}"
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
            f"runs:/{run.info.run_id}/model"
        )

        print("=" * 60)

        if registered_model:
            write_pipeline_outputs(
                run_id=run.info.run_id,
                model_version=registered_model.version,
                metrics=result.metrics,
            )

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "metrics": result.metrics,
            "model_uri": (
                f"runs:/{run.info.run_id}/model"
            ),
            "model_version": (
                registered_model.version
                if registered_model
                else None
            ),
        }


if __name__ == "__main__":
    main()
