"""
train.py
   │
   ├── load config
   ├── start MLflow run
   ├── call data.py
   ├── call features.py
   ├── call training.py
   ├── log metrics
   ├── log artifact
   └── register model
"""
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
import sklearn
import xgboost as xgb

from src.data import load_california_housing, split_data
from src.training import train_model


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

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

    X, y = load_california_housing()

    dataset_config = config["dataset"]

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=dataset_config["test_size"],
        random_state=dataset_config["random_state"],
    )

    training_params = config["training"]

    with mlflow.start_run() as run:

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
        # Register model
        # --------------------------------------------------

        model_uri = f"runs:/{run.info.run_id}/model"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Run ID:          {run.info.run_id}")
        print(f"Model Name:      {model_name}")
        print(f"Model Version:   {registered_model.version}")
        print(f"RMSE:            {result.metrics['rmse']:.4f}")
        print(f"MAE:             {result.metrics['mae']:.4f}")
        print(f"R2:              {result.metrics['r2']:.4f}")
        print(f"Model URI:       {model_uri}")
        print("=" * 60)


if __name__ == "__main__":
    main()
