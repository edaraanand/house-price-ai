import argparse
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import yaml


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote an MLflow model version."
    )

    parser.add_argument(
        "--version",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--stage",
        choices=["Staging", "Production"],
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    mlflow.set_tracking_uri(tracking_uri)

    model_name = os.getenv(
        "MLFLOW_MODEL_NAME",
        config["model"]["name"],
    )

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    print(
        f"Promoting {model_name} v{args.version} "
        f"to {args.stage}..."
    )

    client.transition_model_version_stage(
        name=model_name,
        version=args.version,
        stage=args.stage,
        archive_existing_versions=(
            args.stage == "Production"
        ),
    )

    print(
        f"✓ Model {model_name} v{args.version} "
        f"is now {args.stage}."
    )

    # The serving side (serving/service.py) loads the model by alias
    # (models:/<name>@<alias>), not by classic registry stage — so the
    # stage transition above is for bookkeeping/visibility only. Setting
    # the matching alias here is what actually makes the promotion take
    # effect for anything reading models:/<name>@<alias>.
    alias = args.stage.lower()

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(args.version),
    )

    print(
        f"✓ Alias @{alias} -> version {args.version}"
    )


if __name__ == "__main__":
    main()
