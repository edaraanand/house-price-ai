from typing import Dict

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model(params: Dict[str, object]) -> Pipeline:
    """
    Create the complete preprocessing + model pipeline.

    The returned object is the artifact that will eventually
    be served by BentoML.
    """

    model = xgb.XGBRegressor(**params)

    pipeline = Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("model", model),
        ]
    )

    return pipeline
