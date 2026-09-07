from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .model import create_model
from .schemas import TrainingResult


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    params: Dict[str, object],
) -> TrainingResult:
    """
    Train the model and calculate evaluation metrics.
    """

    model = create_model(params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    return TrainingResult(
        model=model,
        metrics=metrics,
        params=params,
    )
