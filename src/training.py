from dataclasses import dataclass

import xgboost as xgb

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class TrainingResult:
    model: object
    metrics: dict


def train_xgboost(
    X_train,
    y_train,
    params,
):
    """
    Train an XGBoost regression model.
    """

    params = dict(params)

    model = xgb.XGBRegressor(
        **params
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_random_forest(
    X_train,
    y_train,
    params,
):
    """
    Train a Random Forest regression model.
    """

    params = dict(params)

    model = RandomForestRegressor(
        **params
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_hist_gradient_boosting(
    X_train,
    y_train,
    params,
):
    """
    Train a HistGradientBoosting regression model.
    """

    params = dict(params)

    model = HistGradientBoostingRegressor(
        **params
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    algorithm,
    params,
):
    """
    Train a model using the requested algorithm
    and evaluate it on the test set.

    Supported algorithms:

        xgboost
        random_forest
        hist_gradient_boosting
    """

    algorithm = algorithm.lower()

    # --------------------------------------------------
    # Select model
    # --------------------------------------------------

    if algorithm == "xgboost":

        model = train_xgboost(
            X_train=X_train,
            y_train=y_train,
            params=params,
        )

    elif algorithm == "random_forest":

        model = train_random_forest(
            X_train=X_train,
            y_train=y_train,
            params=params,
        )

    elif algorithm == "hist_gradient_boosting":

        model = train_hist_gradient_boosting(
            X_train=X_train,
            y_train=y_train,
            params=params,
        )

    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. "
            f"Supported algorithms are: "
            f"xgboost, random_forest, "
            f"hist_gradient_boosting"
        )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

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

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    return TrainingResult(
        model=model,
        metrics=metrics,
    )
