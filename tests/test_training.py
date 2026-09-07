import pandas as pd
import pytest

from src.training import train_model


@pytest.mark.parametrize(
    "algorithm,params",
    [
        (
            "xgboost",
            {
                "n_estimators": 10,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "objective": "reg:squarederror",
                "random_state": 42,
            },
        ),
        (
            "random_forest",
            {
                "n_estimators": 10,
                "max_depth": 2,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 10,
                "learning_rate": 0.1,
                "max_leaf_nodes": 15,
                "max_depth": None,
                "l2_regularization": 0.0,
                "random_state": 42,
            },
        ),
    ],
)
def test_training_produces_model(
    algorithm,
    params,
):
    X = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [2, 4, 6, 8, 10, 12],
        }
    )

    y = pd.Series(
        [1, 2, 3, 4, 5, 6]
    )

    result = train_model(
        X_train=X.iloc[:4],
        y_train=y.iloc[:4],
        X_test=X.iloc[4:],
        y_test=y.iloc[4:],
        params=params,
        algorithm=algorithm,
    )

    assert result.model is not None

    assert "rmse" in result.metrics
    assert "mae" in result.metrics
    assert "r2" in result.metrics

    assert result.metrics["rmse"] >= 0
    assert result.metrics["mae"] >= 0
