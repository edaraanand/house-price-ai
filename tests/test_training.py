import pandas as pd

from src.training import train_model


def test_training_produces_model():
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
        params={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
        },
    )

    assert result.model is not None

    assert "rmse" in result.metrics
    assert "mae" in result.metrics
    assert "r2" in result.metrics
