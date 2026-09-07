import pandas as pd

from src.features import build_preprocessor


def test_preprocessor_can_fit():
    X = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
        }
    )

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape == X.shape
