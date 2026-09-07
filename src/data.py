import os
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


SKLEARN_DATA_HOME = os.getenv(
    "SKLEARN_DATA_HOME",
    "/home/appuser/scikit_learn_data",
)


def load_california_housing() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset.

    The dataset is expected to already exist in the
    scikit-learn cache. The pipeline Docker image
    populates this cache during image build.

    Returns:
        X: feature dataframe
        y: target series
    """
    data = fetch_california_housing(
        data_home=SKLEARN_DATA_HOME,
        as_frame=True,
        download_if_missing=False,
    )

    X = data.data
    y = data.target

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
):
    """
    Split data into training and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
