from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_california_housing() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset.

    Returns:
        X: feature dataframe
        y: target series
    """
    data = fetch_california_housing(as_frame=True)

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
