from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor():
    """
    Build the feature preprocessing step.

    California Housing contains numeric features, so we use
    StandardScaler for now.
    """
    return StandardScaler()


def transform_features(preprocessor, X_train, X_test):
    """
    Fit preprocessing on training data and transform both datasets.
    """
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed
