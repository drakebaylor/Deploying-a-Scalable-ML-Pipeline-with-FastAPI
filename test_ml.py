import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics
from ml.data import process_data
import pandas as pd


# Sample fixture to load and preprocess data once for reuse
@pytest.fixture(scope="module")
def processed_data():
    data = pd.read_csv("data/census.csv")
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    train, _ = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train


def test_model_training_returns_random_forest(processed_data):
    """
    Test that train_model returns an instance of RandomForestClassifier.
    """
    X_train, y_train = processed_data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not RandomForestClassifier"


def test_compute_metrics_returns_expected_format():
    """
    Test that compute_model_metrics returns three float values between 0 and 1.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    for metric in (precision, recall, fbeta):
        assert isinstance(metric, float), "Metric is not a float"
        assert 0.0 <= metric <= 1.0, "Metric is out of bounds"


def test_training_data_has_correct_shape(processed_data):
    """
    Test that processed training data has matching sample size between X and y.
    """
    X_train, y_train = processed_data
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in X and y rows"
    assert len(y_train.shape) == 1, "y should be 1D"
    assert isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray), "Wrong data types"

