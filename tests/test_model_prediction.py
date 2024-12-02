import pytest
import numpy as np
from src.model_prediction import predict

@pytest.fixture
def test_data_path():
    return "/data/processed/test.csv"

def test_predict(test_data_path):
    predictions = predict(test_data_path)
    assert isinstance(predictions, np.ndarray), "Predictions are not in the correct format"
    assert len(predictions) > 0, "No predictions were generated"
