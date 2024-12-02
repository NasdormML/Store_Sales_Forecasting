import pytest
import numpy as np
import pandas as pd 
from src.model_prediction import predict

@pytest.fixture
def test_data_path():
    return "data/processed/test.csv"

@pytest.fixture(scope="module", autouse=True)
def ensure_model_trained():
    from src.model_training import train_model
    train_data = pd.read_csv("data/processed/train.csv")
    categorical_features = ['holiday_type', 'locale', 'locale_name', 'store_nbr', 'family', 'city', 'state', 'cluster']
    numerical_features = ['onpromotion', 'transactions', 'oil_price', 'lag_7_sales', 'lag_14_sales', 'rolling_mean_7', 'rolling_mean_14']
    
    # Тренируем модель
    import os
    if not os.path.exists("models/trained_model.pkl"):
        train_model(train_data, categorical_features, numerical_features)

def test_predict(test_data_path):
    # Прогнозируем
    predictions = predict(test_data_path)
    
    # Проверки
    assert isinstance(predictions, np.ndarray), "Predictions are not in the correct format"
    assert len(predictions) > 0, "No predictions were generated"
    assert predictions.ndim == 1, "Predictions should be a 1-dimensional array"
