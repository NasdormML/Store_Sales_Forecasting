import pytest
import pandas as pd
from src.model_training import train_model

@pytest.fixture
def processed_train_data():
    return pd.read_csv("/data/processed/train.csv")

def test_train_model(processed_train_data):
    categorical_features = ['holiday_type', 'locale', 'locale_name', 'store_nbr', 'family', 'city', 'state', 'cluster']
    numerical_features = ['onpromotion', 'transactions', 'oil_price', 'lag_7_sales', 'lag_14_sales', 'rolling_mean_7', 'rolling_mean_14']

    model = train_model(processed_train_data, categorical_features, numerical_features)
    assert model is not None, "Model training failed"
