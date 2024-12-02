import pytest
import pandas as pd
from src.data_preparation import load_and_prepare_data

@pytest.fixture
def raw_data_paths():
    return {
        "train_path": "data/raw/train.csv",
        "test_path": "data/raw/test.csv",
        "stores_path": "data/raw/stores.csv",
        "transactions_path": "data/raw/transactions.csv",
        "oil_path": "data/raw/oil.csv",
        "holidays_path": "data/raw/holidays.csv"
    }

def test_load_and_prepare_data(raw_data_paths):
    train, test = load_and_prepare_data(**raw_data_paths)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert not train.empty, "Train data is empty after preparation"
    assert not test.empty, "Test data is empty after preparation"
    assert "oil_price" in train.columns, "oil_price column is missing"
    assert "holiday_type" in train.columns, "holiday_type column is missing"
