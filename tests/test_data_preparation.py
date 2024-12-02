import pytest
import pandas as pd
from src.data_preparation import load_and_prepare_data

def test_load_and_prepare_data():
    train, test = load_and_prepare_data()

    # Проверяем, что данные загружаются и обрабатываются корректно
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert not train.empty, "Train data is empty after preparation"
    assert not test.empty, "Test data is empty after preparation"

    # Проверяем наличие ключевых колонок
    for col in ["oil_price", "holiday_type", "lag_7_sales", "rolling_mean_7"]:
        assert col in train.columns, f"{col} column is missing in train dataset"
        assert col in test.columns, f"{col} column is missing in test dataset"
