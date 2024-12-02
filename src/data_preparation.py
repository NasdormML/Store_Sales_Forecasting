import os
import pandas as pd
import numpy as np

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Загрузка данных
train_df = pd.read_csv(f"{RAW_DATA_DIR}/train.csv", parse_dates=["date"])
test_df = pd.read_csv(f"{RAW_DATA_DIR}/test.csv", parse_dates=["date"])
stores_df = pd.read_csv(f"{RAW_DATA_DIR}/stores.csv")
transactions_df = pd.read_csv(f"{RAW_DATA_DIR}/transactions.csv", parse_dates=["date"])
oil_df = pd.read_csv(f"{RAW_DATA_DIR}/oil.csv", parse_dates=["date"])
holidays_df = pd.read_csv(f"{RAW_DATA_DIR}/holidays.csv", parse_dates=["date"])

# Объединение данных
train = train_df.merge(stores_df, on="store_nbr", how="left")
train = train.merge(transactions_df, on=["date", "store_nbr"], how="left")
train = train.merge(oil_df, on="date", how="left")
train = train.merge(holidays_df, on="date", how="left")

test = test_df.merge(stores_df, on="store_nbr", how="left")
test = test.merge(transactions_df, on=["date", "store_nbr"], how="left")
test = test.merge(oil_df, on="date", how="left")
test = test.merge(holidays_df, on="date", how="left")

# Заполнение пропусков
train["oil_price"] = train["dcoilwtico"].ffill().bfill()
train["holiday_type"] = train["type_y"].fillna("No Holiday")
train["transactions"] = train["transactions"].ffill()
train.drop(columns=["type_y"], inplace=True)

test["oil_price"] = test["dcoilwtico"].ffill().bfill()
test["holiday_type"] = test["type_y"].fillna("No Holiday")
test["transactions"] = test["transactions"].ffill()
test.drop(columns=["type_y"], inplace=True)

# Преобразование типов
train["store_nbr"] = train["store_nbr"].astype(str)
train["cluster"] = train["cluster"].astype(str)

test["store_nbr"] = test["store_nbr"].astype(str)
test["cluster"] = test["cluster"].astype(str)

# Создание временных признаков
def create_date_features(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"] = (df["holiday_type"] != "Work Day").astype(int)
    return df

train = create_date_features(train)
test = create_date_features(test)

# Лаговые признаки и скользящие средние для train
train["lag_7_sales"] = train["sales"].shift(7)
train["lag_14_sales"] = train["sales"].shift(14)
train["rolling_mean_7"] = train["sales"].shift(1).rolling(window=7).mean()
train["rolling_mean_14"] = train["sales"].shift(1).rolling(window=14).mean()

# Удаление строк с пропущенными значениями
train = train.dropna(subset=["lag_7_sales", "lag_14_sales", "rolling_mean_7", "rolling_mean_14"])

# Лог-трансформация целевой переменной
train["log_sales"] = np.log1p(train["sales"])

# Лаговые признаки и скользящие средние для test
last_lag_7_sales = train["lag_7_sales"].iloc[-1]
last_lag_14_sales = train["lag_14_sales"].iloc[-1]
last_rolling_mean_7 = train["rolling_mean_7"].iloc[-1]
last_rolling_mean_14 = train["rolling_mean_14"].iloc[-1]

test["lag_7_sales"] = last_lag_7_sales
test["lag_14_sales"] = last_lag_14_sales
test["rolling_mean_7"] = last_rolling_mean_7
test["rolling_mean_14"] = last_rolling_mean_14

# Удаление строк с пропущенными значениями
test = test.dropna(subset=["lag_7_sales", "lag_14_sales", "rolling_mean_7", "rolling_mean_14"])

# Сохранение обработанных данных
train.to_csv(f"{PROCESSED_DATA_DIR}/train.csv", index=False)
test.to_csv(f"{PROCESSED_DATA_DIR}/test.csv", index=False)

print(f"Train and test datasets are processed and saved to {PROCESSED_DATA_DIR}.")
