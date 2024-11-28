import joblib
import xgboost as xgb
import pandas as pd

# Загрузка модели и пайплайна
model_path = "notebook_output/trained_xgb.pkl"
final_model, final_target_encoder, final_preprocessor = joblib.load(model_path)

test_df = pd.read_csv('../store-sales-time-series-forecasting/test.csv', parse_dates=['date'])
oil_df = pd.read_csv('../store-sales-time-series-forecasting/oil.csv', parse_dates=['date'])
holidays_df = pd.read_csv('../store-sales-time-series-forecasting/holidays_events.csv', parse_dates=['date'])
transactions_df = pd.read_csv('../store-sales-time-series-forecasting/transactions.csv', parse_dates=['date'])
stores_df = pd.read_csv('../store-sales-time-series-forecasting/stores.csv')

test = test_df.merge(stores_df, on='store_nbr', how='left')
test = test.merge(transactions_df, on=['date', 'store_nbr'], how='left')
test = test.merge(oil_df, on='date', how='left')
test = test.merge(holidays_df, on='date', how='left')

test['oil_price'] = test['dcoilwtico'].ffill().bfill()
test['holiday_type'] = test['type_y'].fillna('No Holiday')
test['transactions'] = test['transactions'].ffill()
test.drop(columns=['type_y'], inplace=True)

test['store_nbr'] = test['store_nbr'].astype(str)
test['cluster'] = test['cluster'].astype(str)


test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day_of_week'] = test['date'].dt.dayofweek
test['week_of_year'] = test['date'].dt.isocalendar().week
test['is_weekend'] = test['day_of_week'].isin([5, 6]).astype(int)
test['is_holiday'] = (test['holiday_type'] != 'Work Day').astype(int)

# Лаговые и скользящие средние признаки в test
last_lag_7_sales = train['lag_7_sales'].iloc[-1]
last_lag_14_sales = train['lag_14_sales'].iloc[-1]
last_rolling_mean_7 = train['rolling_mean_7'].iloc[-1]
last_rolling_mean_14 = train['rolling_mean_14'].iloc[-1]

test['lag_7_sales'] = last_lag_7_sales
test['lag_14_sales'] = last_lag_14_sales
test['rolling_mean_7'] = last_rolling_mean_7
test['rolling_mean_14'] = last_rolling_mean_14

test = test.dropna(subset=['lag_7_sales', 'lag_14_sales', 'rolling_mean_7', 'rolling_mean_14'])

# Предобработка данных
test_encoded = final_target_encoder.transform(test)
test_processed = final_preprocessor.transform(test_encoded)
dtest_new = xgb.DMatrix(test_processed)

# Предсказание
predictions = final_model.predict(dtest_new)
print(predictions)
