import pandas as pd
import numpy as np

def load_and_prepare_data(train_path, test_path, stores_path, transactions_path, oil_path, holidays_path):
    # Загрузка данных
    train = pd.read_csv(train_path, parse_dates=["date"])
    test = pd.read_csv(test_path, parse_dates=["date"])
    stores = pd.read_csv(stores_path)
    transactions = pd.read_csv(transactions_path)
    oil = pd.read_csv(oil_path)
    holidays = pd.read_csv(holidays_path)
    
    # Объединение данных
    for df in [train, test]:
        df.merge(stores, on='store_nbr', how='left')
        df.merge(transactions, on=['date', 'store_nbr'], how='left')
        df.merge(oil, on='date', how='left')
        df.merge(holidays, on='date', how='left')
    
    # Заполнение пропусков
    for df in [train, test]:
        df['oil_price'] = df['dcoilwtico'].ffill().bfill()
        df['holiday_type'] = df['type_y'].fillna('No Holiday')
        df['transactions'] = df['transactions'].ffill()
        df.drop(columns=['type_y'], inplace=True)
        
    return train, test
