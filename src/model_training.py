import xgboost as xgb
import category_encoders as ce
import os
import joblib
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_model(train, categorical_features, numerical_features):
    # Правильный BASE_DIR для работы с models
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    HYPERPARAMS_PATH = os.path.join(BASE_DIR, "src", "hyperparameters.json")

    with open(HYPERPARAMS_PATH, "r") as f:
        params = json.load(f)

    # Обработка данных
    X = train.drop(columns=['sales', 'log_sales'])
    y = train['log_sales']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    target_encoder = ce.TargetEncoder(cols=categorical_features)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_val_encoded = target_encoder.transform(X_val)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train_encoded)
    X_val_processed = preprocessor.transform(X_val_encoded)

    dtrain = xgb.DMatrix(X_train_processed, label=y_train)
    dval = xgb.DMatrix(X_val_processed, label=y_val)

    # Обучение модели
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get("num_boost_round", 100),
        evals=[(dval, 'validation')],
        early_stopping_rounds=15
    )

    # Сохранение модели
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "trained_model.pkl")
    joblib.dump((model, target_encoder, preprocessor), model_path)

    assert os.path.exists(model_path), "Model file was not saved successfully."
    return model


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Путь к обработанным данным
    train_data_path = os.path.join(BASE_DIR, "data", "processed", "train.csv")

    # Загрузка данных
    train_data = pd.read_csv(train_data_path)

    categorical_features = ['holiday_type', 'locale', 'locale_name', 'store_nbr', 'family', 'city', 'state', 'cluster']
    numerical_features = ['onpromotion', 'transactions', 'oil_price', 'lag_7_sales', 'lag_14_sales', 'rolling_mean_7', 'rolling_mean_14']

    # Обучение модели
    train_model(train_data, categorical_features, numerical_features)
