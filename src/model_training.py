import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import os
import joblib
from sklearn.model_selection import train_test_split
import json

def train_model(train, categorical_features, numerical_features):
    with open("hyperparameters.json", "r") as f:
        params = json.load(f)

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

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get('n_estimators', 500),
        evals=[(dval, 'validation')],
        early_stopping_rounds=15
    )

    if not os.path.exists('models'):
        os.makedirs('models')

    model_path = os.path.join(os.getcwd(), "models", "trained_model.pkl")
    joblib.dump((model, target_encoder, preprocessor), model_path)
    return model

if __name__ == "__main__":
    import pandas as pd

    train_data = pd.read_csv("data/processed/train.csv")
    categorical_features = ['holiday_type', 'locale', 'locale_name', 'store_nbr', 'family', 'city', 'state', 'cluster']
    numerical_features = ['onpromotion', 'transactions', 'oil_price', 'lag_7_sales', 'lag_14_sales', 'rolling_mean_7', 'rolling_mean_14']

    train_model(train_data, categorical_features, numerical_features)