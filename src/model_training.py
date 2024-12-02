import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import os
import joblib

def train_model(train, categorical_features, numerical_features):
    # Разделение данных
    X = train.drop(columns=['sales', 'log_sales'])
    y = train['log_sales']

    # Подготовка данных
    target_encoder = ce.TargetEncoder(cols=categorical_features)
    X_encoded = target_encoder.fit_transform(X, y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X_encoded)

    # Обучение модели
    dtrain = xgb.DMatrix(X_processed, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_estimators': 500
    }
    model = xgb.train(params, dtrain)

    # Создание папки models, если её нет
    if not os.path.exists('models'):
        os.makedirs('models')

    # Сохранение модели
    model_path = os.path.join(os.getcwd(), "models", "trained_model.pkl")
    joblib.dump((model, target_encoder, preprocessor), model_path)
    return model

if __name__ == "__main__":
    import pandas as pd
    train_data = pd.read_csv("data/processed/train.csv")
    categorical_features = ['holiday_type', 'locale', 'locale_name', 'store_nbr', 'family', 'city', 'state', 'cluster']
    numerical_features = ['onpromotion', 'transactions', 'oil_price', 'lag_7_sales', 'lag_14_sales', 'rolling_mean_7', 'rolling_mean_14']
    
    train_model(train_data, categorical_features, numerical_features)
