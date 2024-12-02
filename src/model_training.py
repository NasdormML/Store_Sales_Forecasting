import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

    # Сохранение
    joblib.dump((model, target_encoder, preprocessor), "models/trained_model.pkl")
    return model
