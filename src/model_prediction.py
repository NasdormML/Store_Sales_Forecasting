import pandas as pd
import joblib
import xgboost as xgb

def predict(test_data_path):
    # Загрузка данных
    test = pd.read_csv(test_data_path)
    
    # Загрузка модели
    model, target_encoder, preprocessor = joblib.load("models/trained_model.pkl")

    # Подготовка данных
    X_test_encoded = target_encoder.transform(test)
    X_test_processed = preprocessor.transform(X_test_encoded)
    
    dtest = xgb.DMatrix(X_test_processed)
    
    # Предсказания
    predictions = model.predict(dtest)
    return predictions

if __name__ == "__main__":
    predict()