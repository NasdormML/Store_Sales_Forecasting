import joblib
import os

MODEL_PATH = "models/trained_xgb.pkl"  # Укажите путь к модели

def test_model_exists():
    """Проверяем, что файл модели существует."""
    assert os.path.exists(MODEL_PATH), f"Файл модели {MODEL_PATH} не найден!"

def test_model_load():
    """Проверяем, что модель загружается без ошибок."""
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Модель не имеет метода predict!"
