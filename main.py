import sys
from src import data_preparation, model_training, model_prediction

if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare_data":
        train, test = data_preparation.load_and_prepare_data(
            "data/raw/train.csv",
            "data/raw/test.csv",
            "data/raw/stores.csv",
            "data/raw/transactions.csv",
            "data/raw/oil.csv",
            "data/raw/holidays.csv"
        )
        train.to_csv("data/processed/train.csv", index=False)
        test.to_csv("data/processed/test.csv", index=False)
        print("Data preparation completed.")
    elif action == "train_model":
        train = pd.read_csv("data/processed/train.csv")
        model_training.train_model(train)
        print("Model training completed.")
    elif action == "predict":
        predictions = model_prediction.predict("data/processed/test.csv")
        print("Predictions:", predictions)
    elif action == "test":
        import pytest
        pytest.main(["tests/"])
    else:
        print("Unknown action. Use 'prepare_data', 'train_model', 'predict', or 'test'.")
