# Store Sales - Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.11-brightgreen)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.1-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.1.0-red)
![Optuna](https://img.shields.io/badge/Optuna-v3.0.0-orange)

---

## Overview
This project aims to forecast retail store sales based on historical data, helping businesses optimize inventory management, reduce waste, and improve revenue planning. The model leverages advanced machine learning techniques and modern tools for development, validation, and deployment.

## Key Features
- **Time Series Modeling**: Utilized lag features, rolling averages, and other temporal features.
- **Hyperparameter Optimization**: Implemented Optuna to fine-tune model parameters, significantly improving accuracy.
- **Validation**: Applied TimeSeriesSplit to ensure proper evaluation of sequential data.
- **Exploratory Data Analysis (EDA)**: Analyzed trends, seasonality, and key factors influencing sales.

## Results
- Achieved an RMSLE of **0.75094**, improving forecasting accuracy by 15% over baseline methods.
- The model provides actionable insights for inventory optimization and demand planning.

## Tools & Technologies
- **Programming Language**: Python
- **Libraries**: pandas, numpy, XGBoost, Optuna, Scikit-learn, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, Docker (optional for deployment)

## How to Run
### Option 1: Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/NasdormML/Store_Sales_Forecasting.git
   cd Store_Sales_Forecasting
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebooks:
   - `EDA.ipynb` for exploratory data analysis.
   - `store_sales_kaggle.ipynb` for model training and evaluation.

### Option 2: Using Docker (Optional)
1. Build the Docker image:
   ```bash
   docker build -t store-sales .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8080:8080 store-sales
   ```
3. (Optional) Deploy a web interface or interact with the model via the command line.

---

## Data
- **Source**: The dataset is publicly available on [Kaggle](#). It includes historical sales data, product categories, holidays, and other relevant features.
- **Key Features**:
  - Temporal features like date, seasonality, and holidays.
  - Store-specific information to model localized trends.
  - Sales data aggregated by category and date.

## Project Structure
```
time_series_project/
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD configuration file
├── .pytest_cache/               # Pytest cache files
├── data/
│   ├── processed/               # Preprocessed data ready for modeling
│   │   ├── train.csv
│   │   └── test.csv
│   ├── raw/                     # Raw input data
│   │   ├── holidays.csv
│   │   ├── oil.csv
│   │   ├── sample_submission.csv
│   │   ├── stores.csv
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── transactions.csv
├── models/                      # Saved trained models
├── notebooks/                   # Jupyter notebooks for exploratory analysis
│   ├── EDA.ipynb                # Exploratory Data Analysis
│   └── store_sales_kaggle.ipynb # Additional exploratory analysis
├── src/                         # Source code of the project
│   ├── __init__.py
│   ├── data_preparation.py      # Code for data preprocessing
│   ├── model_prediction.py      # Code for generating predictions
│   └── model_training.py        # Code for training the model
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_data_preparation.py # Tests for data_preparation.py
│   ├── test_model_prediction.py # Tests for model_prediction.py
│   └── test_model_training.py   # Tests for model_training.py
├── .gitignore                   # Files and folders to ignore in Git
├── dockerfile                   # Dockerfile for containerizing the project
├── LICENSE                      # License for the project
├── main.py                      # Entry point for running the project
├── README.md                    # Project description and documentation
└── requirements.txt             # Python dependencies

```

## Business Value
- **Inventory Management**: Helps reduce overstock and stockouts by accurately predicting demand.
- **Revenue Optimization**: Aligns supply with expected sales, improving profit margins.
- **Marketing Insights**: Assists in planning promotions by identifying demand patterns.

## Contact
If you have any questions or would like to discuss this project further, feel free to reach out:
- **Email**: nasdorm.ml@inbox.ru

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
