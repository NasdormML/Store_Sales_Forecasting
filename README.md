# Store Sales - Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.11+-brightgreen)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.1-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.1.0-red)
![Optuna](https://img.shields.io/badge/Optuna-v3.0.0-orange)

## Overview

This project uses **time series forecasting** techniques to predict future sales for various stores. It leverages advanced machine learning models and optimizes them using **Optuna** to achieve the best possible predictions. Key features include data preprocessing, feature engineering, and model optimization.

## Table of Contents

- [Project Structure](#project-structure)
- [Data Preparation and Feature Engineering](#data-preparation-and-feature-engineering)
- [Modeling](#modeling)
- [Hyperparameter Optimization](#hyperparameter-optimization-with-optuna)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [License](#license)

## Key Features

- **Store ID**: Unique identification for each store.
- **Date**: The time component used for forecasting.
- **Sales**: The target variable representing store sales, which we aim to predict.
  
Comprehensive **exploratory data analysis (EDA)** and preprocessing steps are included in the notebook.

---

## Project Structure

```bash
├── store-sales-time-series-forecasting/    # Datasets used in the project
├── notebooks/store_sales_kaggle.ipynb      # Jupyter notebooks with EDA and modeling
├── README.md                               # Project overview and setup
└── requirements.txt                        # Dependencies and libraries
```

---

## Data Preparation and Feature Engineering

1. **Data Loading**: Loading of multiple datasets, including:
    - `holidays_events.csv`
    - `oil.csv`
    - `stores.csv`
    - `transactions.csv`
    - `train.csv` and `test.csv`

   ```python
   df_holidays = pd.read_csv(comp_dir / "holidays_events.csv",parse_dates=['date'])
   df_oil = pd.read_csv(comp_dir /'oil.csv', parse_dates=['date'])
   df_stores = pd.read_csv(comp_dir / 'stores.csv')
   df_trans = pd.read_csv(comp_dir / 'transactions.csv', parse_dates=['date'])
   df_train = pd.read_csv(comp_dir / 'train.csv', parse_dates=['date'])
   df_test = pd.read_csv(comp_dir / 'test.csv', parse_dates=['date'])
   ```

2. **Exploratory Data Analysis (EDA)**:
   - Analyzing the impact of oil prices and sales trends using **line plots** and **box plots**.
   - Monthly sales analysis and identifying trends using `seaborn`.

   ```python
   sns.lineplot(data=df_oil, x='date', y='dcoilwtico')
   plt.title('Oil Prices Over Time')
   plt.show()
   ```

3. **Feature Engineering**:
   - Extracting key time-based features (`year`, `month`, `day_of_week`, `week_of_year`).
   - Creating lag and rolling average features to capture temporal dependencies.

   ```python
   train['lag_7_sales'] = train['sales'].shift(7)
   train['lag_14_sales'] = train['sales'].shift(14)
   train['rolling_mean_7'] = train['sales'].shift(1).rolling(window=7).mean()
   train['rolling_mean_14'] = train['sales'].shift(1).rolling(window=14).mean()

   train['log_sales'] = np.log1p(train['sales'])
   ```

4. **Data Processing**: Target encoding, missing value imputation, and feature scaling using **scikit-learn** pipelines.

   ```python
   target_encoder = ce.TargetEncoder(cols=categorical_features)

   numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
   ])
   preprocessor = ColumnTransformer(
       transformers=[
           ('num', numerical_transformer, numerical_features),
           ('cat', 'passthrough', categorical_features)
            ]
        )
   
   ```

---

## Modeling

The primary model is **XGBoost**, chosen for its efficiency and support for GPU acceleration. The model is trained using **TimeSeriesSplit** for temporal cross-validation, followed by hyperparameter tuning with Optuna.

```python
        model = xgb.train(
            param, 
            dtrain,
            num_boost_round=trial.suggest_int('n_estimators', 100, 1000),
            evals=[(dtest, 'validation')],
            early_stopping_rounds=15,
            verbose_eval=False,
        )
```

---

## Hyperparameter Optimization with Optuna

**Optuna** is used to automatically optimize the model's hyperparameters, improving performance and reducing error metrics like RLMSE.

### Example of optimized hyperparameters:

```bash
Best RLMSE: 0.75094

Best hyperparameters:
{
    'n_estimators': 992,
    'max_depth': 12,
    'learning_rate': 0.0083,
    'subsample': 0.7746,
    'colsample_bytree': 0.8047,
    'min_child_weight': 6,
    'reg_alpha': 3.62,
    'reg_lambda': 4.25
}
```

#### Visualization of Optimization Process:

Optuna provides several visualization tools:
- **Optimization History**
- **Hyperparameter Importance**

```python
vis.plot_optimization_history(study)
vis.plot_param_importances(study)
```

---

## Getting Started

### Prerequisites

Ensure that the following libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `optuna`
- `matplotlib`
- `seaborn`

You can install them using:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn optuna
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/NasdormML/Time_Series.git
cd Time_Series
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Build Docker image:
   ```bash
   docker build -t time_series_image .
   ```

4. Launch the port spread container for Jupyter:
   ```bash
   docker run -it --name TS_container -p 8888:8888 time_series_image
   ```

5. Open Jupyter Notebook in browser:
   - Run this: `http://localhost:8888`
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
